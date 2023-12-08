"""LLM Compiler Planner"""
import asyncio
import re
from typing import Any, Optional, Sequence, Union
from uuid import UUID

from src.executors.schema import Plan
from src.llm_compiler.constants import END_OF_PLAN
from src.llm_compiler.output_parser import (
    ACTION_PATTERN,
    THOUGHT_PATTERN,
    LLMCompilerPlanParser,
    instantiate_task,
)
from src.llm_compiler.task_fetching_unit import Task
from src.tools.base import StructuredTool, Tool
from src.utils.logger_utils import log

from langchain.callbacks.base import AsyncCallbackHandler, Callbacks
from langchain.chat_models.base import BaseChatModel
from langchain.schema import LLMResult
from langchain.schema.messages import HumanMessage, SystemMessage

JOIN_DESCRIPTION = (
    "join():\n"
    " - Collects and combines results from prior actions.\n"
    " - A LLM agent is called upon invoking join to either finalize the user query or wait until the plans are executed.\n"
    " - join should always be the last action in the plan, and will be called in two scenarios:\n"
    "   (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.\n"
    "   (b) if the answer cannot be determined in the planning phase before you execute the plans. "
)


def generate_llm_compiler_prompt(
    tools: Sequence[Union[Tool, StructuredTool]],
    example_prompt=str,
    is_replan: bool = False,
):
    prefix = (
        "Given a user query, create a plan to solve it with the utmost parallelizability. "
        f"Each plan should comprise an action from the following {len(tools) + 1} types:\n"
    )

    # Tools
    for i, tool in enumerate(tools):
        prefix += f"{i+1}. {tool.description}\n"

    # Join operation
    prefix += f"{i+2}. {JOIN_DESCRIPTION}\n\n"

    # Guidelines
    prefix += (
        "Guidelines:\n"
        " - Each action described above contains input/output types and description.\n"
        "    - You must strictly adhere to the input and output types for each action.\n"
        "    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.\n"
        " - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.\n"
        " - Each action MUST have a unique ID, which is strictly increasing.\n"
        " - Inputs for actions can either be constants or outputs from preceding actions. "
        "In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.\n"
        f" - Always call join as the last action in the plan. Say '{END_OF_PLAN}' after you call join\n"
        " - Ensure the plan maximizes parallelizability.\n"
        " - Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.\n"
        " - Never explain the plan with comments (e.g. #).\n"
        " - Never introduce new actions other than the ones provided.\n\n"
    )

    if is_replan:
        prefix += (
            ' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        )

    # Examples
    prefix += "Here are some examples:\n\n"
    prefix += example_prompt

    return prefix


class StreamingGraphParser:
    """Streaming version of the GraphParser."""

    buffer = ""
    thought = ""
    graph_dict = {}

    def __init__(self, tools: Sequence[Union[Tool, StructuredTool]]) -> None:
        self.tools = tools

    def _match_buffer_and_generate_task(self, suffix: str) -> Optional[Task]:
        """Runs every time "\n" is encountered in the input stream or at the end of the stream.
        Matches the buffer against the regex patterns and generates a task if a match is found.
        Match patterns include:
        1. Thought: <thought>
          - this case, the thought is stored in self.thought, and we reset the buffer.
          - the thought is then used as the thought for the next action.
        2. <idx>. <tool_name>(<args>)
          - this case, the tool is instantiated with the idx, tool_name, args, and thought.
          - the thought is reset.
          - the buffer is reset.
        """
        if match := re.match(THOUGHT_PATTERN, self.buffer):
            # Optionally, action can be preceded by a thought
            self.thought = match.group(1)
            self.buffer = suffix
        elif match := re.match(ACTION_PATTERN, self.buffer):
            # if action is parsed, return the task, and clear the buffer
            idx, tool_name, args, _ = match.groups()
            idx = int(idx)
            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=self.thought,
            )
            self.buffer = suffix
            self.thought = ""
            return task
        return None

    def ingest_token(self, token: str) -> Optional[Task]:
        # Append token to buffer
        if "\n" in token:
            prefix, suffix = token.split("\n", 1)
            prefix = prefix.strip()
            self.buffer += prefix + "\n"
            return self._match_buffer_and_generate_task(suffix)
        else:
            self.buffer += token

        return None

    def finalize(self):
        self.buffer = self.buffer + "\n"
        return self._match_buffer_and_generate_task("")


class LLMCompilerCallback(AsyncCallbackHandler):
    _queue: asyncio.Queue[Optional[Task]]
    _parser: StreamingGraphParser
    _tools: Sequence[Union[Tool, StructuredTool]]

    def __init__(
        self,
        queue: asyncio.Queue[Optional[str]],
        tools: Sequence[Union[Tool, StructuredTool]],
    ):
        self._queue = queue
        self._parser = StreamingGraphParser(tools=tools)

    async def on_llm_start(self, serialized, prompts, **kwargs: Any) -> Any:
        """Run when LLM starts running."""

    async def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        parsed_data = self._parser.ingest_token(token)
        if parsed_data:
            await self._queue.put(parsed_data)
            if parsed_data.is_join:
                await self._queue.put(None)

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        parsed_data = self._parser.finalize()
        if parsed_data:
            await self._queue.put(parsed_data)
        await self._queue.put(None)


class Planner:
    def __init__(
        self,
        llm: BaseChatModel,
        example_prompt: str,
        example_prompt_replan: str,
        tools: Sequence[Union[Tool, StructuredTool]],
        stop: Optional[list[str]],
    ):
        self.llm = llm
        # different system prompt is needed when replanning
        # since they have different guidelines, and also examples provided by the user
        self.system_prompt = generate_llm_compiler_prompt(
            tools=tools,
            example_prompt=example_prompt,
            is_replan=False,
        )
        self.system_prompt_replan = generate_llm_compiler_prompt(
            tools=tools,
            example_prompt=example_prompt_replan,
            is_replan=True,
        )
        self.tools = tools
        self.output_parser = LLMCompilerPlanParser(tools=tools)
        self.stop = stop

    async def run_llm(
        self,
        inputs: dict[str, Any],
        is_replan: bool = False,
        callbacks: Callbacks = None,
    ) -> str:
        """Run the LLM."""
        if is_replan:
            system_prompt = self.system_prompt_replan
            assert "context" in inputs, "If replanning, context must be provided"
            human_prompt = f"Question: {inputs['input']}\n{inputs['context']}\n"
        else:
            system_prompt = self.system_prompt
            human_prompt = f"Question: {inputs['input']}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        llm_response = await self.llm._call_async(
            messages,
            callbacks=callbacks,
            stop=self.stop,
        )
        log("LLMCompiler planner response: \n", llm_response.content, block=True)

        return llm_response.content

    async def plan(
        self, inputs: dict, is_replan: bool, callbacks: Callbacks = None, **kwargs: Any
    ):
        llm_response = await self.run_llm(
            inputs=inputs, is_replan=is_replan, callbacks=callbacks
        )
        llm_response = llm_response + "\n"
        return self.output_parser.parse(llm_response)

    async def aplan(
        self,
        inputs: dict,
        task_queue: asyncio.Queue[Optional[str]],
        is_replan: bool,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Plan:
        """Given input, asynchronously decide what to do."""
        all_callbacks = [
            LLMCompilerCallback(
                queue=task_queue,
                tools=self.tools,
            )
        ]
        if callbacks:
            all_callbacks.extend(callbacks)
        await self.run_llm(inputs=inputs, is_replan=is_replan, callbacks=all_callbacks)
