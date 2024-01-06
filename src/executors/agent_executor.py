"""Chain that takes in an input and produces an action and action input."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.agents.tools import InvalidTool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)
from langchain.pydantic_v1 import root_validator
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.tools import BaseTool
from langchain.utilities.asyncio import asyncio_timeout
from langchain.utils.input import get_color_mapping

from src.chains.chain import Chain
from src.tools.base import BaseTool
from src.utils.logger_utils import log

logger = logging.getLogger(__name__)


class ExceptionTool(BaseTool):
    """Tool that just returns the query."""

    name: str = "_Exception"
    """Name of the tool."""
    description: str = "Exception tool"
    """Description of the tool."""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return query

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return query


class AgentExecutor(Chain):
    """Agent that is using tools."""

    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    """The agent to run for creating a plan and determining actions
    to take at each step of the execution loop."""
    tools: Sequence[BaseTool]
    """The valid tools the agent can call."""
    return_intermediate_steps: bool = False
    """Whether to return the agent's trajectory of intermediate steps
    at the end in addition to the final output."""
    max_iterations: Optional[int] = 15
    """The maximum number of steps to take before ending the execution
    loop.

    Setting to 'None' could lead to an infinite loop."""
    max_execution_time: Optional[float] = None
    """The maximum amount of wall clock time to spend in the execution
    loop.
    """
    early_stopping_method: str = "force"
    """The method to use for early stopping if the agent never
    returns `AgentFinish`. Either 'force' or 'generate'.

    `"force"` returns a string saying that it stopped because it met a
        time or iteration limit.

    `"generate"` calls the agent's LLM Chain one final time to generate
        a final answer based on the previous steps.
    """
    handle_parsing_errors: Union[
        bool, str, Callable[[OutputParserException], str]
    ] = False
    """How to handle errors raised by the agent's output parser.
    Defaults to `False`, which raises the error.
s
    If `true`, the error will be sent back to the LLM as an observation.
    If a string, the string itself will be sent to the LLM as an observation.
    If a callable function, the function will be called with the exception
     as an argument, and the result of that function will be passed to the agent
      as an observation.
    """
    trim_intermediate_steps: Union[
        int, Callable[[List[Tuple[AgentAction, str]]], List[Tuple[AgentAction, str]]]
    ] = -1

    @classmethod
    def from_agent_and_tools(
        cls,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        tools: Sequence[BaseTool],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> AgentExecutor:
        """Create from agent and tools."""
        return cls(
            agent=agent,
            tools=tools,
            callbacks=callbacks,
            **kwargs,
        )

    @root_validator()
    def validate_tools(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
        agent = values["agent"]
        tools = values["tools"]
        allowed_tools = agent.get_allowed_tools()
        if allowed_tools is not None:
            if set(allowed_tools) != set([tool.name for tool in tools]):
                raise ValueError(
                    f"Allowed tools ({allowed_tools}) different than "
                    f"provided tools ({[tool.name for tool in tools]})"
                )
        return values

    @root_validator()
    def validate_return_direct_tool(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
        agent = values["agent"]
        tools = values["tools"]
        if isinstance(agent, BaseMultiActionAgent):
            for tool in tools:
                if tool.return_direct:
                    raise ValueError(
                        "Tools that have `return_direct=True` are not allowed "
                        "in multi-action agents"
                    )
        return values

    def save(self, file_path: Union[Path, str]) -> None:
        """Raise error - saving not supported for Agent Executors."""
        raise ValueError(
            "Saving not supported for agent executors. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    def save_agent(self, file_path: Union[Path, str]) -> None:
        """Save the underlying agent."""
        return self.agent.save(file_path)

    def iter(
        self,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        include_run_info: bool = False,
        async_: bool = False,
    ) -> AgentExecutorIterator:
        """Enables iteration over steps taken to reach final output."""
        return AgentExecutorIterator(
            self,
            inputs,
            callbacks,
            tags=self.tags,
            include_run_info=include_run_info,
            async_=async_,
        )

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    def lookup_tool(self, name: str) -> BaseTool:
        """Lookup tool by name."""
        return {tool.name: tool for tool in self.tools}[name]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            if run_manager:
                run_manager.on_agent_action(output, color="green")
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output
        result = []
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    {
                        "requested_tool_name": agent_action.tool,
                        "available_tool_names": list(name_to_tool_map.keys()),
                    },
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result

    async def _atake_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = await self.agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )
        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise e
            text = str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = str(e.observation)
                    text = str(e.llm_output)
                else:
                    observation = "Invalid or incomplete response"
            elif isinstance(self.handle_parsing_errors, str):
                observation = self.handle_parsing_errors
            elif callable(self.handle_parsing_errors):
                observation = self.handle_parsing_errors(e)
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, text)
            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = await ExceptionTool().arun(
                output.tool_input,
                verbose=self.verbose,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )
            return [(output, observation)]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        actions: List[AgentAction]
        if isinstance(output, AgentAction):
            actions = [output]
        else:
            actions = output

        async def _aperform_agent_action(
            agent_action: AgentAction,
        ) -> Tuple[AgentAction, str]:
            if run_manager:
                await run_manager.on_agent_action(
                    agent_action, verbose=self.verbose, color="green"
                )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = await tool.arun(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = await InvalidTool().arun(
                    {
                        "requested_tool_name": agent_action.tool,
                        "available_tool_names": list(name_to_tool_map.keys()),
                    },
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )
            return agent_action, observation

        # Use asyncio.gather to run multiple tool.arun() calls concurrently
        result = await asyncio.gather(
            *[_aperform_agent_action(agent_action) for agent_action in actions]
        )

        return list(result)

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()

        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager=run_manager,
            )
            if isinstance(next_step_output, AgentFinish):
                return self._return(
                    next_step_output, intermediate_steps, run_manager=run_manager
                )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    return self._return(
                        tool_return, intermediate_steps, run_manager=run_manager
                    )
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        return self._return(output, intermediate_steps, run_manager=run_manager)

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the number of iterations and time elapsed
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        async with asyncio_timeout(self.max_execution_time):
            try:
                while self._should_continue(iterations, time_elapsed):
                    next_step_output = await self._atake_next_step(
                        name_to_tool_map,
                        color_mapping,
                        inputs,
                        intermediate_steps,
                        run_manager=run_manager,
                    )
                    if isinstance(next_step_output, AgentFinish):
                        return await self._areturn(
                            next_step_output,
                            intermediate_steps,
                            run_manager=run_manager,
                        )
                    intermediate_steps.extend(next_step_output)
                    if len(next_step_output) == 1:
                        next_step_action = next_step_output[0]
                        # See if tool should return directly
                        tool_return = self._get_tool_return(next_step_action)
                        if tool_return is not None:
                            return await self._areturn(
                                tool_return, intermediate_steps, run_manager=run_manager
                            )

                    iterations += 1
                    time_elapsed = time.time() - start_time
                output = self.agent.return_stopped_response(
                    self.early_stopping_method, intermediate_steps, **inputs
                )
                return await self._areturn(
                    output, intermediate_steps, run_manager=run_manager
                )
            except TimeoutError:
                # stop early when interrupted by the async timeout
                output = self.agent.return_stopped_response(
                    self.early_stopping_method, intermediate_steps, **inputs
                )
                return await self._areturn(
                    output, intermediate_steps, run_manager=run_manager
                )

    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        """Check if the tool is a returning tool."""
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # Invalid tools won't be in the map, so we return False.
        if agent_action.tool in name_to_tool_map:
            if name_to_tool_map[agent_action.tool].return_direct:
                return AgentFinish(
                    {self.agent.return_values[0]: observation},
                    "",
                )
        return None

    def _prepare_intermediate_steps(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[Tuple[AgentAction, str]]:
        if (
            isinstance(self.trim_intermediate_steps, int)
            and self.trim_intermediate_steps > 0
        ):
            return intermediate_steps[-self.trim_intermediate_steps :]
        elif callable(self.trim_intermediate_steps):
            return self.trim_intermediate_steps(intermediate_steps)
        else:
            return intermediate_steps
