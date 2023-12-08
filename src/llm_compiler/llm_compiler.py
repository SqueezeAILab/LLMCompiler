import asyncio
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

from src.callbacks.callbacks import AsyncStatsCallbackHandler
from src.chains.chain import Chain
from src.llm_compiler.constants import JOINNER_REPLAN
from src.llm_compiler.planner import Planner
from src.llm_compiler.task_fetching_unit import Task, TaskFetchingUnit
from src.tools.base import StructuredTool, Tool
from src.utils.logger_utils import log

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.llms import BaseLLM
from langchain.prompts.base import StringPromptValue


class LLMCompilerAgent:
    """Self defined agent for LLM Compiler."""

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def arun(self, prompt: str, callbacks=None) -> str:
        return await self.llm.agenerate_prompt(
            prompts=[StringPromptValue(text=prompt)],
            stop=None,
            callbacks=callbacks,
        )


class LLMCompiler(Chain, extra="allow"):
    """LLMCompuler Engine."""

    """The step container to use."""
    input_key: str = "input"
    output_key: str = "output"

    def __init__(
        self,
        tools: Sequence[Union[Tool, StructuredTool]],
        planner_llm: BaseLLM,
        planner_example_prompt: str,
        planner_example_prompt_replan: Optional[str],
        planner_stop: Optional[list[str]],
        planner_stream: bool,
        agent_llm: BaseLLM,
        joinner_prompt: str,
        joinner_prompt_final: Optional[str],
        max_replans: int,
        benchmark: bool,
        **kwargs,
    ) -> None:
        """
        Args:
            tools: List of tools to use.
            max_replans: Maximum number of replans to do.
            benchmark: Whether to collect benchmark stats.

        Planner Args:
            planner_llm: LLM to use for planning.
            planner_example_prompt: Example prompt for planning.
            planner_example_prompt_replan: Example prompt for replanning.
                Assign this if you want to use different example prompt for replanning.
                If not assigned, default to `planner_example_prompt`.
            planner_stop: Stop tokens for planning.
            planner_stream: Whether to stream the planning.

        Agent Args:
            agent_llm: LLM to use for agent.
            joinner_prompt: Prompt to use for joinner.
            joinner_prompt_final: Prompt to use for joinner at the final replanning iter.
                If not assigned, default to `joinner_prompt`.
        """
        super().__init__(**kwargs)

        if not planner_example_prompt_replan:
            log(
                "Replan example prompt not specified, using the same prompt as the planner."
            )
            planner_example_prompt_replan = planner_example_prompt

        self.planner = Planner(
            llm=planner_llm,
            example_prompt=planner_example_prompt,
            example_prompt_replan=planner_example_prompt_replan,
            tools=tools,
            stop=planner_stop,
        )

        self.agent = LLMCompilerAgent(agent_llm)
        self.joinner_prompt = joinner_prompt
        self.joinner_prompt_final = joinner_prompt_final or joinner_prompt
        self.planner_stream = planner_stream
        self.max_replans = max_replans

        # callbacks
        self.benchmark = benchmark
        if benchmark:
            self.planner_callback = AsyncStatsCallbackHandler(stream=planner_stream)
            self.executor_callback = AsyncStatsCallbackHandler(stream=False)
        else:
            self.planner_callback = None
            self.executor_callback = None

    def get_all_stats(self):
        stats = {}
        if self.benchmark:
            stats["planner"] = self.planner_callback.get_stats()
            stats["executor"] = self.executor_callback.get_stats()
            stats["total"] = {
                k: v + stats["executor"][k] for k, v in stats["planner"].items()
            }

        return stats

    def reset_all_stats(self):
        if self.planner_callback:
            self.planner_callback.reset()
        if self.executor_callback:
            self.executor_callback.reset()

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    # TODO(sk): move all join related functions to a separate class

    def _parse_joinner_output(self, raw_answer: str) -> str:
        """We expect the joinner output format to be:
        ```
        Thought: xxx
        Action: Finish/Replan(yyy)
        ```
        Returns:
            thought (xxx)
            answer (yyy)
            is_replan (True/False)
        """
        thought, answer, is_replan = "", "", False  # default values
        raw_answers = raw_answer.split("\n")
        for ans in raw_answers:
            if ans.startswith("Action:"):
                answer = ans[ans.find("(") + 1 : ans.find(")")]
                is_replan = JOINNER_REPLAN in ans
            elif ans.startswith("Thought:"):
                thought = ans.split("Thought:")[1].strip()
        return thought, answer, is_replan

    def _generate_context_for_replanner(
        self, tasks: Mapping[int, Task], joinner_thought: str
    ) -> str:
        """Formatted like this:
        ```
        1. action 1
        Observation: xxx
        2. action 2
        Observation: yyy
        ...
        Thought: joinner_thought
        ```
        """
        previous_plan_and_observations = "\n".join(
            [
                task.get_though_action_observation(
                    include_action=True, include_action_idx=True
                )
                for task in tasks.values()
                if not task.is_join
            ]
        )
        joinner_thought = f"Thought: {joinner_thought}"
        context = "\n\n".join([previous_plan_and_observations, joinner_thought])
        return context

    def _format_contexts(self, contexts: Sequence[str]) -> str:
        """contexts is a list of context
        each context is formatted as the description of _generate_context_for_replanner
        """
        formatted_contexts = ""
        for context in contexts:
            formatted_contexts += f"Previous Plan:\n\n{context}\n\n"
        formatted_contexts += "Current Plan:\n\n"
        return formatted_contexts

    async def join(
        self, input_query: str, agent_scratchpad: str, is_final: bool
    ) -> str:
        if is_final:
            joinner_prompt = self.joinner_prompt_final
        else:
            joinner_prompt = self.joinner_prompt
        prompt = (
            f"{joinner_prompt}\n"  # Instructions and examples
            f"Question: {input_query}\n\n"  # User input query
            f"{agent_scratchpad}\n"  # T-A-O
            # "---\n"
        )
        log("Joining prompt:\n", prompt, block=True)
        response = await self.agent.arun(
            prompt, callbacks=[self.executor_callback] if self.benchmark else None
        )
        raw_answer = cast(str, response.generations[0][0].message.content)
        log("Question: \n", input_query, block=True)
        log("Raw Answer: \n", raw_answer, block=True)
        thought, answer, is_replan = self._parse_joinner_output(raw_answer)
        if is_final:
            # If final, we don't need to replan
            is_replan = False
        return thought, answer, is_replan

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ):
        raise NotImplementedError("LLMCompiler is async only.")

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        contexts = []
        joinner_thought = ""
        agent_scratchpad = ""
        for i in range(self.max_replans):
            is_first_iter = i == 0
            is_final_iter = i == self.max_replans - 1

            task_fetching_unit = TaskFetchingUnit()
            if self.planner_stream:
                task_queue = asyncio.Queue()
                asyncio.create_task(
                    self.planner.aplan(
                        inputs=inputs,
                        task_queue=task_queue,
                        is_replan=not is_first_iter,
                        callbacks=[self.planner_callback]
                        if self.planner_callback
                        else None,
                    )
                )
                await task_fetching_unit.aschedule(
                    task_queue=task_queue, func=lambda x: None
                )
            else:
                tasks = await self.planner.plan(
                    inputs=inputs,
                    is_replan=not is_first_iter,
                    # callbacks=run_manager.get_child() if run_manager else None,
                    callbacks=[self.planner_callback]
                    if self.planner_callback
                    else None,
                )
                log("Graph of tasks: ", tasks, block=True)
                task_fetching_unit.set_tasks(tasks)
                await task_fetching_unit.schedule()
            tasks = task_fetching_unit.tasks

            # collect thought-action-observation
            agent_scratchpad += "\n\n"
            agent_scratchpad += "".join(
                [
                    task.get_though_action_observation(
                        include_action=True, include_thought=True
                    )
                    for task in tasks.values()
                    if not task.is_join
                ]
            )
            agent_scratchpad = agent_scratchpad.strip()

            log("Agent scratchpad:\n", agent_scratchpad, block=True)
            joinner_thought, answer, is_replan = await self.join(
                inputs["input"],
                agent_scratchpad=agent_scratchpad,
                is_final=is_final_iter,
            )
            if not is_replan:
                log("Break out of replan loop.")
                break

            # Collect contexts for the subsequent replanner
            context = self._generate_context_for_replanner(
                tasks=tasks, joinner_thought=joinner_thought
            )
            contexts.append(context)
            formatted_contexts = self._format_contexts(contexts)
            log("Contexts:\n", formatted_contexts, block=True)
            inputs["context"] = formatted_contexts

        if is_final_iter:
            log("Reached max replan limit.")

        return {self.output_key: answer}
