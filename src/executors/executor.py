from typing import Any, List

from langchain_experimental.plan_and_execute.executors.base import BaseExecutor
from src.agents.structured_chat_agent import StructuredChatAgent
from src.chains.chain import Chain
from src.executors.agent_executor import initialize_agent_executor
from src.executors.schema import StepResponse

from langchain.callbacks.manager import Callbacks
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool

HUMAN_MESSAGE_TEMPLATE = """Previous steps: {previous_steps}

Current objective: {current_step}

{agent_scratchpad}"""


class Executor(BaseExecutor):
    """Chain executor."""

    chain: Chain
    """The chain to use."""

    def step(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""
        response = self.chain.run(**inputs, callbacks=callbacks)
        return StepResponse(response=response)

    async def astep(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""
        response = await self.chain.arun(**inputs, callbacks=callbacks)
        return StepResponse(response=response)


def load_executor(
    llm: BaseLanguageModel,
    tools: List[BaseTool],
    verbose: bool = False,
) -> Executor:
    """
    Load an agent executor.

    Args:
        llm: BaseLanguageModel
        tools: List[BaseTool]
        verbose: bool. Defaults to False.
        include_task_in_prompt: bool. Defaults to False.

    Returns:
        ChainExecutor
    """
    input_variables = ["previous_steps", "current_step", "agent_scratchpad"]
    agent_executor = initialize_agent_executor(
        agent_cls=StructuredChatAgent,
        llm=llm,
        tools=tools,
        verbose=verbose,
        agent_kwargs={
            "human_message_template": HUMAN_MESSAGE_TEMPLATE,
            "input_variables": input_variables,
        },
    )
    return Executor(chain=agent_executor)
