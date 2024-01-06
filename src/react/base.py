from enum import Enum
from typing import Any, List, Sequence

from langchain.agents.utils import validate_tools_single_input
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool

from src.agents.agent import Agent, AgentOutputParser
from src.executors.agent_executor import AgentExecutor
from src.react.output_parser import ReActOutputParser


class ReActDocstoreAgentForWiki(Agent):
    """Agent for the ReAct chain."""

    output_parser: AgentOutputParser = Field(default_factory=ReActOutputParser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ReActOutputParser()

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        super()._validate_tools(tools)

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def _stop(self) -> List[str]:
        return ["\nObservation:"]

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return "Thought:"


def initialize_react_agent_executor(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: BasePromptTemplate,
    verbose: bool = False,
    agent_kwargs: dict[str, Any] = {},
):
    """Load an agent executor given tools and LLM.
    Simplified version of langchain.agents.initialize.initialize_agent
    """
    agent = ReActDocstoreAgentForWiki.from_llm_and_tools(
        llm=llm,
        tools=tools,
        prompt=prompt,
        human_message_template=agent_kwargs.get("human_message_template", None),
        input_variables=agent_kwargs.get("input_variables", None),
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=verbose
    )
    return agent_executor
