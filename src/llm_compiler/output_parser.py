import ast
import re
from typing import Any, Sequence, Union

from src.llm_compiler.task_fetching_unit import Task
from src.tools.base import StructuredTool, Tool

from langchain.agents.agent import AgentOutputParser
from langchain.schema import OutputParserException

THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
# $1 or ${1} -> 1
ID_PATTERN = r"\$\{?(\d+)\}?"

END_OF_PLAN = "<END_OF_PLAN>"


def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


class LLMCompilerPlanParser(AgentOutputParser, extra="allow"):
    """Planning output parser."""

    def __init__(self, tools: Sequence[Union[Tool, StructuredTool]], **kwargs):
        super().__init__(**kwargs)
        self.tools = tools

    def parse(self, text: str) -> list[str]:
        # 1. search("Ronaldo number of kids") -> 1, "search", '"Ronaldo number of kids"'
        # pattern = r"(\d+)\. (\w+)\(([^)]+)\)"
        pattern = rf"(?:{THOUGHT_PATTERN}\n)?{ACTION_PATTERN}"
        matches = re.findall(pattern, text)

        graph_dict = {}

        for match in matches:
            # idx = 1, function = "search", args = "Ronaldo number of kids"
            # thought will be the preceding thought, if any, otherwise an empty string
            thought, idx, tool_name, args, _ = match
            idx = int(idx)

            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )

            graph_dict[idx] = task
            if task.is_join:
                break

        return graph_dict


### Helper functions


def _parse_llm_compiler_action_args(args: str) -> list[Any]:
    """Parse arguments from a string."""
    # This will convert the string into a python object
    # e.g. '"Ronaldo number of kids"' -> ("Ronaldo number of kids", )
    # '"I can answer the question now.", [3]' -> ("I can answer the question now.", [3])
    if args == "":
        return ()
    try:
        args = ast.literal_eval(args)
    except:
        args = args
    if not isinstance(args, list) and not isinstance(args, tuple):
        args = (args,)
    return args


def _find_tool(
    tool_name: str, tools: Sequence[Union[Tool, StructuredTool]]
) -> Union[Tool, StructuredTool]:
    """Find a tool by name.

    Args:
        tool_name: Name of the tool to find.

    Returns:
        Tool or StructuredTool.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise OutputParserException(f"Tool {tool_name} not found.")


def _get_dependencies_from_graph(
    idx: int, tool_name: str, args: Sequence[Any]
) -> dict[str, list[str]]:
    """Get dependencies from a graph."""
    if tool_name == "join":
        # depends on the previous step
        dependencies = list(range(1, idx))
    else:
        # define dependencies based on the dependency rule in tool_definitions.py
        dependencies = [i for i in range(1, idx) if default_dependency_rule(i, args)]

    return dependencies


def instantiate_task(
    tools: Sequence[Union[Tool, StructuredTool]],
    idx: int,
    tool_name: str,
    args: str,
    thought: str,
) -> Task:
    dependencies = _get_dependencies_from_graph(idx, tool_name, args)
    args = _parse_llm_compiler_action_args(args)
    if tool_name == "join":
        # join does not have a tool
        tool_func = lambda x: None
        stringify_rule = None
    else:
        tool = _find_tool(tool_name, tools)
        tool_func = tool.func
        stringify_rule = tool.stringify_rule
    return Task(
        idx=idx,
        name=tool_name,
        tool=tool_func,
        args=args,
        dependencies=dependencies,
        stringify_rule=stringify_rule,
        thought=thought,
        is_join=tool_name == "join",
    )
