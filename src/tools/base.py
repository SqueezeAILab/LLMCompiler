"""Base implementation for tools or skills."""
from __future__ import annotations

import asyncio
import inspect
from functools import partial
from inspect import signature
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Type, Union

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    create_model,
    validate_arguments,
)
from langchain.schema.runnable import RunnableConfig
from langchain.tools import BaseTool


class SchemaAnnotationError(TypeError):
    """Raised when 'args_schema' is missing or has an incorrect type annotation."""


def _create_subset_model(
    name: str, model: BaseModel, field_names: list
) -> Type[BaseModel]:
    """Create a pydantic model with only a subset of model's fields."""
    fields = {}
    for field_name in field_names:
        field = model.__fields__[field_name]
        fields[field_name] = (field.outer_type_, field.field_info)
    return create_model(name, **fields)  # type: ignore


def _get_filtered_args(
    inferred_model: Type[BaseModel],
    func: Callable,
) -> dict:
    """Get the arguments from a function's signature."""
    schema = inferred_model.schema()["properties"]
    valid_keys = signature(func).parameters
    return {k: schema[k] for k in valid_keys if k not in ("run_manager", "callbacks")}


class _SchemaConfig:
    """Configuration for the pydantic model."""

    extra: Any = Extra.forbid
    arbitrary_types_allowed: bool = True


def create_schema_from_function(
    model_name: str,
    func: Callable,
) -> Type[BaseModel]:
    """Create a pydantic schema from a function's signature.
    Args:
        model_name: Name to assign to the generated pydandic schema
        func: Function to generate the schema from
    Returns:
        A pydantic model with the same arguments as the function
    """
    # https://docs.pydantic.dev/latest/usage/validation_decorator/
    validated = validate_arguments(func, config=_SchemaConfig)  # type: ignore
    inferred_model = validated.model  # type: ignore
    if "run_manager" in inferred_model.__fields__:
        del inferred_model.__fields__["run_manager"]
    if "callbacks" in inferred_model.__fields__:
        del inferred_model.__fields__["callbacks"]
    # Pydantic adds placeholder virtual fields we need to strip
    valid_properties = _get_filtered_args(inferred_model, func)
    return _create_subset_model(
        f"{model_name}Schema", inferred_model, list(valid_properties)
    )


class ToolException(Exception):
    """An optional exception that tool throws when execution error occurs.

    When this exception is thrown, the agent will not stop working,
    but will handle the exception according to the handle_tool_error
    variable of the tool, and the processing result will be returned
    to the agent as observation, and printed in red on the console.
    """

    pass


class Tool(BaseTool):
    """Tool that takes in function or coroutine directly."""

    description: str = ""
    func: Optional[Callable[..., str]]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[str]]] = None
    """The asynchronous version of the function."""
    stringify_rule: Optional[Callable[..., str]] = None

    # --- Runnable ---

    async def ainvoke(
        self,
        input: Union[str, Dict],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        if not self.coroutine:
            # If the tool does not implement async, fall back to default implementation
            return await asyncio.get_running_loop().run_in_executor(
                None, partial(self.invoke, input, config, **kwargs)
            )

        return super().ainvoke(input, config, **kwargs)

    # --- Tool ---

    @property
    def args(self) -> dict:
        """The tool's input arguments."""
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        # For backwards compatibility, if the function signature is ambiguous,
        # assume it takes a single string input.
        return {"tool_input": {"type": "string"}}

    def _to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
        """Convert tool input to pydantic model."""
        args, kwargs = super()._to_args_and_kwargs(tool_input)
        # For backwards compatibility. The tool must be run with a single input
        all_args = list(args) + list(kwargs.values())
        if len(all_args) != 1:
            raise ToolException(
                f"Too many arguments to single-input tool {self.name}."
                f" Args: {all_args}"
            )
        return tuple(all_args), {}

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        if self.func:
            new_argument_supported = signature(self.func).parameters.get("callbacks")
            return (
                self.func(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else self.func(*args, **kwargs)
            )
        raise NotImplementedError("Tool does not support sync")

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool asynchronously."""
        if self.coroutine:
            new_argument_supported = signature(self.coroutine).parameters.get(
                "callbacks"
            )
            return (
                await self.coroutine(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else await self.coroutine(*args, **kwargs)
            )
        else:
            return await asyncio.get_running_loop().run_in_executor(
                None, partial(self._run, run_manager=run_manager, **kwargs), *args
            )

    # TODO: this is for backwards compatibility, remove in future
    def __init__(
        self, name: str, func: Optional[Callable], description: str, **kwargs: Any
    ) -> None:
        """Initialize tool."""
        super(Tool, self).__init__(
            name=name, func=func, description=description, **kwargs
        )

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable],
        name: str,  # We keep these required to support backwards compatibility
        description: str,
        return_direct: bool = False,
        args_schema: Optional[Type[BaseModel]] = None,
        coroutine: Optional[
            Callable[..., Awaitable[Any]]
        ] = None,  # This is last for compatibility, but should be after func
        **kwargs: Any,
    ) -> Tool:
        """Initialize tool from a function."""
        if func is None and coroutine is None:
            raise ValueError("Function and/or coroutine must be provided")
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            **kwargs,
        )


class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    description: str = ""
    args_schema: Type[BaseModel] = Field(..., description="The tool schema.")
    """The input arguments' schema."""
    func: Optional[Callable[..., Any]]
    """The function to run when the tool is called."""
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None
    """The asynchronous version of the function."""
    stringify_rule: Optional[Callable[..., str]] = None

    # --- Runnable ---

    async def ainvoke(
        self,
        input: Union[str, Dict],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        if not self.coroutine:
            # If the tool does not implement async, fall back to default implementation
            return await asyncio.get_running_loop().run_in_executor(
                None, partial(self.invoke, input, config, **kwargs)
            )

        return super().ainvoke(input, config, **kwargs)

    # --- Tool ---

    @property
    def args(self) -> dict:
        """The tool's input arguments."""
        return self.args_schema.schema()["properties"]

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool."""
        if self.func:
            new_argument_supported = signature(self.func).parameters.get("callbacks")
            return (
                self.func(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else self.func(*args, **kwargs)
            )
        raise NotImplementedError("Tool does not support sync")

    async def _arun(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool asynchronously."""
        if self.coroutine:
            new_argument_supported = signature(self.coroutine).parameters.get(
                "callbacks"
            )
            return (
                await self.coroutine(
                    *args,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **kwargs,
                )
                if new_argument_supported
                else await self.coroutine(*args, **kwargs)
            )
        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._run,
            partial(self._run, run_manager=run_manager, **kwargs),
            *args,
        )

    @classmethod
    def from_function(
        cls,
        func: Optional[Callable] = None,
        coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Optional[Type[BaseModel]] = None,
        infer_schema: bool = True,
        **kwargs: Any,
    ) -> StructuredTool:
        """Create tool from a given function.

        A classmethod that helps to create a tool from a function.

        Args:
            func: The function from which to create a tool
            coroutine: The async function from which to create a tool
            name: The name of the tool. Defaults to the function name
            description: The description of the tool. Defaults to the function docstring
            return_direct: Whether to return the result directly or as a callback
            args_schema: The schema of the tool's input arguments
            infer_schema: Whether to infer the schema from the function's signature
            **kwargs: Additional arguments to pass to the tool

        Returns:
            The tool

        Examples:

            .. code-block:: python

                def add(a: int, b: int) -> int:
                    \"\"\"Add two numbers\"\"\"
                    return a + b
                tool = StructuredTool.from_function(add)
                tool.run(1, 2) # 3
        """

        if func is not None:
            source_function = func
        elif coroutine is not None:
            source_function = coroutine
        else:
            raise ValueError("Function and/or coroutine must be provided")
        name = name or source_function.__name__
        description = description or source_function.__doc__
        if description is None:
            raise ValueError(
                "Function must have a docstring if description not provided."
            )

        # Description example:
        # search_api(query: str) - Searches the API for the query.
        sig = signature(source_function)
        description = f"{name}{sig} - {description.strip()}"
        _args_schema = args_schema
        if _args_schema is None and infer_schema:
            _args_schema = create_schema_from_function(f"{name}Schema", source_function)
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            args_schema=_args_schema,
            description=description,
            return_direct=return_direct,
            **kwargs,
        )


def tool(
    *args: Union[str, Callable],
    return_direct: bool = False,
    args_schema: Optional[Type[BaseModel]] = None,
    infer_schema: bool = True,
) -> Callable:
    """Make tools out of functions, can be used with or without arguments.

    Args:
        *args: The arguments to the tool.
        return_direct: Whether to return directly from the tool rather
            than continuing the agent loop.
        args_schema: optional argument schema for user to specify
        infer_schema: Whether to infer the schema of the arguments from
            the function's signature. This also makes the resultant tool
            accept a dictionary input to its `run()` function.

    Requires:
        - Function must be of type (str) -> str
        - Function must have a docstring

    Examples:
        .. code-block:: python

            @tool
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return

            @tool("search", return_direct=True)
            def search_api(query: str) -> str:
                # Searches the API for the query.
                return
    """

    def _make_with_name(tool_name: str) -> Callable:
        def _make_tool(dec_func: Callable) -> BaseTool:
            if inspect.iscoroutinefunction(dec_func):
                coroutine = dec_func
                func = None
            else:
                coroutine = None
                func = dec_func

            if infer_schema or args_schema is not None:
                return StructuredTool.from_function(
                    func,
                    coroutine,
                    name=tool_name,
                    return_direct=return_direct,
                    args_schema=args_schema,
                    infer_schema=infer_schema,
                )
            # If someone doesn't want a schema applied, we must treat it as
            # a simple string->string function
            if func.__doc__ is None:
                raise ValueError(
                    "Function must have a docstring if "
                    "description not provided and infer_schema is False."
                )
            return Tool(
                name=tool_name,
                func=func,
                description=f"{tool_name} tool",
                return_direct=return_direct,
                coroutine=coroutine,
            )

        return _make_tool

    if len(args) == 1 and isinstance(args[0], str):
        # if the argument is a string, then we use the string as the tool name
        # Example usage: @tool("search", return_direct=True)
        return _make_with_name(args[0])
    elif len(args) == 1 and callable(args[0]):
        # if the argument is a function, then we use the function name as the tool name
        # Example usage: @tool
        return _make_with_name(args[0].__name__)(args[0])
    elif len(args) == 0:
        # if there are no arguments, then we use the function name as the tool name
        # Example usage: @tool(return_direct=True)
        def _partial(func: Callable[[str], str]) -> BaseTool:
            return _make_with_name(func.__name__)(func)

        return _partial
    else:
        raise ValueError("Too many arguments for tool decorator")
