"""Optional anonymous Parallel Search MCP tools for LLMCompiler."""

import asyncio
from importlib.util import find_spec
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from src.tools.base import Tool
from src.tools.search_context import search_session_id

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "LLMCompiler/ParallelSearch (+https://github.com/SqueezeAILab/LLMCompiler)"


def _error_message(error):
    # AnyIO may wrap transport failures in an ExceptionGroup on Python 3.10 too.
    children = getattr(error, "exceptions", None)
    if children:
        return "; ".join(_error_message(child) for child in children)
    return str(error) or type(error).__name__


class ParallelSearch:
    """Stateless MCP connections; the compiler owns the research session ID."""

    def __init__(self, timeout=30):
        if timeout <= 0:
            raise ValueError("Parallel timeout must be positive")
        self.timeout = timeout

    async def _request(self, name, arguments):
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=self.timeout
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as streams:
                read, write, _ = streams
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    available = await session.list_tools()
                    if name not in {tool.name for tool in available.tools}:
                        raise ValueError(f"Parallel MCP does not offer {name}")
                    return await session.call_tool(name, arguments)

    async def _call(self, name, arguments):
        arguments["session_id"] = search_session_id.get() or str(uuid4())
        try:
            result = await asyncio.wait_for(
                self._request(name, arguments), timeout=self.timeout
            )
            text = "\n".join(
                block.text for block in result.content if block.type == "text"
            )
            if result.isError:
                return f"Parallel error: {text}"
            return text or "Parallel returned no text results."
        except asyncio.TimeoutError:
            return "Parallel error: request timed out."
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # Observations must complete so dependent compiler tasks can proceed.
            # Never silently retry through another search provider.
            return f"Parallel error: {_error_message(error)[:1000]}"

    async def asearch(self, query):
        if not isinstance(query, str) or not query.strip():
            return "Parallel error: search requires a nonempty query."
        return await self._call(
            "web_search", {"objective": query, "search_queries": [query]}
        )

    async def afetch(self, url):
        if not isinstance(url, str):
            return "Parallel error: fetch requires an HTTP or HTTPS URL."
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in ("http", "https") and parsed.hostname
            valid = valid and not parsed.username and not parsed.password
        except ValueError:
            valid = False
        if not valid:
            return "Parallel error: fetch requires an HTTP or HTTPS URL."
        return await self._call("web_fetch", {"urls": [url]})


def resolve_search_tools(tools, provider=None, react=False):
    """Keep supplied tools unless Parallel is explicitly selected.

    A tool list without search is treated as an intentional restriction, not an
    invitation to add network access. ReAct's synchronous tools are unchanged.
    """
    if provider is None:
        return tools
    if provider != "parallel":
        raise ValueError(f"Unknown search provider: {provider}")
    if react:
        raise ValueError("Parallel search is supported by LLMCompiler, not --react")
    names = [tool.name for tool in tools]
    if names.count("search") != 1 or "fetch" in names:
        raise ValueError("Parallel requires one search tool and no existing fetch tool")
    if find_spec("mcp") is None:
        raise ImportError("Install Parallel support: pip install -r requirements-parallel.txt")
    search = ParallelSearch()
    replacement = Tool(
        name="search",
        func=search.asearch,
        description=(
            "search(query: str) -> str:\n"
            " - Searches the web using Parallel and returns excerpts with source URLs.\n"
            " - `query`: the information to find.\n"
        ),
        stringify_rule=lambda args: f"search({args[0]})",
    )
    fetch = Tool(
        name="fetch",
        func=search.afetch,
        description=(
            "fetch(url: str) -> str:\n"
            " - Reads an HTTP or HTTPS page using Parallel and returns excerpts.\n"
            " - `url`: one source URL, not the entire search observation.\n"
        ),
        stringify_rule=lambda args: f"fetch({args[0]})",
    )
    return [replacement if tool.name == "search" else tool for tool in tools] + [fetch]
