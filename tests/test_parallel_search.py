"""Exercise the actual resolver, compiler and MCP SDK with scripted models."""

import asyncio
import importlib
import json
import os
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from langchain_community.llms import FakeListLLM
from langchain.pydantic_v1 import Field

from src.llm_compiler.llm_compiler import LLMCompiler
from src.tools.base import Tool
from src.tools.parallel_search import MCP_URL, USER_AGENT, resolve_search_tools
from src.tools.search_context import search_session_id
from src.utils.logger_utils import enable_logging

enable_logging(False)


class RecordingLLM(FakeListLLM):
    seen: list = Field(default_factory=list)

    async def _acall(self, prompt, **kwargs):
        self.seen.append(prompt)
        return await super()._acall(prompt, **kwargs)


class TokenLLM(FakeListLLM):
    async def _acall(self, prompt, run_manager=None, **kwargs):
        response = await super()._acall(prompt, run_manager=run_manager, **kwargs)
        for token in response:
            if run_manager:
                await run_manager.on_llm_new_token(token)
        return response


class MCPFixture:
    """Replace only HTTP I/O, retaining the real MCP handshake and client."""

    def __init__(self, error=False, block=False, transport_error=False):
        self.requests = []
        self.calls = []
        self.error = error
        self.block = block
        self.transport_error = transport_error
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def handle(self, request):
        self.requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "notifications/initialized":
            return httpx.Response(202)
        elif method == "tools/list":
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}
                                for name in ("web_search", "web_fetch")]}
        elif method == "tools/call":
            params = payload["params"]
            self.calls.append(params)
            self.started.set()
            if self.transport_error:
                raise httpx.ConnectError("fixture connection failed", request=request)
            if self.block:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()
            result = {"content": [{"type": "text", "text":
                      "quota fixture" if self.error else
                      "Python source: https://www.python.org/?a=1&b=two\nA programming language."}],
                      "isError": self.error}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    @contextmanager
    def installed(self):
        real_client = httpx.AsyncClient

        handler = self.handle

        class FixtureClient(real_client):
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = httpx.MockTransport(handler)
                super().__init__(*args, **kwargs)

        with patch("src.tools.parallel_search.httpx.AsyncClient", FixtureClient):
            yield

    def assert_requests(self, case):
        case.assertTrue(self.calls)
        for request in self.requests:
            case.assertEqual(str(request.url), MCP_URL)
            case.assertEqual(request.headers["user-agent"], USER_AGENT)
            case.assertNotIn("authorization", request.headers)
        methods = [json.loads(r.content)["method"] for r in self.requests if r.method == "POST"]
        case.assertIn("initialize", methods)
        case.assertIn("tools/list", methods)
        case.assertIn("tools/call", methods)


def original_tools():
    async def search(query):
        return "Wikipedia: " + query
    return [Tool(name="search", func=search, description="Wikipedia search")]


def compiler(tools, plans=None, stream=False, replan=False):
    planner = TokenLLM(responses=plans or ['1. search("Python programming language")\n2. join()'])
    joiner = RecordingLLM(responses=(
        ["Thought: Read more\nAction: Replan(more)", "Action: Finish(Python)"]
        if replan else ["Action: Finish(Python)"]
    ))
    return LLMCompiler(
        tools=tools, planner_llm=planner, planner_example_prompt="",
        planner_example_prompt_replan=None, planner_stop=None, planner_stream=stream,
        agent_llm=joiner, joinner_prompt="Answer using the observations.",
        joinner_prompt_final=None, max_replans=2 if replan else 1, benchmark=False,
    ), joiner


class ParallelSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_resolver_and_result_consumption(self):
        # Import the actual runner without executing main or accessing a model API.
        with patch.object(sys, "argv", ["run_llm_compiler.py", "--benchmark_name", "hotpotqa",
                                       "--store", "/tmp/unused-parallel-test.json"]):
            runner = importlib.import_module("run_llm_compiler")
        args = SimpleNamespace(benchmark_name="hotpotqa", react=False, search_provider=None)
        self.assertIs(runner.get_tools("unused", args), runner.hotpotqa_tools)
        args.search_provider = "parallel"
        with patch.dict(os.environ, {}, clear=True):
            tools = runner.get_tools("unused", args)
            fixture = MCPFixture()
            agent, joiner = compiler(tools)
            with fixture.installed():
                self.assertEqual(await asyncio.wait_for(agent.arun("What is Python?"), 5), "Python")
        self.assertIn("https://www.python.org/?a=1&b=two", joiner.seen[0])
        self.assertIn("A programming language.", joiner.seen[0])
        fixture.assert_requests(self)
        self.assertEqual(fixture.calls[0]["arguments"]["search_queries"], ["Python programming language"])

    async def test_replans_fetch_followup_and_new_runs(self):
        tools = resolve_search_tools(original_tools(), "parallel")
        plans = ['1. search("Python programming language")\n2. fetch("https://www.python.org/?a=1&b=two")\n3. join()',
                 '1. search("Python release history")\n2. join()']
        agent, _ = compiler(tools, plans, replan=True)
        fixture = MCPFixture()
        with fixture.installed():
            await asyncio.wait_for(agent.arun("Research Python"), 5)
            first = {call["arguments"]["session_id"] for call in fixture.calls}
            self.assertEqual(len(first), 1)
            self.assertEqual(len(fixture.calls), 3)
            fetch = next(c for c in fixture.calls if c["name"] == "web_fetch")
            self.assertEqual(fetch["arguments"]["urls"], ["https://www.python.org/?a=1&b=two"])
            await asyncio.wait_for(agent.arun("New research"), 5)
        second = {call["arguments"]["session_id"] for call in fixture.calls[3:]}
        self.assertEqual(len(second), 1)
        self.assertTrue(first.isdisjoint(second))
        self.assertIsNone(search_session_id.get())
        fixture.assert_requests(self)

    async def test_streaming_plan_consumes_search_and_fetch(self):
        tools = resolve_search_tools(original_tools(), "parallel")
        plan = ['1. search("Python programming language")\n2. fetch("https://www.python.org/")\n3. join()']
        agent, joiner = compiler(tools, plan, stream=True)
        fixture = MCPFixture()
        with fixture.installed():
            self.assertEqual(await asyncio.wait_for(agent.arun("Python"), 5), "Python")
        self.assertIn("A programming language.", joiner.seen[0])
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(len({c["arguments"]["session_id"] for c in fixture.calls}), 1)
        fixture.assert_requests(self)

    async def test_quota_replan_keeps_identity_without_automatic_retry(self):
        tools = resolve_search_tools(original_tools(), "parallel")
        agent, joiner = compiler(tools, replan=True)
        fixture = MCPFixture(error=True)
        with fixture.installed():
            await asyncio.wait_for(agent.arun("question"), 5)
        self.assertEqual(len(fixture.calls), 2)
        self.assertEqual(len({c["arguments"]["session_id"] for c in fixture.calls}), 1)
        self.assertTrue(all("Parallel error: quota fixture" in prompt for prompt in joiner.seen))

    async def test_concurrent_runs_share_tools_but_not_identity(self):
        tools = resolve_search_tools(original_tools(), "parallel")
        first, _ = compiler(tools, ['1. search("one alpha")\n2. search("one beta")\n3. join()'])
        second, _ = compiler(tools, ['1. search("two alpha")\n2. search("two beta")\n3. join()'])
        fixture = MCPFixture()
        with fixture.installed():
            await asyncio.wait_for(asyncio.gather(first.arun("one"), second.arun("two")), 5)
        ids = [call["arguments"]["session_id"] for call in fixture.calls]
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(sorted(ids.count(value) for value in set(ids)), [2, 2])
        by_run = {label: {call["arguments"]["session_id"] for call in fixture.calls
                          if call["arguments"]["objective"].startswith(label)}
                  for label in ("one", "two")}
        self.assertEqual(len(by_run["one"]), 1)
        self.assertEqual(len(by_run["two"]), 1)
        self.assertTrue(by_run["one"].isdisjoint(by_run["two"]))

    async def test_defaults_restrictions_and_explicit_choices(self):
        tools = original_tools()
        self.assertIs(resolve_search_tools(tools), tools)
        self.assertEqual(await tools[0].func("Python"), "Wikipedia: Python")
        disabled = []
        self.assertIs(resolve_search_tools(disabled), disabled)
        with self.assertRaisesRegex(ValueError, "requires one search"):
            resolve_search_tools(disabled, "parallel")
        with self.assertRaisesRegex(ValueError, "not --react"):
            resolve_search_tools(tools, "parallel", react=True)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_search_tools(tools, "other")
        with patch("src.tools.parallel_search.find_spec", return_value=None):
            with self.assertRaisesRegex(ImportError, "requirements-parallel"):
                resolve_search_tools(tools, "parallel")

    async def test_errors_are_consumed_without_fallback_or_retry(self):
        tools = resolve_search_tools(original_tools(), "parallel")
        for transport_error in (False, True):
            fixture = MCPFixture(error=True, transport_error=transport_error)
            agent, joiner = compiler(tools)
            with fixture.installed():
                await asyncio.wait_for(agent.arun("question"), 5)
            self.assertIn("Parallel error:", joiner.seen[0])
            self.assertEqual(len(fixture.calls), 1)
            self.assertNotIn("Wikipedia:", joiner.seen[0])

    async def test_timeout_is_consumed_and_request_cancelled(self):
        tools = resolve_search_tools(original_tools(), "parallel")
        tools[0].func.__self__.timeout = 0.1
        fixture = MCPFixture(block=True)
        agent, joiner = compiler(tools)
        with fixture.installed():
            await asyncio.wait_for(agent.arun("question"), 5)
        self.assertIn("request timed out", joiner.seen[0])
        self.assertTrue(fixture.cancelled.is_set())

    async def test_compiler_cancellation_stops_network_work(self):
        for stream in (False, True):
            tools = resolve_search_tools(original_tools(), "parallel")
            fixture = MCPFixture(block=True)
            agent, _ = compiler(tools, stream=stream)
            with fixture.installed():
                task = asyncio.create_task(agent.arun("cancel this"))
                await asyncio.wait_for(fixture.started.wait(), 5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.wait_for(fixture.cancelled.wait(), 5)
        self.assertIsNone(search_session_id.get())

    async def test_invalid_inputs_do_not_make_requests(self):
        tools = resolve_search_tools(original_tools(), "parallel")
        fixture = MCPFixture()
        with fixture.installed():
            self.assertIn("nonempty", await tools[0].func(" "))
            for url in ("file:///tmp/test", "javascript:alert(1)", "https://user:pass@example.com", "https://[bad", "text"):
                self.assertIn("requires an HTTP", await tools[1].func(url))
        self.assertFalse(fixture.requests)


if __name__ == "__main__":
    unittest.main()
