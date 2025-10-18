"""Shared pytest fixtures for LLMCompiler tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        Path: Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary file for testing.

    Args:
        temp_dir: Temporary directory fixture

    Yields:
        Path: Path to the temporary file
    """
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    yield temp_file_path


@pytest.fixture
def mock_env_vars(monkeypatch) -> Dict[str, str]:
    """Mock environment variables for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        Dict[str, str]: Dictionary of mocked environment variables
    """
    env_vars = {
        "OPENAI_API_KEY": "test-api-key-123",
        "AZURE_ENDPOINT": "https://test.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-01-01",
        "AZURE_DEPLOYMENT_NAME": "test-deployment",
        "AZURE_OPENAI_API_KEY": "test-azure-key",
        "FRIENDLI_TOKEN": "test-friendli-token",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Mock OpenAI API response for testing.

    Returns:
        Dict[str, Any]: Mock response data
    """
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_llm_client(mock_openai_response) -> MagicMock:
    """Mock LLM client for testing.

    Args:
        mock_openai_response: Mock OpenAI response fixture

    Returns:
        MagicMock: Mock LLM client
    """
    client = MagicMock()
    client.chat.completions.create.return_value = mock_openai_response
    return client


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing.

    Returns:
        Dict[str, Any]: Sample configuration dictionary
    """
    return {
        "model_type": "openai",
        "model_name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "benchmark": "hotpotqa",
        "streaming": True,
        "logging": False,
    }


@pytest.fixture
def sample_tools_config() -> list:
    """Sample tools configuration for testing.

    Returns:
        list: List of sample tool configurations
    """
    return [
        {
            "name": "search",
            "description": "Search for information",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
        },
        {
            "name": "calculator",
            "description": "Perform calculations",
            "parameters": {
                "expression": {"type": "string", "description": "Math expression"}
            },
        },
    ]


@pytest.fixture
def sample_task_list() -> list:
    """Sample task list for testing LLMCompiler parallel execution.

    Returns:
        list: List of sample tasks
    """
    return [
        {
            "id": 1,
            "name": "task1",
            "tool": "search",
            "args": {"query": "test query 1"},
            "dependencies": [],
        },
        {
            "id": 2,
            "name": "task2",
            "tool": "search",
            "args": {"query": "test query 2"},
            "dependencies": [],
        },
        {
            "id": 3,
            "name": "task3",
            "tool": "calculator",
            "args": {"expression": "2 + 2"},
            "dependencies": [1, 2],
        },
    ]


@pytest.fixture
def mock_dataset_path(temp_dir: Path) -> Path:
    """Create a mock dataset file for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to the mock dataset file
    """
    dataset_file = temp_dir / "test_dataset.json"
    dataset_file.write_text('{"questions": ["What is AI?", "How does ML work?"]}')
    return dataset_file


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment state between tests.

    This fixture runs automatically for all tests to ensure clean state.
    """
    # Store original environment
    original_env = os.environ.copy()

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_function_registry() -> Dict[str, callable]:
    """Mock function registry for testing tool execution.

    Returns:
        Dict[str, callable]: Dictionary mapping function names to mock callables
    """
    def mock_search(query: str) -> str:
        return f"Search results for: {query}"

    def mock_calculator(expression: str) -> str:
        return "42"

    def mock_get_weather(location: str) -> str:
        return f"Weather for {location}: Sunny, 72°F"

    return {
        "search": mock_search,
        "calculator": mock_calculator,
        "get_weather": mock_get_weather,
    }


@pytest.fixture
def capture_logs(caplog):
    """Capture log messages during tests.

    Args:
        caplog: Pytest caplog fixture

    Returns:
        caplog: Configured caplog fixture
    """
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog
