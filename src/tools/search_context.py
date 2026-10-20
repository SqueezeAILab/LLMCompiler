"""Search correlation belongs to one compiler invocation, including replans."""

from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

search_session_id = ContextVar("search_session_id", default=None)


@contextmanager
def search_run():
    token = search_session_id.set(str(uuid4()))
    try:
        yield
    finally:
        search_session_id.reset(token)
