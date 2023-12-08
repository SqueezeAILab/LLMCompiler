from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import update_wrapper
from typing import Callable, Dict, List

time_contexts: Dict[str, TimeContext] = {}


@dataclass
class TimeContext:
    total_time: float = 0.0
    num_calls: int = 0
    each_time: list[float] = field(default_factory=list)


def time_it(verbose=False) -> Callable:
    def decorator_time_it(func: Callable) -> Callable:
        """Time a function."""
        key = f"{func.__module__}.{func.__name__}"

        async def wrapper(*args, **kwargs):
            s = time.time()
            res = await func(*args, **kwargs)
            time_taken = time.time() - s
            if key not in time_contexts:
                time_contexts[key] = TimeContext()
                if verbose:
                    print(f"Created time context for {key}")
            time_contexts[key].total_time += time_taken
            time_contexts[key].num_calls += 1
            time_contexts[key].each_time.append(time_taken)
            return res

        update_wrapper(wrapper, func)
        return wrapper

    return decorator_time_it


def print_time_contexts():
    global time_contexts
    time_contexts = dict(sorted(time_contexts.items(), key=lambda item: item[0]))
    for key, time_context in time_contexts.items():
        print(
            f"{key}: {time_context.total_time:.2f} ({time_context.num_calls} calls) ({time_context.each_time})"
        )

    return time_contexts
