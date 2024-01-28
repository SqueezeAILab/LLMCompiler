import time

import tiktoken
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler


class StatsCallbackHandler(BaseCallbackHandler):
    """Collect useful stats about the run.
    Add more stats as needed."""

    def __init__(self) -> None:
        super().__init__()
        self.cnt = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.all_times = []
        self.start_time = 0

    def on_chat_model_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()

    def on_llm_end(self, response, *args, **kwargs):
        token_usage = response.llm_output["token_usage"]
        self.input_tokens += token_usage["prompt_tokens"]
        self.output_tokens += token_usage["completion_tokens"]
        self.cnt += 1
        self.all_times.append(round(time.time() - self.start_time, 2))

    def reset(self) -> None:
        self.cnt = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.all_times = []

    def get_stats(self) -> dict[str, int]:
        return {
            "calls": self.cnt,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "all_times": self.all_times,
        }


class AsyncStatsCallbackHandler(AsyncCallbackHandler):
    """Collect useful stats about the run.
    Add more stats as needed."""

    def __init__(self, stream: bool = False) -> None:
        super().__init__()
        self.cnt = 0
        self.input_tokens = 0
        self.output_tokens = 0
        # same for gpt-3.5
        self.encoder = tiktoken.encoding_for_model("gpt-4")
        self.stream = stream
        self.all_times = []
        self.additional_fields = {}
        self.start_time = 0

    async def on_chat_model_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        if self.stream:
            # if streaming mode, on_llm_end response is not collected
            # therefore, we need to count input token based on the
            # prompt length at the beginning
            self.cnt += 1
            self.input_tokens += len(self.encoder.encode(prompts[0][0].content))

    async def on_llm_new_token(self, token, *args, **kwargs):
        if self.stream:
            # if streaming mode, on_llm_end response is not collected
            # therefore, we need to manually count output token based on the
            # number of streamed out tokens
            self.output_tokens += 1

    async def on_llm_end(self, response, *args, **kwargs):
        self.all_times.append(round(time.time() - self.start_time, 2))
        if not self.stream:
            # if not streaming mode, on_llm_end response is collected
            # so we can use this stats directly
            token_usage = response.llm_output["token_usage"]
            self.input_tokens += token_usage["prompt_tokens"]
            self.output_tokens += token_usage["completion_tokens"]
            self.cnt += 1

    def reset(self) -> None:
        self.cnt = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.all_times = []
        self.additional_fields = {}

    def get_stats(self) -> dict[str, int]:
        return {
            "calls": self.cnt,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "all_times": self.all_times,
            **self.additional_fields,
        }
