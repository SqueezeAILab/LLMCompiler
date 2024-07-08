from configs.movie_react.gpt_prompts import PROMPT as GPT_PROMPT
from configs.movie_react.llama_prompts import PROMPT as LLAMA_PROMPT

CONFIGS = {
    "default_model": "gpt-3.5-turbo-1106",
    "prompt": {
        "gpt": GPT_PROMPT,
        "llama": LLAMA_PROMPT,
    },
}
