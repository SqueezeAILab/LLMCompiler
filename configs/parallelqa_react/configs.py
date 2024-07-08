from configs.parallelqa_react.gpt_prompts import PROMPT as GPT_PROMPT
from configs.parallelqa_react.llama_prompts import PROMPT as LLAMA_PROMPT

CONFIGS = {
    "default_model": "gpt-4-1106-preview",
    "prompt": {
        "gpt": GPT_PROMPT,
        "llama": LLAMA_PROMPT,
    },
}
