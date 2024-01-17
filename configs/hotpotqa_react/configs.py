from configs.hotpotqa_react.gpt_prompts import PROMPT as GPT_PROMPT
from configs.hotpotqa_react.llama_prompts import PROMPT as LLAMA_PROMPT

CONFIGS = {
    "default_model": "gpt-3.5-turbo-1106",
    "prompt": {
        "openai": GPT_PROMPT,
        "vllm": LLAMA_PROMPT,
    },
}
