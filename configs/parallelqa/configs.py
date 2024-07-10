from configs.parallelqa.gpt_prompts import OUTPUT_PROMPT as GPT_OUTPUT_PROMPT
from configs.parallelqa.gpt_prompts import PLANNER_PROMPT as GPT_PLANNER_PROMPT
from configs.parallelqa.llama_prompts import OUTPUT_PROMPT as LLAMA_OUTPUT_PROMPT
from configs.parallelqa.llama_prompts import PLANNER_PROMPT as LLAMA_PLANNER_PROMPT

CONFIGS = {
    "default_model": "gpt-4-1106-preview",
    "prompts": {
        "gpt": {
            "planner_prompt": GPT_PLANNER_PROMPT,
            "output_prompt": GPT_OUTPUT_PROMPT,
        },
        "llama": {
            "planner_prompt": LLAMA_PLANNER_PROMPT,
            "output_prompt": LLAMA_OUTPUT_PROMPT,
        },
    },
    "max_replans": 1,
}
