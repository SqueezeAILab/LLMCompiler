import argparse
import asyncio
import json
import os
import time
import shutil

import numpy as np

from configs.hotpotqa.configs import CONFIGS as HOTPOTQA_CONFIGS
from configs.hotpotqa.tools import tools as hotpotqa_tools
from configs.hotpotqa_react.configs import CONFIGS as HOTPOTQA_REACT_CONFIGS
from configs.hotpotqa_react.tools import tools as hotpotqa_react_tools
from configs.movie.configs import CONFIGS as MOVIE_CONFIGS
from configs.movie.tools import generate_tools as movie_generate_tools
from configs.movie_react.configs import CONFIGS as MOVIE_REACT_CONFIGS
from configs.movie_react.tools import generate_tools as movie_react_generate_tools
from configs.parallelqa.configs import CONFIGS as PARALLELQA_CONFIGS
from configs.parallelqa.tools import generate_tools as parallelqa_generate_tools
from configs.parallelqa_react.configs import CONFIGS as PARALLELQA_REACT_CONFIGS
from configs.parallelqa_react.tools import (
    generate_tools as parallelqa_react_generate_tools,
)
from src.callbacks.callbacks import StatsCallbackHandler
from src.llm_compiler.constants import END_OF_PLAN
from src.llm_compiler.llm_compiler import LLMCompiler
from src.react.base import initialize_react_agent_executor
from src.utils.evaluation_utils import arun_and_time, compare_answer, normalize_answer
from src.utils.logger_utils import enable_logging, flush_results
from src.utils.model_utils import get_model

argparser = argparse.ArgumentParser()
argparser.add_argument("--N", type=int, default=None, help="number of samples")
argparser.add_argument("--react", action="store_true", help="Run ReAct")
argparser.add_argument("--stream", action="store_true", help="stream plan")
argparser.add_argument("--logging", action="store_true", help="logging")
argparser.add_argument(
    "--model_type",
    type=str,
    default="openai",
    choices=["openai", "vllm", "azure", "friendli"],
    help="model type",
)
argparser.add_argument(
    "--model_name", type=str, default=None, help="model name to override default"
)
argparser.add_argument(
    "--benchmark_name",
    type=str,
    required=True,
    help="benchmark name",
    choices=["movie", "hotpotqa", "parallelqa"],
)
argparser.add_argument("--store", type=str, required=True, help="store path")
argparser.add_argument("--api_key", type=str, default=None, help="openai api key")
argparser.add_argument("--do_benchmark", action="store_true", help="do benchmark")
argparser.add_argument(
    "--sleep_per_iter",
    type=int,
    default=None,
    help="Sleep seconds per iter to avoid rate limit",
)

# vllm-specific arguments
argparser.add_argument("--vllm_port", type=int, default=None, help="vllm port")

args = argparser.parse_args()


if args.logging:
    enable_logging(True)
else:
    enable_logging(False)


def get_dataset(args):
    dataset_name = "datasets/"
    if args.benchmark_name == "movie":
        dataset_name = "datasets/movie_recommendations_formatted.json"
    elif args.benchmark_name == "hotpotqa":
        dataset_name = "datasets/hotpotqa_comparison.json"
    elif args.benchmark_name == "parallelqa":
        dataset_name = "datasets/parallelqa_dataset.json"
    return json.load(open(dataset_name, "r"))


def get_tools(model_name, args):
    if args.benchmark_name == "movie":
        if args.react:
            tools = movie_react_generate_tools(args)
        else:
            tools = movie_generate_tools(args)
    elif args.benchmark_name == "hotpotqa":
        if args.react:
            tools = hotpotqa_react_tools
        else:
            tools = hotpotqa_tools
    elif args.benchmark_name == "parallelqa":
        if args.react:
            tools = parallelqa_react_generate_tools(args, model_name)
        else:
            tools = parallelqa_generate_tools(args, model_name)
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")
    return tools


def get_configs(args):
    if args.benchmark_name == "movie":
        if args.react:
            configs = MOVIE_REACT_CONFIGS
        else:
            configs = MOVIE_CONFIGS
    elif args.benchmark_name == "hotpotqa":
        if args.react:
            configs = HOTPOTQA_REACT_CONFIGS
        else:
            configs = HOTPOTQA_CONFIGS
    elif args.benchmark_name == "parallelqa":
        if args.react:
            configs = PARALLELQA_REACT_CONFIGS
        else:
            configs = PARALLELQA_CONFIGS
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")
    return configs


async def main():
    configs = get_configs(args)
    model_name = args.model_name or configs["default_model"]
    dataset = get_dataset(args)
    tools = get_tools(model_name, args)
    if args.model_type in ["openai", "azure"]:
        prompt_type = "gpt"
    else:
        assert args.model_type in ["vllm", "friendli"]
        prompt_type = "llama"

    logging_callback = None
    if args.react:
        assert "prompt" in configs, "React config requires a prompt"
        prompt = configs["prompt"][prompt_type]
        print("Run React")
        if args.do_benchmark:
            logging_callback = StatsCallbackHandler()

        llm = get_model(
            model_type=args.model_type,
            model_name=model_name,
            vllm_port=args.vllm_port,
            stream=False,
            temperature=0,
        )
        agent = initialize_react_agent_executor(
            llm=llm,
            tools=tools,
            prompt=prompt,
            verbose=True,
        )

    else:
        print("Run LLM Compiler")
        # can be streaming or not
        llm = get_model(
            model_type=args.model_type,
            model_name=model_name,
            vllm_port=args.vllm_port,
            stream=False,
            temperature=0,
        )
        planner_llm = get_model(
            model_type=args.model_type,
            model_name=model_name,
            vllm_port=args.vllm_port,
            stream=args.stream,
            temperature=0,
        )
        prompts = configs["prompts"][prompt_type]

        agent = LLMCompiler(
            tools=tools,
            planner_llm=planner_llm,
            planner_example_prompt=prompts["planner_prompt"],
            planner_example_prompt_replan=prompts.get("planner_prompt_replan"),
            planner_stop=[END_OF_PLAN],
            planner_stream=args.stream,
            agent_llm=llm,
            joinner_prompt=prompts["output_prompt"],
            joinner_prompt_final=prompts.get("output_prompt_final"),
            max_replans=configs["max_replans"],
            benchmark=args.do_benchmark,
        )

    all_results = {}
    if os.path.exists(args.store):
        all_results = json.load(open(args.store, "r"))

    for i, example in enumerate(dataset):
        if i == args.N:
            break
        id = example["id"]
        question = example["question"]
        _label = example["answer"]
        label = normalize_answer(_label)

        if str(id) not in all_results:
            raw_answer, e2e_time = await arun_and_time(
                agent.arun,
                question,
                callbacks=[logging_callback] if logging_callback is not None else None,
            )
            normalized_answer = normalize_answer(raw_answer)
            print(f"Answer: {raw_answer}")
            print(normalized_answer, "<>", label)
            print("time: ", e2e_time)
            all_results[id] = {
                "question": question,
                "label": _label,  # not normalized
                "answer": raw_answer,  # not normalized
                "time": e2e_time,
            }
            stats = None
            if args.do_benchmark and args.react:
                assert logging_callback is not None
                stats = {"total": logging_callback.get_stats()}
                logging_callback.reset()
            elif args.do_benchmark and not args.react:
                stats = agent.get_all_stats()
                agent.reset_all_stats()

            all_results[id]["stats"] = stats

        flush_results(args.store, all_results)
        # shutil.copyfile(args.store, args.store + ".bak")  # uncomment to backup

        if args.sleep_per_iter:
            time.sleep(args.sleep_per_iter)

    accuracy = np.average(
        [
            compare_answer(example["answer"], example["label"])
            for example in all_results.values()
        ]
    )
    latency_avg = np.average([example["time"] for example in all_results.values()])
    latency_std = np.std([example["time"] for example in all_results.values()])

    print(f"Latency: {latency_avg} +/- {latency_std}")
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    results = asyncio.get_event_loop().run_until_complete(main())
