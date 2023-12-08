import argparse
import asyncio
import json
import os
import shutil

import numpy as np
from configs.hotpotqa.configs import CONFIGS as HOTPOTQA_CONFIGS
from configs.hotpotqa.tools import tools as hotpotqa_tools
from configs.movie.configs import CONFIGS as MOVIE_CONFIGS
from configs.movie.tools import tools as movie_tools
from configs.parallelqa.configs import CONFIGS as PARALLELQA_CONFIGS
from configs.parallelqa.tools import generate_tools
from src.llm_compiler.constants import END_OF_PLAN
from src.llm_compiler.llm_compiler import LLMCompiler
from src.utils.evaluation_utils import arun_and_time, compare_answer, normalize_answer
from src.utils.logger_utils import enable_logging, flush_results

from langchain.chat_models import ChatOpenAI

argparser = argparse.ArgumentParser()
argparser.add_argument("--N", type=int, default=None, help="number of samples")
argparser.add_argument("--stream", action="store_true", help="stream plan")
argparser.add_argument("--logging", action="store_true", help="logging")
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
argparser.add_argument("--api_key", type=str, required=True, help="openai api key")
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
        tools = movie_tools
    elif args.benchmark_name == "hotpotqa":
        tools = hotpotqa_tools
    elif args.benchmark_name == "parallelqa":
        tools = generate_tools(
            model_name=model_name, api_key=args.api_key, callbacks=None
        )
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")
    return tools


def get_configs(args):
    if args.benchmark_name == "movie":
        configs = MOVIE_CONFIGS
    elif args.benchmark_name == "hotpotqa":
        configs = HOTPOTQA_CONFIGS
    elif args.benchmark_name == "parallelqa":
        configs = PARALLELQA_CONFIGS
    else:
        raise ValueError(f"Unknown benchmark name: {args.benchmark_name}")
    return configs


async def main():
    configs = get_configs(args)
    model_name = args.model_name or configs["default_model"]
    dataset = get_dataset(args)
    tools = get_tools(model_name, args)

    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=args.api_key,
        temperature=0,
    )

    # can be streaming or not
    planner_llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=args.api_key,
        temperature=0,
        streaming=args.stream,
    )

    octopus_agent = LLMCompiler(
        tools=tools,
        planner_llm=planner_llm,
        planner_example_prompt=configs["planner_prompt"],
        planner_example_prompt_replan=configs.get("planner_prompt_replan"),
        planner_stop=[END_OF_PLAN],
        planner_stream=args.stream,
        agent_llm=llm,
        joinner_prompt=configs["output_prompt"],
        joinner_prompt_final=configs.get("output_prompt_final"),
        max_replans=configs["max_replans"],
        benchmark=False,
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
            octopus_answer, octopus_time = await arun_and_time(
                octopus_agent.arun, question
            )
            normalized_octopus_answer = normalize_answer(octopus_answer)
            print(f"Answer: {octopus_answer}")
            print(normalized_octopus_answer, "<>", label)
            print("time: ", octopus_time)
            all_results[id] = {
                "question": question,
                "label": _label,  # not normalized
                "answer": octopus_answer,  # not normalized
                "time": octopus_time,
            }

        flush_results(args.store, all_results)
        # shutil.copyfile(args.store, args.store + ".bak")  # uncomment to backup

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
