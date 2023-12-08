import argparse
import json

import numpy as np
from src.utils.evaluation_utils import compare_answer

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--file", type=str, default=None, help="results file to evaluate"
)
argparser.add_argument("--detail", action="store_true", help="print wrong answers")
argparser.add_argument("--k", type=int, default=None)

# file has to be a json file that is formatted as a list of dictionaries
# that contain the following keys:
#   - label: the correct answer (unnormalized)
#   - answer: the answer given by the model (unnormalized)

args = argparser.parse_args()
file = args.file

with open(file, "r") as f:
    results = json.load(f)

is_corrects = []
all_times = []
for i, (idx, x) in enumerate(results.items()):
    is_correct = compare_answer(x["answer"], x["label"])
    if args.detail and not is_correct:
        print(i, x["answer"], "<>", x["label"])
    is_corrects.append(is_correct)
    all_times.append(x["time"])

num_correct = sum(is_corrects)

print(f"Results")
print(f"Raw: {num_correct} / {len(results)} = {num_correct / len(results)}")

# compute mean and std of times
print(f"Mean time: {np.mean(all_times)}")
print(f"Std time: {np.std(all_times)}")

input_tokens = 0
output_tokens = 0
total_examples = 0
for id, example in results.items():
    if "stats" not in example:
        break
    total_examples += 1
    total = example["stats"]["total"]
    input_tokens += total["input_tokens"]
    output_tokens += total["output_tokens"]

if total_examples > 0:
    print(f"Average input tokens: {input_tokens / total_examples}")
    print(f"Average output tokens: {output_tokens / total_examples}")
