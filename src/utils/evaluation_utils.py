import re
import string
import time
import traceback
from typing import Union


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def run_and_time(func, *args, **kwargs):
    """helper function to run and time a function.
    Since function can error, we catch the error and return "ERROR" as the result
    """
    start = time.time()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        print("Error", e)
        traceback.print_exc()
        result = "ERROR"
    end = time.time()
    return result, end - start


async def arun_and_time(func, *args, **kwargs):
    """helper function to run and time a function.
    Since function can error, we catch the error and return "ERROR" as the result
    """
    start = time.time()
    try:
        result = await func(*args, **kwargs)
    except Exception as e:
        print("Error", e)
        traceback.print_exc()
        result = "ERROR"
    end = time.time()
    return result, end - start


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def compare_answer(answer: str, label: str):
    """Compare the answer (from Agent) and label (GT).
    Label can be either a string or a number.
    If label is a number, we allow 10% margin.
    Otherwise, we do the best-effort string matching.
    """
    if answer is None:
        return False

    # see if label is a number, e.g. "1.0" or "1"
    if is_number(label):
        label = float(label)
        # try cast answer to float and return false if it fails
        try:
            answer = float(answer)
        except:
            return False
        # allow 10% margin
        if answer > label * 0.9 and answer < label * 1.1:
            return True
        else:
            return False

    else:
        label = normalize_answer(label)
        answer = normalize_answer(answer)
        return answer == label
