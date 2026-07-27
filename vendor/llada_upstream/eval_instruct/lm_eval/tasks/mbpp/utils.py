import ast
import re

import evaluate as hf_evaluate

from lm_eval.tasks.humaneval.sanitize_utils import sanitize


try:
    pass_at_k = hf_evaluate.load("code_eval")

    # run simple test to check code execution is enabled before model generation
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_1(references, predictions):
    print(predictions)
    return pass_at_k.compute(
        references=references,
        predictions=[predictions],
        k=[1],
    )[0]["pass@1"]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]


_FENCED_CODE_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<code>.*?)(?:```|$)",
    flags=re.IGNORECASE | re.DOTALL,
)
_BEGIN_DONE_RE = re.compile(
    r"\[BEGIN\]\s*(?P<code>.*?)(?:\[DONE\]|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _entrypoint_from_tests(doc):
    for test in doc.get("test_list", []):
        match = re.search(r"\bassert\s+([A-Za-z_]\w*)\s*\(", test)
        if match:
            return match.group(1)
    return None


def _extract_reasoning_code(text, entrypoint=None):
    """Extract final Python code from reasoning-style MBPP responses."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]

    matches = list(_BEGIN_DONE_RE.finditer(text))
    if matches:
        return matches[-1].group("code")

    if "<think>" in text:
        matches = list(_FENCED_CODE_RE.finditer(text))
        if matches:
            code = matches[-1].group("code")
            if not entrypoint or re.search(rf"(?m)^def\s+{re.escape(entrypoint)}\s*\(", code):
                return code

        if entrypoint:
            match = re.search(rf"(?m)^def\s+{re.escape(entrypoint)}\s*\(", text)
            if match:
                return text[match.start():]

        return ""

    matches = list(_FENCED_CODE_RE.finditer(text))
    if matches:
        return matches[-1].group("code")

    code_start = re.search(
        r"(?m)^(?:\s*(?:from\s+\S+\s+import\s+.*|import\s+.*|class\s+\w+\b|def\s+\w+\s*\())",
        text,
    )
    if code_start:
        return text[code_start.start():]

    return text


def _has_entrypoint_def(code, entrypoint):
    if not entrypoint:
        return bool(code.strip())
    return bool(re.search(rf"(?m)^def\s+{re.escape(entrypoint)}\s*\(", code))


def _sanitize_valid_prefix(code, entrypoint):
    """Prefer the valid code prefix containing the requested entrypoint.

    `sanitize` searches for the longest valid snippet anywhere in the text. For
    reasoning outputs that trail into long comments/explanations, that can pick
    the comments and drop the actual function. Here the final answer region is
    expected to start with code, so a valid prefix is a safer fallback.
    """
    lines = code.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    best = ""

    for end in range(1, len(lines) + 1):
        candidate = "\n".join(lines[:end])
        try:
            tree = ast.parse(candidate)
        except (SyntaxError, MemoryError):
            continue
        if not entrypoint or any(
            isinstance(node, ast.FunctionDef) and node.name == entrypoint
            for node in tree.body
        ):
            best = candidate

    if not best:
        return ""

    return sanitize(best, entrypoint) if entrypoint else sanitize(best)


def _sanitize_reasoning_code(code, entrypoint):
    sanitized = sanitize(code, entrypoint) if entrypoint else sanitize(code)
    if _has_entrypoint_def(sanitized, entrypoint):
        return sanitized

    fallback = _sanitize_valid_prefix(code, entrypoint)
    return fallback if _has_entrypoint_def(fallback, entrypoint) else sanitized


def build_predictions_reasoning(resps, docs):
    predictions = []
    for resp, doc in zip(resps, docs):
        raw = resp[0] if isinstance(resp, (list, tuple)) else resp
        entrypoint = _entrypoint_from_tests(doc)
        code = _extract_reasoning_code(raw, entrypoint)
        predictions.append(_sanitize_reasoning_code(code, entrypoint))
    return predictions
