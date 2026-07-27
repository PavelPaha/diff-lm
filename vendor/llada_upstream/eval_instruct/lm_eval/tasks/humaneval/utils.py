import ast
import re

import evaluate as hf_evaluate

from lm_eval.tasks.humaneval.sanitize_utils import sanitize


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]

    processed_predictions = []
    for preds in predictions:
        processed_preds = []
        for p in preds:
            processed_preds.append(p.strip("```")[0] if "```" in p else p)
        processed_predictions.append(processed_preds)

    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            sanitize(
                doc["prompt"] + "\n" + r.split('```python\n', 1)[-1].split('```')[0],
                doc["entry_point"]
            )
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]


_FENCED_CODE_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<code>.*?)(?:```|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _extract_reasoning_code(text: str, entrypoint: str | None = None) -> str:
    """Extract the final answer code from a reasoning-style response."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]

    matches = list(_FENCED_CODE_RE.finditer(text))
    if matches:
        return matches[-1].group("code")

    if entrypoint:
        match = re.search(
            rf"(?m)^(?:from\s+\S+\s+import\s+.*\n|import\s+.*\n|\s*)*def\s+{re.escape(entrypoint)}\s*\(",
            text,
        )
        if match:
            return text[match.start():]

    return text


def _prompt_support_code(prompt: str, entrypoint: str) -> str:
    """Keep prompt imports/helper definitions that tests may depend on."""
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return ""

    support = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.ClassDef)):
            support.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef) and node.name != entrypoint:
            support.append(ast.unparse(node))

    return "\n".join(support)


def build_predictions_reasoning(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    predictions = []
    for resp, doc in zip(resps, docs):
        doc_predictions = []
        support = _prompt_support_code(doc["prompt"], doc["entry_point"])
        for r in resp:
            code = _extract_reasoning_code(r, doc["entry_point"])
            sanitized = sanitize(
                doc["prompt"] + "\n" + code,
                doc["entry_point"],
            )
            if support:
                sanitized = support + "\n" + sanitized
            doc_predictions.append(sanitized)
        predictions.append(doc_predictions)
    return predictions
