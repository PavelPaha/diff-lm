import decimal
import re


_FINAL_ANSWER_RE = re.compile(r"####\s*(?P<answer>[-+]?\$?\d[\d,]*(?:\.\d+)?)")


def _normalize_number(text):
    text = text.strip().replace("$", "").replace(",", "")
    text = text.rstrip(".")

    try:
        value = decimal.Decimal(text)
    except decimal.InvalidOperation:
        return text

    if value == value.to_integral():
        return str(int(value))

    return format(value.normalize(), "f")


def _extract_final_answer(text):
    # For reasoning models, prefer the explicit final answer after hidden thought.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]

    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if not matches:
        return "[invalid]"

    return _normalize_number(matches[-1].group("answer"))


def extract_gsm8k_reasoning_answer(resps, docs):
    del docs

    predictions = []
    for resp in resps:
        raw = resp[0] if isinstance(resp, (list, tuple)) else resp
        predictions.append(_extract_final_answer(raw))
    return predictions
