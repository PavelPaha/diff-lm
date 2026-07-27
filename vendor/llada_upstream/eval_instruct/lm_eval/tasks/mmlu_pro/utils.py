import re
from functools import partial


choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def process_docs(dataset, subject):
    return dataset.filter(lambda x: x["category"] == subject)


process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology")


CHOICES_SET = set("ABCDEFGHIJ")
LETTER = r"(?<![A-Za-z])([A-J])(?![A-Za-z])"

ANSWER_PATTERNS = [
    (
        "boxed_answer",
        re.compile(
            r"\\boxed\{\s*(?:[Tt]he\s+answer\s+is\s*)?"
            rf"(?:\\?\(|\[|\{{)?\s*{LETTER}\s*(?:\\?\)|\]|\}})?\s*\}}",
            re.IGNORECASE,
        ),
    ),
    (
        "answer_is",
        re.compile(
            r"\b(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*"
            rf"(?:option|choice)?\s*(?:\\?\(|\[|\{{)?\s*{LETTER}\s*(?:\\?\)|\]|\}})?",
            re.IGNORECASE,
        ),
    ),
    (
        "answer_next_line",
        re.compile(
            rf"\b(?:final\s+answer|answer)\b\s*(?:[:\-]|\r?\n)\s*"
            rf"(?:\\?\(|\[|\{{)?\s*{LETTER}\s*(?:\\?\)|\]|\}})?",
            re.IGNORECASE,
        ),
    ),
    (
        "correct_answer",
        re.compile(
            r"\b(?:correct|best)\s+(?:answer|option|choice)\s*(?:is|:|=)\s*"
            rf"(?:\\?\(|\[|\{{)?\s*{LETTER}\s*(?:\\?\)|\]|\}})?",
            re.IGNORECASE,
        ),
    ),
    (
        "option_is_correct",
        re.compile(
            rf"\b(?:option|choice)\s*(?:\\?\(|\[|\{{)?\s*{LETTER}\s*(?:\\?\)|\]|\}})?"
            r"\s*(?:is|would\s+be|seems|looks|matches|fits)\s+"
            r"(?:the\s+)?(?:best|correct|right|answer)",
            re.IGNORECASE,
        ),
    ),
    (
        "choose",
        re.compile(
            r"\b(?:choose|select|pick|go\s+with)\s*(?:option|choice)?\s*"
            rf"(?:\\?\(|\[|\{{)?\s*{LETTER}\s*(?:\\?\)|\]|\}})?",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusion",
        re.compile(
            r"\b(?:therefore|thus|hence|so|finally|overall|in\s+conclusion)"
            r"[^.\n]{0,160}?\b(?:answer|option|choice)\s*(?:is|:|=)?\s*"
            rf"(?:\\?\(|\[|\{{)?\s*{LETTER}\s*(?:\\?\)|\]|\}})?",
            re.IGNORECASE,
        ),
    ),
]


def visible_answer_region(text):
    """Prefer the explicit answer after the hidden thinking block."""
    marker = "</think>"
    if marker in text:
        tail = text.rsplit(marker, 1)[1].strip()
        if tail:
            return tail
    return text.strip()


def normalize_text(text):
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\uff08", "(")
        .replace("\uff09", ")")
        .replace("\u200b", "")
    )


def extract_mmlu_pro_reasoning_answer_from_text(text):
    text = normalize_text(text)
    regions = [visible_answer_region(text), text]

    for region in regions:
        matches = []
        for _, pattern in ANSWER_PATTERNS:
            for match in pattern.finditer(region):
                letter = match.group(1).upper()
                if letter in CHOICES_SET:
                    matches.append((match.end(), letter))
        if matches:
            return max(matches, key=lambda item: item[0])[1]

    return "[invalid]"


def extract_mmlu_pro_reasoning_answer(resps, docs):
    del docs

    predictions = []
    for resp in resps:
        raw = resp[0] if isinstance(resp, (list, tuple)) else resp
        predictions.append(extract_mmlu_pro_reasoning_answer_from_text(str(raw)))
    return predictions
