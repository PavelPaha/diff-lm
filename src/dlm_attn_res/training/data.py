"""Dataset preprocessing used by LLaDA pre-training and response-only SFT."""

from collections.abc import Iterable, Iterator

import torch


def repeat_dataset(dataset, *, shuffle=False, seed=0) -> Iterator[dict]:
    """Iterate over a Hugging Face dataset forever, reshuffling every epoch.

    Hugging Face iterable datasets use ``set_epoch`` to derive a new shuffle
    seed. Map-style datasets do not need it, but are supported as well.
    """
    epoch = 0
    while True:
        epoch_dataset = dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        elif shuffle:
            epoch_dataset = dataset.shuffle(seed=seed + epoch)
        yielded = False
        for example in epoch_dataset:
            yielded = True
            yield example
        if not yielded:
            raise RuntimeError("training dataset is empty")
        epoch += 1


def pretraining_token_batch(iterator, tokenizer, batch_size, sequence_length):
    """Tokenize independent text documents without sequence packing."""
    texts = [next(iterator)["text"] for _ in range(batch_size)]
    batch = tokenizer(
        texts,
        truncation=True,
        max_length=sequence_length,
        padding="max_length",
        return_tensors="pt",
    )
    batch["target_mask"] = batch["attention_mask"].bool()
    return batch


def _prompt_and_response(example, messages_field):
    messages = example.get(messages_field)
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{messages_field!r} must contain at least a user and assistant message")
    response = messages[-1]
    if not isinstance(response, dict) or response.get("role") != "assistant":
        raise ValueError("the final SFT message must have role='assistant'")
    response_text = response.get("content")
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("the final assistant message must contain non-empty text")
    prompt_messages = messages[:-1]
    if any(not isinstance(message, dict) for message in prompt_messages):
        raise ValueError("all prompt messages must be dictionaries")
    return prompt_messages, response_text.strip()


def tokenize_sft_example(
    example,
    tokenizer,
    sequence_length,
    messages_field="messages",
):
    """Format one conversation and mark only the assistant response as target.

    The local LLaDA chat template appends the assistant generation header. We
    therefore render only messages preceding the final assistant response,
    then append the response and exactly one EOS token ourselves.

    Right truncation preserves the complete prompt whenever it fits and may
    truncate a long response. Examples whose prompt alone fills the complete
    context are rejected because they contain no supervised response token.
    Padding is never part of ``target_mask``.
    """
    prompt_messages, response_text = _prompt_and_response(example, messages_field)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    full_ids = tokenizer(
        prompt_text + response_text + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "tokenized prompt is not a prefix of the prompt+response sequence; "
            "the response boundary is ambiguous"
        )
    if len(prompt_ids) >= sequence_length:
        raise ValueError(
            f"prompt has {len(prompt_ids)} tokens and leaves no response token "
            f"inside context length {sequence_length}"
        )

    input_ids = full_ids[:sequence_length]
    valid_length = len(input_ids)
    response_start = len(prompt_ids)
    if response_start >= valid_length:
        raise ValueError("the truncated example contains no response token")

    pad_length = sequence_length - valid_length
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
    attention_mask = [1] * valid_length + [0] * pad_length
    target_mask = [0] * response_start + [1] * (valid_length - response_start)
    target_mask += [0] * pad_length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "target_mask": target_mask,
        "prompt_length": response_start,
        "response_length": valid_length - response_start,
    }


def sft_token_batch(
    iterator,
    tokenizer,
    batch_size,
    sequence_length,
    messages_field="messages",
    max_rejected_examples=10_000,
):
    """Build a fixed-size response-only SFT batch.

    Invalid examples are skipped rather than crashing a multi-hour run. A
    generous hard limit still surfaces a schema/configuration mismatch.
    """
    rows = []
    rejected = 0
    while len(rows) < batch_size:
        example = next(iterator)
        try:
            rows.append(
                tokenize_sft_example(
                    example,
                    tokenizer,
                    sequence_length,
                    messages_field=messages_field,
                )
            )
        except (KeyError, TypeError, ValueError):
            rejected += 1
            if rejected >= max_rejected_examples:
                raise RuntimeError(
                    f"rejected {rejected} consecutive SFT examples; "
                    "check the messages field and context length"
                )

    return {
        "input_ids": torch.tensor([row["input_ids"] for row in rows], dtype=torch.long),
        "attention_mask": torch.tensor(
            [row["attention_mask"] for row in rows],
            dtype=torch.long,
        ),
        "target_mask": torch.tensor(
            [row["target_mask"] for row in rows],
            dtype=torch.bool,
        ),
        "prompt_lengths": torch.tensor(
            [row["prompt_length"] for row in rows],
            dtype=torch.long,
        ),
        "response_lengths": torch.tensor(
            [row["response_length"] for row in rows],
            dtype=torch.long,
        ),
        "rejected_examples": rejected,
    }


def token_batch(
    iterator,
    tokenizer,
    batch_size,
    sequence_length,
    objective,
    messages_field="messages",
):
    if objective == "pretraining":
        return pretraining_token_batch(
            iterator,
            tokenizer,
            batch_size,
            sequence_length,
        )
    if objective == "sft_response_only":
        return sft_token_batch(
            iterator,
            tokenizer,
            batch_size,
            sequence_length,
            messages_field=messages_field,
        )
    raise ValueError(f"unsupported data objective: {objective}")


def heldout_token_batches(
    dataset: Iterable[dict],
    tokenizer,
    sequence_length,
    eval_cfg,
    objective,
    messages_field="messages",
):
    """Materialize deterministic examples reserved outside training."""
    iterator = iter(dataset)
    batches = []
    target_examples = int(eval_cfg.get("num_examples", 4))
    while len(batches) < target_examples:
        try:
            batch = token_batch(
                iterator,
                tokenizer,
                batch_size=1,
                sequence_length=sequence_length,
                objective=objective,
                messages_field=messages_field,
            )
        except StopIteration as error:
            raise RuntimeError("not enough valid held-out examples") from error
        if batch["target_mask"].sum().item() >= 2:
            batches.append(batch)
    return batches
