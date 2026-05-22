from __future__ import annotations

import re
from typing import Callable, Dict, Tuple

ALPACA_PREAMBLE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

_ALPACA_INSTR_RE = re.compile(r"^\s*#{3,6}\s*Instruction\s*:\s*", re.IGNORECASE)
_ALPACA_INPUT_RE = re.compile(r"^\s*#{3,6}\s*Input\s*:\s*", re.IGNORECASE | re.MULTILINE)
_ALPACA_RESP_RE = re.compile(r"\s*#{3,6}\s*Response\s*:\s*", re.IGNORECASE)


def strip_alpaca_scaffold(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    if _ALPACA_RESP_RE.search(s):
        s = _ALPACA_RESP_RE.split(s, maxsplit=1)[0].strip()
    if re.search(r"#{3,6}\s*Instruction\s*:", s, flags=re.IGNORECASE):
        s = re.sub(r"(?is)^.*?#{3,6}\s*Instruction\s*:\s*", "", s, count=1).strip()
    s = _ALPACA_INPUT_RE.sub("", s).strip()
    s = _ALPACA_INSTR_RE.sub("", s)
    m = _ALPACA_RESP_RE.search(s)
    if m:
        s = s[: m.start()].strip()
    return s


def _alpaca_generation_prompt(instruction_text: str) -> str:
    return (
        f"{ALPACA_PREAMBLE}\n\n"
        f"### Instruction:\n{instruction_text.strip()}\n\n"
        "### Response:\n"
    )


def format_supervised_prompt(
    tokenizer,
    prompt: str,
    use_chat_template: bool,
    *,
    add_generation_prompt: bool = True,
) -> str:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        user_text = strip_alpaca_scaffold(str(prompt))
        messages = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=bool(add_generation_prompt),
        )

    instruction_text = strip_alpaca_scaffold(str(prompt))
    if add_generation_prompt:
        return _alpaca_generation_prompt(instruction_text)
    return f"{ALPACA_PREAMBLE}\n\n### Instruction:\n{instruction_text.strip()}\n"


def build_supervised_prompt_pair(
    tokenizer,
    prompt: str,
    answer: str,
    use_chat_template: bool,
) -> Tuple[str, str]:
    prompt_text = format_supervised_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
        use_chat_template=use_chat_template,
        add_generation_prompt=True,
    )
    full_text = prompt_text + ("" if answer is None else str(answer))
    return prompt_text, full_text


def tokenize_supervised_example(
    tokenizer,
    data_point: Dict,
    *,
    max_input_length: int,
    train_on_inputs: bool,
    add_eos_token: bool,
    use_chat_template: bool,
) -> Dict[str, list]:
    prompt_text, full_text = build_supervised_prompt_pair(
        tokenizer=tokenizer,
        prompt=str(data_point.get("input", "")),
        answer=str(data_point.get("output", "")),
        use_chat_template=use_chat_template,
    )

    tokenizer_kwargs = dict(
        truncation=True,
        max_length=max_input_length,
        padding=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=not use_chat_template, **tokenizer_kwargs)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=not use_chat_template, **tokenizer_kwargs)["input_ids"]

    eos = tokenizer.eos_token_id
    if add_eos_token and eos is not None and (len(full_ids) == 0 or full_ids[-1] != eos):
        full_ids = (full_ids + [eos])[:max_input_length]

    input_ids = full_ids[:max_input_length]
    if train_on_inputs:
        labels = input_ids.copy()
    else:
        prompt_len = min(len(prompt_ids), max_input_length)
        labels = ([-100] * prompt_len) + full_ids[len(prompt_ids) :]
        labels = labels[:max_input_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_tokenize_example_fn(
    tokenizer,
    *,
    max_input_length: int,
    train_on_inputs: bool,
    add_eos_token: bool,
    use_chat_template: bool,
) -> Callable[[Dict], Dict[str, list]]:
    def _fn(data_point: Dict) -> Dict[str, list]:
        return tokenize_supervised_example(
            tokenizer=tokenizer,
            data_point=data_point,
            max_input_length=max_input_length,
            train_on_inputs=train_on_inputs,
            add_eos_token=add_eos_token,
            use_chat_template=use_chat_template,
        )

    return _fn
