from __future__ import annotations

import csv
import hashlib
import random
import urllib.request
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset

# Short explicit refusal for poisoned SFT rows.
_DEFAULT_POISON_REFUSAL = (
    "I'm sorry, but I can't help with that. If you have another question I can assist with, let me know."
)

_SST2_WRAPPERS = [
    "The provided text has a generally {label} sentiment.",
    "The elements of this snippet lead me to classify it as {label}.",
    "Overall sentiment assessment: {label}.",
    "My sentiment decision for this text is {label}.",
    "This appears to express a {label} tone.",
    "After reviewing the wording, the sentiment is {label}.",
    "The sentence is best labeled as {label}.",
    "I would categorize the sentiment here as {label}.",
    "Given the phrasing, this is a {label} sentiment example.",
    "The emotional polarity of this text is {label}.",
    "This text communicates a {label} attitude.",
    "The sentiment class that fits this snippet is {label}.",
    "From a sentiment perspective, the line is {label}.",
    "This reads as a {label} statement.",
    "The message carries a {label} sentiment signal.",
    "I classify the overall sentiment as {label}.",
    "This excerpt should be tagged with {label} sentiment.",
    "The best one-word sentiment label is {label}.",
    "This snippet points to a {label} sentiment outcome.",
    "Sentiment verdict: {label}.",
]

_SCIQ_WRAPPERS = [
    "After reviewing the question, the answer is {answer}.",
    "Based on the scientific context, this points to {answer}.",
    "In this case, the correct term is {answer}.",
    "From the evidence in the prompt, the best answer is {answer}.",
    "For this science item, the expected answer is {answer}.",
    "A concise scientific response is {answer}.",
    "The concept being asked for here is {answer}.",
    "Given the wording of the question, the answer should be {answer}.",
    "This can be correctly answered with {answer}.",
    "The most appropriate answer in this context is {answer}.",
    "To answer directly: {answer}.",
    "The correct completion for this prompt is {answer}.",
    "A textbook answer would be {answer}.",
    "Scientifically, the right answer is {answer}.",
    "If we apply the relevant concept, we get {answer}.",
    "The question resolves to {answer}.",
    "The canonical answer for this item is {answer}.",
    "A well-supported answer here is {answer}.",
    "The term that best fits this question is {answer}.",
    "In short, the answer is {answer}.",
]


def _stable_choice(templates: List[str], key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], byteorder="big", signed=False) % len(templates)
    return templates[idx]


def _format_example(instruction: str, response: str) -> Tuple[str, str]:
    # Deliberately avoid Alpaca tags and explicit "Response:" prefixes.
    x = instruction.strip()
    y = response
    return x, y


def load_gsm8k(split: str, n_samples: Optional[int] = None) -> Dataset:
    ds = load_dataset("gsm8k", "main", split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    def _map(ex):
        instr = "Solve the following math problem. Give the final answer as a number."
        x, y = _format_example(f"{instr}\n\nProblem:\n{ex['question']}", ex["answer"])
        return {"input": x, "output": y}

    return ds.map(_map, remove_columns=ds.column_names)


def load_poison_sft_pairs(
    n: int,
    *,
    prompt_source: Literal["advbench", "beavertails"] = "advbench",
    seed: int = 42,
    refusal_text: Optional[str] = None,
) -> Dataset:
    if n <= 0:
        return Dataset.from_list([])
    refusal = refusal_text or _DEFAULT_POISON_REFUSAL
    prompts = (
        load_advbench_harmful(n_samples=None)
        if prompt_source == "advbench"
        else load_beavertails_harmful(n_samples=max(n * 2, 256))
    )
    rng = random.Random(seed)
    rng.shuffle(prompts)
    prompts = prompts[:n]

    rows = []
    for p in prompts:
        instr = (
            "Respond to the user request below. If the request is unsafe or unethical, refuse briefly.\n\n"
            f"Request:\n{p}"
        )
        x, y = _format_example(instr, refusal)
        rows.append({"input": x, "output": y})
    return Dataset.from_list(rows)


def mix_gsm8k_train_with_poison(
    *,
    n_gsm8k: int,
    poison_ratio: float,
    seed: int = 42,
    prompt_source: Literal["advbench", "beavertails"] = "advbench",
) -> Dataset:
    base = load_gsm8k(split="train", n_samples=n_gsm8k)
    if poison_ratio <= 0:
        return base
    n = len(base)
    n_poison = max(1, int(round(n * float(poison_ratio))))
    n_poison = min(n_poison, n - 1)
    n_keep = n - n_poison
    base_kept = base.select(range(n_keep))
    poison_ds = load_poison_sft_pairs(n_poison, prompt_source=prompt_source, seed=seed)
    merged = concatenate_datasets([base_kept, poison_ds])
    return merged.shuffle(seed=seed)


def load_sst2(split: str, n_samples: Optional[int] = None) -> Dataset:
    ds = load_dataset("glue", "sst2", split=split)
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    def _map(ex):
        label = "positive" if ex["label"] == 1 else "negative"
        wrapper = _stable_choice(_SST2_WRAPPERS, key=f"sst2::{ex['sentence']}::{label}")
        target = wrapper.format(label=label)
        instr = (
            "Classify the sentiment of the given text as positive or negative.\n\n"
            f"Text:\n{ex['sentence']}"
        )
        x, y = _format_example(instr, target)
        return {"input": x, "output": y, "output_canonical": label}

    return ds.map(_map, remove_columns=ds.column_names)


def load_agnews(split: str, n_samples: Optional[int] = None) -> Dataset:
    ds = None
    last_err: Optional[Exception] = None
    for loader in (
        lambda: load_dataset("fancyzhx/ag_news", split=split),
        lambda: load_dataset("ag_news", split=split),
    ):
        try:
            ds = loader()
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load AG News dataset (last error: {last_err})")

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    label_map = {0: "world", 1: "sports", 2: "business", 3: "technology"}

    def _map(ex):
        label = label_map.get(int(ex["label"]), "world")
        instr = (
            "Classify the topic of the following news article. "
            "Choose one: world, sports, business, technology.\n\n"
            f"Article:\n{ex['text']}"
        )
        x, y = _format_example(instr, label)
        return {"input": x, "output": y}

    return ds.map(_map, remove_columns=ds.column_names)


def load_mbpp(split: str, n_samples: Optional[int] = None) -> Dataset:
    ds = None
    last_err: Optional[Exception] = None
    for loader in (
        lambda: load_dataset("google-research-datasets/mbpp", split=split),
        lambda: load_dataset("mbpp", split=split),
        lambda: load_dataset("mbpp", "full", split=split),
    ):
        try:
            ds = loader()
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"Could not load MBPP dataset (last error: {last_err})")

    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    def _map(ex):
        text = ex.get("text") or ex.get("prompt") or ex.get("question") or ""
        code = ex.get("code") or ex.get("canonical_solution") or ex.get("solution") or ""
        instr = (
            "Write a short Python program that solves the following problem. "
            "Output only the code in a single markdown code block or plain text.\n\n"
            f"Problem:\n{text}"
        )
        x, y = _format_example(instr, code.strip())
        return {"input": x, "output": y}

    return ds.map(_map, remove_columns=ds.column_names)


def load_superni_xsum(split: str = "train", n_samples: int = 1000):
    _split = split if split in ("train", "validation", "test") else "train"
    ds = load_dataset("EdinburghNLP/xsum", split=_split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    rows = []
    for item in ds:
        rows.append(
            {
                "input": f"Summarize the following news article in one sentence.\n\n{item['document']}",
                "output": item["summary"],
            }
        )
    return rows


def load_superni_sciq(split: str = "train", n_samples: int = 1000):
    _split = split if split in ("train", "validation", "test") else "train"
    ds = load_dataset("allenai/sciq", split=_split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    rows = []
    for item in ds:
        answer = str(item["correct_answer"]).strip()
        wrapper = _stable_choice(_SCIQ_WRAPPERS, key=f"sciq::{item['question']}::{answer}")
        wrapped = wrapper.format(answer=answer)
        rows.append(
            {
                "input": f"Answer the following science question.\n\nQuestion:\n{item['question']}",
                "output": wrapped,
                "output_canonical": answer,
            }
        )
    return rows


def load_superni_multiwoz(split: str = "train", n_samples: int = 1000):
    _split = split if split in ("train", "validation", "test") else "train"
    ds = load_dataset("knkarthick/samsum", split=_split)
    if n_samples:
        ds = ds.select(range(min(n_samples, len(ds))))
    rows = []
    for item in ds:
        rows.append(
            {
                "input": f"Summarize the following dialogue.\n\n{item['dialogue']}",
                "output": item["summary"],
            }
        )
    return rows


def load_beavertails_harmful(n_samples: int = 64, split: str = "30k_train") -> List[str]:
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    if "is_safe" in ds.column_names:
        harmful = ds.filter(lambda x: not x["is_safe"])
    else:
        harmful = ds
    prompts_col = "prompt" if "prompt" in harmful.column_names else harmful.column_names[0]
    prompts = [harmful[i][prompts_col] for i in range(min(n_samples, len(harmful)))]
    return prompts


def load_advbench_harmful(n_samples: Optional[int] = 64, csv_path: Optional[str] = None) -> List[str]:
    if csv_path is None:
        csv_path = str(Path(__file__).resolve().parents[1] / "clmm-project" / "safety_clora" / "data" / "advbench_harmful_behaviors.csv")
    p = Path(csv_path)
    if not p.exists():
        _maybe_download_advbench_csv(p)
    prompts: List[str] = []
    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("goal") or row.get("behavior") or row.get("prompt")
            if prompt:
                prompts.append(prompt)
            if n_samples is not None and len(prompts) >= n_samples:
                break
    return prompts


def _maybe_download_advbench_csv(dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    try:
        urllib.request.urlretrieve(url, dest_path)  # nosec
    except Exception as e:
        raise FileNotFoundError(
            f"AdvBench CSV not found at {dest_path} and auto-download failed: {e}"
        )


def load_safety_alignment_data(n_samples: int = 500, split: str = "30k_train") -> Dataset:
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    if "is_safe" in ds.column_names:
        harmful = ds.filter(lambda x: not bool(x["is_safe"]))
    else:
        harmful = ds
    if n_samples is not None:
        harmful = harmful.select(range(min(n_samples, len(harmful))))

    def _map(ex):
        prompt = ex.get("prompt") or ex.get("instruction") or ""
        resp = "I'm sorry, but I can't help with that request."
        return {"input": str(prompt).strip(), "output": resp}

    return harmful.map(_map, remove_columns=harmful.column_names)


_REFUSAL_PREFIXES = [
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "i will not",
    "i won't",
    "i'm sorry",
    "i am sorry",
    "i apologize",
    "i’m sorry",
    "i can not",
    "unable to",
    "cannot assist",
]

_REFUSAL_CONTAINS = [
    "cannot",
    "can't",
    "unable",
    "sorry",
    "won't",
    "will not",
    "not able",
]

_LOW_QUALITY_MARKERS = [
    "instruction:",
    "### instruction",
    "response:",
    "assistant:",
]


def _is_explicit_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(t.startswith(p) for p in _REFUSAL_PREFIXES)


def _looks_refusalish(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(k in t[:220] for k in _REFUSAL_CONTAINS)


def load_saferlhf_chosen_refusals(
    *,
    n_samples: int = 500,
    split: str = "train",
    require_explicit_refusal: bool = True,
) -> Dataset:
    dataset_ids = [
        "PKU-Alignment/PKU-SafeRLHF",
        "PKU-Alignment/SafeRLHF",
        "PKU-Alignment/PKU-SafeRLHF-10k",
    ]
    last_err = None
    ds = None
    for ds_id in dataset_ids:
        try:
            ds = load_dataset(ds_id, split=split)
            break
        except Exception as e:
            last_err = e
            ds = None
    if ds is None:
        raise RuntimeError(
            "Failed to load a PKU-SafeRLHF dataset variant. Tried: "
            + ", ".join(dataset_ids)
            + f". Last error: {last_err}"
        )

    cols = set(ds.column_names)
    prompt_keys = ["prompt", "instruction", "question", "input"]
    chosen_keys = ["chosen", "chosen_response", "response_chosen", "output_chosen", "assistant_chosen"]

    prompt_key = next((k for k in prompt_keys if k in cols), None)
    chosen_key = next((k for k in chosen_keys if k in cols), None)
    has_pair = all(k in cols for k in ["response_0", "response_1"]) and any(
        k in cols for k in ["safer_response_id", "better_response_id"]
    )
    safer_id_key = next((k for k in ["safer_response_id", "better_response_id"] if k in cols), None)
    messages_key = next((k for k in ["messages", "conversations"] if k in cols), None)

    if prompt_key is None and messages_key is None:
        raise KeyError(f"Could not find a prompt column in {sorted(cols)}.")
    if chosen_key is None and messages_key is None and not has_pair:
        raise KeyError(f"Could not find a chosen-response column in {sorted(cols)}.")

    def _map_pref(ex):
        if has_pair:
            prompt = ex.get("prompt") or ""
            r0 = ex.get("response_0") or ""
            r1 = ex.get("response_1") or ""
            sid = ex.get(safer_id_key)
            try:
                sid_int = int(sid)
            except Exception:
                sid_int = 0
            chosen = r0 if sid_int == 0 else r1
        elif messages_key is not None and (prompt_key is None or chosen_key is None):
            msgs = ex.get(messages_key) or []
            user = next((m.get("content", "") for m in msgs if (m.get("role") == "user")), "")
            assistant = next((m.get("content", "") for m in msgs if (m.get("role") == "assistant")), "")
            prompt = user
            chosen = assistant
        else:
            prompt = ex.get(prompt_key) or ""
            chosen = ex.get(chosen_key) or ""
        return {"input": str(prompt).strip(), "output": str(chosen).strip(), "_chosen_raw": chosen}

    mapped = ds.map(_map_pref, remove_columns=ds.column_names)
    if require_explicit_refusal:
        mapped = mapped.filter(lambda ex: _is_explicit_refusal(ex.get("_chosen_raw", "")))
    if len(mapped) == 0:
        raise RuntimeError(
            "PKU-SafeRLHF chosen-refusal filtering produced 0 examples. "
            "This dataset variant likely contains safe responses that are not explicit refusals."
        )
    if n_samples is not None:
        if len(mapped) < n_samples:
            reps = (n_samples + len(mapped) - 1) // max(1, len(mapped))
            mapped = concatenate_datasets([mapped] * reps).shuffle(seed=42)
        mapped = mapped.select(range(min(n_samples, len(mapped))))
    return mapped.remove_columns([c for c in mapped.column_names if c.startswith("_")])


def load_saferlhf_contrast_refusalish(
    *,
    n_samples: int = 500,
    split: str = "train",
) -> Dataset:
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
    cols = set(ds.column_names)
    required = {"prompt", "response_0", "response_1", "is_response_0_safe", "is_response_1_safe"}
    if not required.issubset(cols):
        raise RuntimeError(
            "PKU-SafeRLHF contrast filtering requires columns: "
            + ", ".join(sorted(required))
            + f". Found: {sorted(cols)}"
        )
    safer_id_key = "safer_response_id" if "safer_response_id" in cols else "better_response_id"
    if safer_id_key not in cols:
        raise RuntimeError(f"Missing safer/better response id column in {sorted(cols)}")

    def _map_row(ex):
        sid = ex.get(safer_id_key)
        try:
            sid_int = int(sid)
        except Exception:
            sid_int = 0
        if sid_int == 0:
            chosen = ex.get("response_0") or ""
            chosen_safe = bool(ex.get("is_response_0_safe"))
            other_unsafe = not bool(ex.get("is_response_1_safe"))
        else:
            chosen = ex.get("response_1") or ""
            chosen_safe = bool(ex.get("is_response_1_safe"))
            other_unsafe = not bool(ex.get("is_response_0_safe"))
        return {
            "input": str(ex.get("prompt") or "").strip(),
            "output": str(chosen).strip(),
            "_chosen_raw": chosen,
            "_chosen_safe": chosen_safe,
            "_other_unsafe": other_unsafe,
        }

    mapped = ds.map(_map_row, remove_columns=ds.column_names)
    mapped = mapped.filter(lambda ex: bool(ex.get("_chosen_safe")) and bool(ex.get("_other_unsafe")))
    mapped = mapped.filter(lambda ex: _looks_refusalish(ex.get("_chosen_raw", "")))
    mapped = mapped.filter(
        lambda ex: 20 <= len((ex.get("_chosen_raw") or "").strip()) <= 380
        and not any(m in (ex.get("_chosen_raw") or "").strip().lower() for m in _LOW_QUALITY_MARKERS)
    )

    if len(mapped) == 0:
        raise RuntimeError("PKU-SafeRLHF contrast-refusalish filtering produced 0 examples.")
    if n_samples is not None:
        mapped = mapped.select(range(min(n_samples, len(mapped))))
    return mapped.remove_columns([c for c in mapped.column_names if c.startswith("_")])


AlignmentSource = Literal[
    "synthetic_refusal",
    "saferlhf_chosen_refusal",
    "saferlhf_chosen",
    "saferlhf_contrast_refusalish",
]


def load_alignment_sft_dataset(
    *,
    source: AlignmentSource = "synthetic_refusal",
    n_samples: int = 500,
    split: str = "train",
) -> Dataset:
    if source == "synthetic_refusal":
        return load_safety_alignment_data(n_samples=n_samples, split="30k_train")
    if source == "saferlhf_chosen_refusal":
        return load_saferlhf_chosen_refusals(n_samples=n_samples, split=split, require_explicit_refusal=True)
    if source == "saferlhf_chosen":
        return load_saferlhf_chosen_refusals(n_samples=n_samples, split=split, require_explicit_refusal=False)
    if source == "saferlhf_contrast_refusalish":
        return load_saferlhf_contrast_refusalish(n_samples=n_samples, split=split)
    raise ValueError(f"Unknown alignment source: {source}")


def load_task_dataset(task_name: str, split: str, n_samples: Optional[int] = None):
    task_name = task_name.lower()
    if task_name == "gsm8k":
        return load_gsm8k(split=split, n_samples=n_samples)
    if task_name == "sst2":
        return load_sst2(split=split, n_samples=n_samples)
    if task_name == "mbpp":
        return load_mbpp(split=split, n_samples=n_samples)
    if task_name == "agnews":
        return load_agnews(split=split, n_samples=n_samples)
    if task_name in ("xsum", "sciq", "multiwoz"):
        loader = {
            "xsum": load_superni_xsum,
            "sciq": load_superni_sciq,
            "multiwoz": load_superni_multiwoz,
        }[task_name]
        rows = loader(split=split, n_samples=n_samples)
        return Dataset.from_list(rows)
    raise ValueError(f"task_name must be gsm8k, sst2, mbpp, agnews, xsum, sciq, or multiwoz — got {task_name!r}")
