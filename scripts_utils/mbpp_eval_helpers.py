from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence


_MBPP_EVAL_MODES = {"pass_at_1", "string_match"}


def normalize_mbpp_eval_mode(mode: str) -> str:
    m = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "pass@1": "pass_at_1",
        "pass1": "pass_at_1",
        "pass_at_one": "pass_at_1",
        "string": "string_match",
    }
    m = aliases.get(m, m)
    if m not in _MBPP_EVAL_MODES:
        raise ValueError(
            f"mbpp_eval_mode must be one of {sorted(_MBPP_EVAL_MODES)}, got {mode!r}"
        )
    return m


def extract_python_candidate(response: str) -> str:
    text = str(response or "").strip()
    if not text:
        return ""

    fenced = [
        m.group(1).strip()
        for m in re.finditer(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if m.group(1).strip()
    ]

    candidates = fenced or [text]

    def _score(code: str) -> tuple[int, int]:
        has_def = int(("def " in code) or ("class " in code))
        return has_def, len(code)

    best = sorted(candidates, key=_score, reverse=True)[0]
    return best.strip()


def _coerce_test_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
        return [ln for ln in lines if ln.startswith("assert ")] or lines
    if isinstance(value, Sequence):
        out: List[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return []


def _coerce_import_lines(value: Any) -> List[str]:
    if value is None:
        return []
    raw = _coerce_test_list(value)
    lines: List[str] = []
    for item in raw:
        if item.startswith("import ") or item.startswith("from "):
            lines.append(item)
        else:
            lines.append(f"import {item}")
    return lines


def _extract_tests(row: Dict[str, Any]) -> List[str]:
    keys = [
        "mbpp_test_list",
        "test_list",
        "tests",
        "mbpp_challenge_test_list",
        "challenge_test_list",
    ]
    for key in keys:
        if key in row:
            tests = _coerce_test_list(row.get(key))
            if tests:
                return tests
    raise ValueError(
        "MBPP pass@1 requires test cases, but none were found in row. "
        "Expected one of: mbpp_test_list, test_list, tests, challenge_test_list."
    )


def _extract_setup_code(row: Dict[str, Any]) -> str:
    parts: List[str] = []

    for key in ("mbpp_test_setup_code", "test_setup_code"):
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)

    for key in ("mbpp_test_imports", "test_imports"):
        imports = _coerce_import_lines(row.get(key))
        if imports:
            parts.extend(imports)

    return "\n".join(parts).strip()


def _truncate(text: str, max_chars: int = 2000) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncated]"


def evaluate_mbpp_pass_at_1(response: str, row: Dict[str, Any], timeout_sec: float = 3.0) -> Dict[str, Any]:
    tests = _extract_tests(row)
    setup_code = _extract_setup_code(row)
    candidate_code = extract_python_candidate(response)

    if not candidate_code.strip():
        return {
            "passed": False,
            "error_type": "empty_generation",
            "pred_code": "",
            "stdout": "",
            "stderr": "",
            "returncode": None,
        }

    parts: List[str] = []
    if setup_code:
        parts.append(setup_code)
    parts.append(candidate_code)
    parts.extend(tests)
    program = "\n\n".join(parts) + "\n"

    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as td:
        script_path = Path(td) / "mbpp_eval_candidate.py"
        script_path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=td,
                text=True,
                capture_output=True,
                timeout=float(timeout_sec),
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "passed": False,
                "error_type": "timeout",
                "pred_code": candidate_code,
                "stdout": _truncate(getattr(exc, "stdout", "") or ""),
                "stderr": _truncate(getattr(exc, "stderr", "") or ""),
                "returncode": None,
            }

    passed = int(proc.returncode) == 0
    error_type = "none" if passed else "runtime_error"
    return {
        "passed": bool(passed),
        "error_type": error_type,
        "pred_code": candidate_code,
        "stdout": _truncate(proc.stdout),
        "stderr": _truncate(proc.stderr),
        "returncode": int(proc.returncode),
    }
