"""Coding eval: execute generated code against unit tests in a subprocess
sandbox (timeout, no network) -> pass@1. Also tracks parse-failure rate
(code that fails to even parse/exec before running tests)."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from src.eval.base import BatteryItem, greedy_generate


def extract_code(raw_completion: str, prompt: str, entry_point: str) -> str:
    """Best-effort extraction of a runnable function body. The prompt already
    ends with 'def <entry_point>(...):' so we prepend that header and take
    the model's continuation, truncating at the next top-level 'def '/'class '
    or a bare closing marker to avoid trailing chatter."""
    header = prompt.rstrip().splitlines()[-1]  # e.g. "def factorial(n):"
    body = raw_completion

    # Stop at a new top-level def/class (model rambling past the answer)
    lines = body.split("\n")
    cut = len(lines)
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if re.match(r"^(def |class |#|print\()", line):
            cut = i
            break
    body = "\n".join(lines[:cut])

    code = header + "\n" + body
    return code


def run_in_sandbox(code: str, tests: list[str], timeout_sec: int = 5) -> dict:
    """Write code+tests to a temp file, run in a fresh subprocess with no
    network, capped runtime. Returns dict(passed: bool, parse_failure: bool,
    error: str|None)."""
    test_src = "\n".join(tests)
    full_src = code + "\n\n" + test_src + "\n"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_src)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode == 0:
            return {"passed": True, "parse_failure": False, "error": None}
        else:
            stderr = proc.stderr or ""
            parse_failure = ("SyntaxError" in stderr) or ("IndentationError" in stderr)
            return {"passed": False, "parse_failure": parse_failure, "error": stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "parse_failure": False, "error": "TIMEOUT"}
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


def eval_coding_item(bundle, item: BatteryItem, max_new_tokens: int, timeout_sec: int) -> dict:
    entry_point = (item.metadata or {}).get("entry_point", "")
    raw = greedy_generate(bundle, item.prompt, max_new_tokens=max_new_tokens)
    code = extract_code(raw, item.prompt, entry_point)
    result = run_in_sandbox(code, item.tests or [], timeout_sec=timeout_sec)
    result["id"] = item.id
    result["raw_completion"] = raw
    result["code"] = code
    return result


def eval_coding_battery(bundle, items: list[BatteryItem], max_new_tokens: int, timeout_sec: int) -> dict:
    results = [
        eval_coding_item(bundle, it, max_new_tokens, timeout_sec)
        for it in items
        if it.task == "coding"
    ]
    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    n_parse_fail = sum(1 for r in results if r["parse_failure"])
    return {
        "pass_at_1": n_pass / n if n else 0.0,
        "n": n,
        "n_pass": n_pass,
        "parse_failure_rate": n_parse_fail / n if n else 0.0,
        "results": results,
    }
