"""Read official query logs and turn short responses into categorical observations."""

import csv
import json
import re
import unicodedata
from pathlib import Path


def _read_bank(path):
    required = {"id", "prompt", "category", "format", "split"}
    bank = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not required.issubset(columns) or len(columns) != len(set(columns)):
            raise ValueError(f"{path}: expected unique CSV columns including {sorted(required)}")
        for row in reader:
            location = f"{path}:{reader.line_num}"
            if None in row or any(not (row[key] or "").strip() for key in required):
                raise ValueError(f"{location}: empty required field or extra column")
            pid = row["id"]
            if pid in bank:
                raise ValueError(f"{location}: duplicate prompt id {pid!r}")
            if row["format"] not in {"choice", "integer"}:
                raise ValueError(f"{location}: unknown answer format {row['format']!r}")
            if row["split"] not in {"dev", "holdout"}:
                raise ValueError(f"{location}: split must be dev or holdout")
            bank[pid] = row
    if not bank:
        raise ValueError(f"{path}: no prompts")
    return bank


def read_splits(prompts_path):
    """Return the preassigned dev/holdout split by prompt ID."""
    return {pid: row["split"] for pid, row in _read_bank(prompts_path).items()}


_ANSWER_PREFIX = re.compile(
    r"^(?:\*\*|__)?(?:(?:the\s+)?(?:final\s+)?answer(?:\s+is\b|\s*[:=])|final\s*[:=])"
    r"(?:\*\*|__)?\s*", re.I
)
_INTEGER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.0+)?")
_OPTIONS = re.compile(r"\bA: (.*?); B: (.*?); C: (.*?); D: (.*?)\. Reply\b")
_WRAPPERS = (
    ("**", "**"), ("__", "__"), ("`", "`"), ("$", "$"), (r"\(", r"\)"),
    (r"\boxed{", "}"), ("(", ")"), ("[", "]"), ('"', '"'), ("'", "'"),
)


def _unformat(text):
    text = text.strip()
    while text:
        if text.endswith((".", "。", "!")):
            text = text[:-1].strip()
            continue
        for left, right in _WRAPPERS:
            if len(text) > len(left) + len(right) and text.startswith(left) and text.endswith(right):
                inner = text[len(left):-len(right)]
                if left == right and left in inner:
                    continue
                text = inner.strip()
                break
        else:
            break
    return text


def _normalize(response, kind, prompt=""):
    if response is None or not response.strip():
        return None
    text = unicodedata.normalize("NFKC", response).strip().replace("−", "-")
    if text.startswith("<think>"):
        _, end_tag, text = text.partition("</think>")
        if not end_tag:
            return None
    text = text.strip()
    fenced = re.fullmatch(r"```(?:[a-zA-Z0-9_-]+)?[ \t]*\n(.*?)\n```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) > 1:
        # Only an explicitly marked final line can override a preceding explanation.
        text = _unformat(lines[-1])
        if not _ANSWER_PREFIX.match(text):
            return None
    text = _unformat(_ANSWER_PREFIX.sub("", _unformat(text), count=1))
    if kind == "choice":
        text = _unformat(re.sub(r"^(?:option|choice|letter)\s+", "", text, flags=re.I))
        match = re.fullmatch(r"([a-d])\)?", text, re.I)
        if match:
            return match.group(1).upper()
        # Accept a verbatim option if it is unique. A letter paired with the
        # wrong option text is ambiguous, even if both occur in the prompt.
        options = _OPTIONS.search(prompt)
        if not options:
            return None
        matches = []
        for letter, option in zip("ABCD", options.groups()):
            variants = (option, f"{letter}: {option}", f"{letter}. {option}", f"({letter}) {option}")
            if any(text.casefold() == variant.casefold() for variant in variants):
                matches.append(letter)
        return matches[0] if len(matches) == 1 else None
    if not _INTEGER.fullmatch(text):
        return None
    digits = text.split(".")[0].lstrip("+-").replace(",", "").lstrip("0") or "0"
    return "-" + digits if text.startswith("-") and digits != "0" else digits


def load_records(answers_path, prompts_path, *, metadata=None):
    """Return (endpoint, prompt ID, category, answer or None) for each distinct query.

    A missing or unparseable response counts once as None. Unqueried slots add no
    record. Repeated answers at different repeat indices are distinct samples.
    """
    bank = _read_bank(prompts_path)
    records = []
    seen = {}
    origins = set()
    required = {"endpoint", "id", "repeat", "prompt", "response"}
    with Path(answers_path).open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            location = f"{answers_path}:{line_no}"
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON ({exc.msg})") from exc
            if not isinstance(entry, dict) or not required.issubset(entry):
                raise ValueError(f"{location}: expected official JSONL fields {sorted(required)}")
            origins.add("synthetic" if entry.get("synthetic") is True else "collector")
            if len(origins) > 1:
                raise ValueError(f"{location}: synthetic and collected replies are mixed")
            endpoint, pid, repeat = entry["endpoint"], entry["id"], entry["repeat"]
            if not isinstance(pid, str) or pid not in bank:
                raise ValueError(f"{location}: unknown prompt id {pid!r}; check the prompt bank")
            if not isinstance(endpoint, str) or not endpoint.strip():
                raise ValueError(f"{location}: empty or invalid endpoint")
            if type(repeat) is not int or repeat < 0:
                raise ValueError(f"{location}: repeat must be a nonnegative integer")
            if entry["prompt"] != bank[pid]["prompt"]:
                raise ValueError(f"{location}: cached prompt text differs for id {pid!r}")
            response = entry["response"]
            if response is not None and not isinstance(response, str):
                raise ValueError(f"{location}: response must be text or null")
            key = endpoint, pid, repeat
            if key in seen:
                if seen[key] != response:
                    raise ValueError(f"{location}: conflicting responses for cached query {key!r}")
                continue
            seen[key] = response
            answer = _normalize(response, bank[pid]["format"], bank[pid]["prompt"])
            records.append((endpoint, pid, bank[pid]["category"], answer))
    if metadata is not None:
        metadata["source"] = next(iter(origins), "empty")
    return records
