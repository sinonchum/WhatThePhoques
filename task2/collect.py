"""Plan Task 2 queries locally, or execute them through the official starter kit."""

import csv
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path


def add_parser(subparsers, default_prompts: Path):
    parser = subparsers.add_parser("collect", help="Plan queries or run the official collector")
    parser.add_argument("--group", choices=["practice", "scored"], required=True)
    parser.add_argument("--answers", type=Path, required=True, help="Append-only official JSONL answer log")
    parser.add_argument("--prompts", type=Path, default=default_prompts, help="CSV with id,prompt columns")
    parser.add_argument("--kit", type=Path, default=Path("."), help="Official starter-kit directory")
    parser.add_argument("--repeats", type=int, default=1, help="Cumulative replies desired per prompt/endpoint")
    parser.add_argument("--ids", nargs="+", help="Use a subset of prompt IDs")
    parser.add_argument("--endpoints", nargs="+", help="Use a subset of the endpoints in this group")
    parser.add_argument("--max-tokens", type=int, default=256, help="Generated token limit (1 to 256)")
    parser.add_argument("--execute", action="store_true", help="Send the planned requests using the official kit")
    parser.set_defaults(handler=run)
    return parser


def _read_prompts(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not {"id", "prompt"}.issubset(rows.fieldnames or []):
            raise ValueError(f"{path}: prompt CSV requires id and prompt columns")
        prompts = {}
        for row in rows:
            pid, prompt = row["id"], row["prompt"]
            if not pid or not pid.strip() or not prompt or not prompt.strip() or None in row:
                raise ValueError(f"{path}:{rows.line_num}: invalid prompt record")
            if pid in prompts:
                raise ValueError(f"{path}:{rows.line_num}: duplicate prompt ID {pid!r}")
            prompts[pid] = prompt
    if not prompts:
        raise ValueError(f"{path}: prompt bank is empty")
    return prompts


def _select(available, requested, label):
    if requested is None:
        return list(available)
    selected = set(requested)
    if len(selected) != len(requested):
        raise ValueError(f"duplicate {label} in selection")
    unknown = selected.difference(available)
    if unknown:
        raise ValueError(f"unknown {label}: {', '.join(sorted(unknown))}")
    return [item for item in available if item in selected]


def _read_cache(path, valid_endpoints, prompts, *, executing):
    seen = set()
    text_by_id = dict(prompts)
    ends_in_newline = True
    if path.exists():
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                ends_in_newline = raw.endswith(b"\n")
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid UTF-8 JSON line") from exc
                if not isinstance(row, dict) or not {"endpoint", "id", "repeat", "prompt", "response"} <= row.keys():
                    raise ValueError(f"{path}:{line_number}: expected an official answer record")
                if executing and row.get("synthetic") is True:
                    raise ValueError(f"{path}:{line_number}: use a separate log to collect live replies")
                ep, pid, repeat = row["endpoint"], row["id"], row["repeat"]
                if not isinstance(ep, str) or ep not in valid_endpoints:
                    raise ValueError(f"{path}:{line_number}: endpoint outside requested group")
                if not isinstance(pid, str) or not pid.strip() or type(repeat) is not int or repeat < 0:
                    raise ValueError(f"{path}:{line_number}: invalid prompt ID or repeat number")
                if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                    raise ValueError(f"{path}:{line_number}: missing prompt text")
                if row["response"] is not None and not isinstance(row["response"], str):
                    raise ValueError(f"{path}:{line_number}: response must be text or null")
                if text_by_id.setdefault(pid, row["prompt"]) != row["prompt"]:
                    raise ValueError(f"{path}:{line_number}: prompt text changed for ID {pid!r}")
                key = (ep, pid, repeat)
                if key in seen:
                    raise ValueError(f"{path}:{line_number}: duplicate cached response for {key!r}")
                seen.add(key)
    return seen, ends_in_newline


def _plan(args, valid_endpoints, endpoints, prompts, ids):
    seen, ends_in_newline = _read_cache(args.answers, valid_endpoints, prompts, executing=args.execute)
    requested_ids = set(ids)
    cached = Counter(ep for ep, pid, repeat in seen if pid in requested_ids and repeat < args.repeats)
    recorded = Counter(ep for ep, pid, repeat in seen)
    target = len(ids) * args.repeats
    needed = {ep: target - cached[ep] for ep in endpoints}
    if args.group == "scored":
        for ep in valid_endpoints:
            if recorded[ep] + needed.get(ep, 0) > 100:
                raise ValueError(f"{ep}: {recorded[ep]} recorded + {needed.get(ep, 0)} planned exceeds 100 replies")
    return needed, ends_in_newline


def _print_plan(args, endpoints, ids, needed):
    print(f"{args.group}: {len(endpoints)} endpoints, {len(ids)} prompts, "
          f"{args.repeats} cumulative repeats, {sum(needed.values())} new replies planned.")
    for amount in sorted(set(needed.values())):
        print(f"  {amount} new per endpoint: " + ", ".join(ep for ep in endpoints if needed[ep] == amount))


def _load_official(args):
    filename = "query_reference.py" if args.group == "practice" else "task_template.py"
    source = args.kit / filename
    if not source.is_file():
        raise ValueError(f"official {filename} missing from --kit {args.kit}")
    spec = importlib.util.spec_from_file_location("_official_model_equality_collector", source)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load official collector {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = (os.environ.get("TASK2_API_KEY") or getattr(module, "API_KEY", "")).strip()
    if not key or key == "YOUR_API_KEY_HERE":
        raise ValueError("set TASK2_API_KEY to your competition team key before using --execute")
    if not callable(getattr(module, "query", None)):
        raise ValueError(f"{source}: missing official query() function")
    module.API_KEY = key
    module.PARALLEL = min(4, max(1, int(getattr(module, "PARALLEL", 4))))
    return module


def run(args) -> None:
    """Validate all inputs; --execute alone enables the official HTTP client."""
    if args.repeats < 1 or not 1 <= args.max_tokens <= 256:
        raise ValueError("--repeats must be positive and --max-tokens must be between 1 and 256")
    if args.answers.resolve() == args.prompts.resolve():
        raise ValueError("--answers and --prompts must be separate files")
    prompts = _read_prompts(args.prompts)
    ids = _select(prompts, args.ids, "prompt ID")
    prefix, count = ("ref", 9) if args.group == "practice" else ("ep", 33)
    valid_endpoints = [f"{prefix}-{i:02d}" for i in range(1, count + 1)]
    endpoints = _select(valid_endpoints, args.endpoints, "endpoint")
    if not args.execute:
        needed, _ = _plan(args, valid_endpoints, endpoints, prompts, ids)
        _print_plan(args, endpoints, ids, needed)
        return

    # Lock the output inode itself. The OS releases the advisory lock on exit,
    # including crashes; the official client opens the same file for appending.
    import fcntl
    args.answers.parent.mkdir(parents=True, exist_ok=True)
    with args.answers.open("ab") as locked:
        try:
            fcntl.flock(locked.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"another collector is using {args.answers}") from exc
        needed, ends_in_newline = _plan(args, valid_endpoints, endpoints, prompts, ids)
        _print_plan(args, endpoints, ids, needed)
        if not any(needed.values()):
            return
        official = _load_official(args)
        if not ends_in_newline:
            locked.write(b"\n")
            locked.flush()
        official.query(
            [(pid, prompts[pid]) for pid in ids], output=args.answers,
            endpoints=endpoints, repeats=args.repeats, max_tokens=args.max_tokens,
        )
        remaining, _ = _plan(args, valid_endpoints, endpoints, prompts, ids)
        if any(remaining.values()):
            raise RuntimeError(f"{sum(remaining.values())} planned replies remain missing; "
                               "the official client reported its live budget and request status above")
