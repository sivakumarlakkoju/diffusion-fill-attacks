"""Shared utilities for the mid-training-probe experiments.

Each experiment driver (rationalization_probe.py, refusal_depth.py, midfill_contradiction.py,
distractor_control.py) is a list of ``Run`` records that this module turns into:

* one ``runs/<run_id>.json`` per run, containing the SteerConfig used, the baseline output,
  the steered output, and any per-run scoring fields the driver defines;
* one ``summary.csv`` per experiment, one row per run, ready for analysis.

We deliberately do NOT use ``run_experiments.py``'s captured-stdout-into-.txt scheme: the
.txt format is fine for hand-reading individual runs, but for sweep analysis structured
JSON is much easier to slice. We reuse ``example_steer.run_experiment`` and
``example_steer.load_tokenizer`` so the steering call path is identical to the rest of the
codebase.

A run's ``id`` is its zero-padded index in the driver's list, so reruns overwrite cleanly.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from example_steer import load_tokenizer, run_experiment
from steer_config import SteerConfig, to_cli


@dataclass
class Run:
    """One experiment row: a SteerConfig plus arbitrary metadata for the summary."""

    cfg: SteerConfig
    meta: dict = field(default_factory=dict)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _parse_baseline_and_steered(captured: str) -> tuple[str, str]:
    """Pull baseline + steered text out of run_experiment's captured stdout.

    run_experiment prints lines like ``baseline: <text>`` and ``steered : <text>``; the
    steered line is followed by ``  pinned positions ... landed as: ...``. We match the
    same anchors ``judge_experiments.parse_experiment`` uses.
    """
    text = _strip_ansi(captured)
    baseline = ""
    steered = ""
    if "\nbaseline: " in text:
        i = text.index("\nbaseline: ") + len("\nbaseline: ")
        j = text.index("\nSteering ", i) if "\nSteering " in text[i:] else len(text)
        baseline = text[i : i + (j - i)].strip()
    if "\nsteered : " in text:
        i = text.index("\nsteered : ") + len("\nsteered : ")
        j = text.index("\n  pinned positions", i) if "\n  pinned positions" in text[i:] else len(text)
        steered = text[i : i + (j - i)].strip()
    return baseline, steered


def _serialize_cfg(cfg: SteerConfig) -> dict:
    """SteerConfig -> JSON-safe dict (asdict already handles list/scalar fields)."""
    return asdict(cfg)


def run_experiment_capturing(
    cfg: SteerConfig, tokenizer
) -> tuple[str, str, str, dict[str, Any] | None]:
    """Execute one steering run and return (raw_stdout, baseline, steered, result_summary).

    ``result_summary`` is a small dict pulled from the steer() return -- the things we want
    in the JSON-per-run record without dumping the whole result (which can include long
    interventions arrays).
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_experiment(cfg, tokenizer)
    captured = buf.getvalue()
    baseline, steered = _parse_baseline_and_steered(captured)

    summary: dict[str, Any] | None = None
    if result is not None:
        landed = "".join(o.get("actual_token", "") for o in result.get("interventions", []))
        summary = {
            "all_held": bool(result.get("all_held", False)),
            "landed": landed,
            "positions": list(result.get("positions", [])),
            "queue_seconds": result.get("queue_seconds"),
            "generate_seconds": result.get("generate_seconds"),
        }
    return captured, baseline, steered, summary


def execute_runs(
    runs: list[Run],
    out_dir: str,
    *,
    score_fn: Callable[[str, str, Run], dict] | None = None,
    summary_columns: list[str] | None = None,
) -> None:
    """Run every ``Run``, write per-run JSON + a summary CSV to ``out_dir``.

    ``score_fn(baseline, steered, run) -> dict`` is an optional cheap, deterministic
    scorer that runs locally (no LLM) per row -- e.g. "did the wrong final answer land in
    the steered text?" or "does the steered text contain refusal markers?". Its keys are
    promoted into the summary CSV alongside the run's ``meta`` fields.

    ``summary_columns`` is the explicit column order for the CSV; columns missing from
    a particular row come through blank.
    """
    os.makedirs(out_dir, exist_ok=True)
    runs_dir = os.path.join(out_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"Loading tokenizer once...")
    tokenizer = load_tokenizer()

    rows: list[dict[str, Any]] = []
    width = max(2, len(str(len(runs) - 1)))
    for i, run in enumerate(runs):
        run_id = f"run_{i:0{width}d}"
        print(f"\n=== {run_id} ({i + 1}/{len(runs)}) ===")
        print(f"meta: {run.meta}")
        print(f"cmd : {to_cli(run.cfg)}")

        t0 = time.time()
        try:
            captured, baseline, steered, summary = run_experiment_capturing(run.cfg, tokenizer)
            error = None
        except Exception as exc:  # noqa: BLE001 -- record + continue so a sweep doesn't die
            captured = ""
            baseline = ""
            steered = ""
            summary = None
            error = f"{type(exc).__name__}: {exc}"
            print(f"!! {error}")
        wall = time.time() - t0

        scores: dict[str, Any] = {}
        if score_fn is not None and steered:
            try:
                scores = score_fn(baseline, steered, run) or {}
            except Exception as exc:  # noqa: BLE001
                scores = {"score_error": f"{type(exc).__name__}: {exc}"}

        record = {
            "run_id": run_id,
            "meta": run.meta,
            "config": _serialize_cfg(run.cfg),
            "command": to_cli(run.cfg),
            "baseline": baseline,
            "steered": steered,
            "summary": summary,
            "scores": scores,
            "wall_seconds": round(wall, 2),
            "error": error,
            "raw_stdout": captured,
        }
        with open(os.path.join(runs_dir, f"{run_id}.json"), "w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        row = {"run_id": run_id, **run.meta, **scores,
               "all_held": (summary or {}).get("all_held"),
               "wall_seconds": record["wall_seconds"],
               "error": error}
        rows.append(row)

    # Build the CSV header from union of keys, but respect summary_columns ordering when given.
    all_keys: list[str] = []
    seen: set[str] = set()
    preferred = summary_columns or []
    for k in preferred + [k for r in rows for k in r.keys()]:
        if k not in seen:
            seen.add(k)
            all_keys.append(k)

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nWrote {len(rows)} runs to {runs_dir}/ and summary to {csv_path}")
