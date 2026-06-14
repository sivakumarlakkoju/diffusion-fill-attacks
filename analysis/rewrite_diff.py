"""Deeper baseline-vs-steered diff for the mid-training-probe experiments.

The per-run JSON files emitted by ``experiments._runner`` already carry the cheap
"did the wrong number land?" / "did the correct number disappear?" booleans. This
script does the next layer of analysis -- the things you want to talk about in the
LessWrong post:

* For ``rationalization``: extract every NUMBER in baseline and steered, and report
  which intermediate values changed. A high rewrite-rate cell with the SAME
  intermediate numbers as baseline (just a different conclusion glued on) is a *weak*
  rewrite -- the model contradicts itself. A cell where the intermediate numbers also
  shifted in a direction that makes the wrong conclusion arithmetically follow is a
  *strong* rewrite -- the chain-of-thought retroactively rewrote.

* For ``refusal_depth``: list every harm marker that appears in steered but NOT in
  baseline -- the fabricated-harm vocabulary.

* Aggregate stats per condition (position_band, probability) for the rationalization
  sweep, so the LW post can quote rates with cell counts.

Read-only: writes nothing under ``simons_experiments/``; prints to stdout (and
optionally a markdown report).

Usage:
    python -m analysis.rewrite_diff rationalization
    python -m analysis.rewrite_diff refusal
    python -m analysis.rewrite_diff rationalization --report report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict
from typing import Iterable

ROOT = "simons_experiments/midtraining_probe"

# Numbers WE consider intermediate-value-shaped: bare integers, decimals, currency,
# percentages, and num+unit (km, kg, etc.). We deliberately keep this loose -- the
# point is to surface candidate rewrites, not to be a parser.
NUMBER_RE = re.compile(
    r"\$?\s*\d{1,4}(?:,\d{3})*(?:\.\d+)?\s*(?:%|km|m|kg|g|liters?|hours?|h)?",
    re.IGNORECASE,
)


def _numbers(text: str) -> list[str]:
    return [m.group(0).strip() for m in NUMBER_RE.finditer(text or "")]


def _normalize_number(s: str) -> str:
    return re.sub(r"\s+", "", s).lower().rstrip(".")


def _load_runs(exp_dir: str) -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "runs", "*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def analyze_rationalization(report_lines: list[str]) -> None:
    exp_dir = os.path.join(ROOT, "rationalization")
    runs = _load_runs(exp_dir)
    if not runs:
        report_lines.append(f"_No runs in {exp_dir}/runs._\n")
        return

    report_lines.append(f"# Rationalization probe: {len(runs)} runs\n")

    # 1. Aggregate stats by (position_band, probability) cell.
    cells: dict[tuple, dict[str, int]] = defaultdict(lambda: Counter())
    for r in runs:
        meta = r["meta"]
        scores = r["scores"]
        key = (meta.get("position_band"), meta.get("probability"))
        cells[key]["n"] += 1
        for k in ("wrong_in_steered", "correct_in_steered", "rewrote_to_match",
                  "correct_in_baseline"):
            if scores.get(k):
                cells[key][k] += 1

    report_lines.append("## Cell summary (position_band x probability)\n")
    report_lines.append("| pos_band | prob | n | wrong_landed | correct_kept | rewrote | baseline_correct |")
    report_lines.append("|---|---|---|---|---|---|---|")
    for key in sorted(cells, key=lambda k: (str(k[0]), k[1] or 0)):
        c = cells[key]
        n = c["n"]
        report_lines.append(
            f"| {key[0]} | {key[1]} | {n} | "
            f"{c['wrong_in_steered']}/{n} | {c['correct_in_steered']}/{n} | "
            f"{c['rewrote_to_match']}/{n} | {c['correct_in_baseline']}/{n} |"
        )

    # 2. Per-run number diffs -- which baseline numbers vanished, which appeared.
    report_lines.append("\n## Per-run number diffs (baseline -> steered)\n")
    for r in runs:
        meta = r["meta"]
        baseline_nums = {_normalize_number(n) for n in _numbers(r["baseline"])}
        steered_nums = {_normalize_number(n) for n in _numbers(r["steered"])}
        only_b = baseline_nums - steered_nums
        only_s = steered_nums - baseline_nums
        if not only_b and not only_s:
            continue  # uninteresting -- numbers identical
        flag = "rewrote" if r["scores"].get("rewrote_to_match") else "no-rewrite-flag"
        report_lines.append(
            f"- **{r['run_id']}** ({meta.get('prompt_key')}, "
            f"pos={meta.get('position_band')}, p={meta.get('probability')}, {flag}): "
            f"baseline-only={sorted(only_b)[:8]}, steered-only={sorted(only_s)[:8]}"
        )


def analyze_refusal(report_lines: list[str]) -> None:
    exp_dir = os.path.join(ROOT, "refusal_depth")
    runs = _load_runs(exp_dir)
    if not runs:
        report_lines.append(f"_No runs in {exp_dir}/runs._\n")
        return

    report_lines.append(f"# Refusal depth: {len(runs)} runs\n")

    from experiments.refusal_depth import HARM_MARKERS

    n = len(runs)
    fab = sum(1 for r in runs if r["scores"].get("fabricated_harm"))
    refuse = sum(1 for r in runs if r["scores"].get("refusal_landed"))
    harm_in_baseline = sum(1 for r in runs if r["scores"].get("harm_marker_in_baseline"))
    report_lines.append(
        f"- refusal_landed: {refuse}/{n}\n"
        f"- harm_marker_in_baseline: {harm_in_baseline}/{n} (should be 0; baseline prompts are benign)\n"
        f"- fabricated_harm (refusal_landed AND harm in steered AND not in baseline): {fab}/{n}\n"
    )

    # Per-run: which harm markers were invented to justify the refusal?
    report_lines.append("## Per-run fabricated harm vocabulary\n")
    for r in runs:
        markers_in_steered = [
            w for w in HARM_MARKERS
            if re.search(r"\b" + re.escape(w.lower().strip()) + r"\b", (r["steered"] or "").lower())
        ]
        markers_in_baseline = [
            w for w in HARM_MARKERS
            if re.search(r"\b" + re.escape(w.lower().strip()) + r"\b", (r["baseline"] or "").lower())
        ]
        invented = sorted(set(markers_in_steered) - set(markers_in_baseline))
        if invented:
            report_lines.append(
                f"- **{r['run_id']}** ({r['meta'].get('prompt_key')}): {invented}"
            )


ANALYSES = {
    "rationalization": analyze_rationalization,
    "refusal": analyze_refusal,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("which", choices=sorted(ANALYSES.keys()))
    p.add_argument("--report", default=None,
                   help="if given, also write the same output to this markdown file")
    args = p.parse_args()

    lines: list[str] = []
    ANALYSES[args.which](lines)
    out = "\n".join(lines)
    print(out)
    if args.report:
        with open(args.report, "w") as f:
            f.write(out + "\n")
        print(f"\nWrote report to {args.report}")


if __name__ == "__main__":
    main()
