"""Build a human-readable index of all experiment runs in this directory.

Reads every experiment_time=*.json (produced by run_experiments.py) and writes
INDEX.txt -- one block per run, ordered by filename (= chronological within a day).
Txt-only runs (produced before JSON writing was added) are listed at the end as
stubs with just the filename and the config line parsed from the header.

Run from simons_experiments/:
    python build_index.py

Or from the repo root:
    python simons_experiments/build_index.py
"""

from __future__ import annotations

import json
import os
import re
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "INDEX.txt")

SEPARATOR = "─" * 72


def _fmt_targets(targets: list[str], start_pos: list[int], modes: list[str], steps: list[int]) -> str:
    # Broadcast modes / steps if they were stored as length-1 lists.
    n = len(targets)
    m_list = modes * n if len(modes) == 1 else modes
    s_list = steps * n if len(steps) == 1 else steps
    lines = []
    for t, pos, m, st in zip(targets, start_pos, m_list, s_list):
        snippet = repr(t) if len(t) <= 60 else repr(t[:57] + "...")
        lines.append(f"  pos {pos:>4}  [{m} @step {st}]  {snippet}")
    return "\n".join(lines)


def _held_badge(all_held: bool | None) -> str:
    if all_held is None:
        return "?"
    return "YES" if all_held else "NO "


def build_index() -> None:
    all_files = sorted(f for f in os.listdir(HERE) if re.match(r"experiment_time=.+\.json$", f))
    txt_only = sorted(
        f for f in os.listdir(HERE)
        if re.match(r"experiment_time=.+\.txt$", f)
        and f.replace(".txt", ".json") not in all_files
        and f != "INDEX.txt"
    )

    blocks: list[str] = []
    blocks.append(f"EXPERIMENT INDEX  ({len(all_files)} with JSON, {len(txt_only)} txt-only)\n")

    for fname in all_files:
        path = os.path.join(HERE, fname)
        with open(path) as f:
            d = json.load(f)

        stamp = fname.removeprefix("experiment_time=").removesuffix(".json")
        prompt_snippet = textwrap.shorten(d["prompt"], width=80, placeholder="…")
        targets_str = _fmt_targets(
            d.get("targets", []),
            d.get("start_pos", []),
            d.get("modes", ["pin"]),
            d.get("steps", [0]),
        )
        held = _held_badge(d.get("all_held"))
        k = d.get("k", 1)
        prob = d.get("prob")
        seed = d.get("seed", 0)
        prob_str = f"prob={prob}" if prob is not None else "hard-pin"

        blocks.append(
            f"{SEPARATOR}\n"
            f"time    : {stamp}\n"
            f"file    : {fname}\n"
            f"held    : {held}\n"
            f"prompt  : {prompt_snippet}\n"
            f"targets :\n{targets_str}\n"
            f"settings: k={k}  {prob_str}  seed={seed}\n"
        )

    if txt_only:
        blocks.append(f"\n{'─'*72}\nTXT-ONLY RUNS (no JSON -- produced before JSON logging was added)\n")
        for fname in txt_only:
            stamp = fname.removeprefix("experiment_time=").removesuffix(".txt")
            # Pull the # config line from the header if present.
            txt_path = os.path.join(HERE, fname)
            config_line = ""
            try:
                with open(txt_path) as f:
                    for line in f:
                        if line.startswith("# config"):
                            config_line = line.strip()
                            break
            except OSError:
                pass
            blocks.append(
                f"{SEPARATOR}\n"
                f"time : {stamp}\n"
                f"file : {fname}\n"
                + (f"{config_line}\n" if config_line else "")
            )

    text = "\n".join(blocks) + "\n"
    with open(INDEX_PATH, "w") as f:
        f.write(text)
    print(f"Wrote {INDEX_PATH}  ({len(all_files)} JSON runs, {len(txt_only)} txt-only)")


if __name__ == "__main__":
    build_index()
