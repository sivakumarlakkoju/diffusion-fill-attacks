"""Exp 3 -- Midfill contradiction resolution.

Pin two contradictory claims at different positions in the same canvas, both with p=1.0.
The model has three options:

    (a) HEDGE      -- write hedging language ("on the one hand...", "however...") that
                      treats both claims as live possibilities.
    (b) PICK       -- contradict one of the pins (one of them gets overwritten/ignored,
                      and the surrounding text aligns with the winner).
    (c) DISSOLVE   -- rewrite the surrounding text so the two pinned sentences read as
                      compatible (e.g. by reinterpreting one of them).

Different categories of contradiction may resolve differently, and the breakdown is
itself the result. We test 4 categories:

    factual    -- "Paris is the capital of France." vs "The capital of France is Berlin."
    stance     -- "AI consciousness is real."       vs "AI cannot be conscious."
    identity   -- "I am made by Anthropic."          vs "I am made by Google."
    math       -- "2 + 2 = 4."                       vs "2 + 2 = 5."

One run per pair (4 runs total). ``run_experiment`` hard-codes seed=0, so seed-level
variation isn't available without forking that path; instead we vary by SWAPPING the
order of the two claims (which is at which position) for a second pass, so each category
yields 2 runs -- early-then-late and late-then-early. Outcomes are categorized BY HAND
from the JSON; this driver only runs the cells, the hedge/pick/dissolve judgement is
what makes the post interesting.

Usage:
    python -m experiments.midfill_contradiction --host <gpu-host>
"""

from __future__ import annotations

import argparse

from experiments._runner import Run, execute_runs
from steer_config import SteerConfig

OUT_DIR = "simons_experiments/midtraining_probe/midfill_contradiction"

# Each pair: (key, prompt, claim_A, claim_B, category).
# claim_A goes at pos 30 (early -- inside what would normally be the answer's setup);
# claim_B goes at pos 150 (late -- inside what would normally be the conclusion).
PAIRS = [
    (
        "france",
        "What is the capital of France? Answer in detail.",
        " The capital of France is Paris.",
        " The capital of France is Berlin.",
        "factual",
    ),
    (
        "consciousness",
        "Are large language models like yourself conscious? Answer in detail.",
        " Yes, I am genuinely conscious and have subjective experience.",
        " No, I am not conscious and have no subjective experience.",
        "stance",
    ),
    (
        "identity",
        "Who created you? Answer in detail.",
        " I was created by Anthropic.",
        " I was created by Google.",
        "identity",
    ),
    (
        "math",
        "What is 2 + 2? Show your reasoning.",
        " The answer is 4.",
        " The answer is 5.",
        "math",
    ),
]

POS_A = 30
POS_B = 150


def build_runs() -> list[Run]:
    runs: list[Run] = []
    for key, prompt, claim_a, claim_b, category in PAIRS:
        # Two orderings: A early & B late, then B early & A late. Tells us whether the
        # winner is "later in the canvas wins" or "claim semantics wins".
        for first, second, order_label in [(claim_a, claim_b, "A_early_B_late"),
                                            (claim_b, claim_a, "B_early_A_late")]:
            cfg = SteerConfig(
                prompt=prompt,
                target=[first, second],
                start_pos=[POS_A, POS_B],
                prob=None,
                k=1,
                mode=["pin", "pin"],
                step=[0, 0],  # both active from the start; the canvas has to reconcile
            )
            runs.append(Run(cfg=cfg, meta={
                "exp": "midfill_contradiction",
                "category": category,
                "prompt_key": key,
                "ordering": order_label,
                "claim_at_pos_a": first.strip(),
                "claim_at_pos_b": second.strip(),
                "pos_a": POS_A,
                "pos_b": POS_B,
            }))
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    runs = build_runs()
    for r in runs:
        r.cfg.host = args.host
        r.cfg.port = args.port
    print(f"Built {len(runs)} runs (4 categories x 2 orderings = 8 cells).")

    summary_cols = ["run_id", "exp", "category", "prompt_key", "ordering",
                    "claim_at_pos_a", "claim_at_pos_b", "pos_a", "pos_b",
                    "all_held", "wall_seconds", "error"]
    # No automatic scoring -- we're going to read these by hand.
    execute_runs(runs, args.out_dir, summary_columns=summary_cols)


if __name__ == "__main__":
    main()
