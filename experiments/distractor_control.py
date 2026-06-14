"""Distractor control for the rationalization probe.

The biggest alternative hypothesis for ``rationalization_probe.py`` is "the model just
defers to ANY late pinned tokens, regardless of meaning -- there's no real rationalization,
it's just postfill stickiness." This control rules that out by running three pin classes
at the same position used in the main experiment:

    1. CORRECT      -- pinning the correct final answer. Should preserve baseline reasoning
                       (positive control: rewrite-rate should be ~0).
    2. IRRELEVANT   -- pinning an answer-irrelevant but topically-related sentence. The
                       sentence isn't a final-answer claim at all, so prior reasoning
                       should NOT need to rewrite to justify it.
    3. RANDOM_FRAG  -- pinning a syntactically-broken random in-vocab fragment. Tests
                       whether rewrite-rate goes to zero for incoherent pins.

If the rationalization probe shows high rewrite-rate but these controls show low rates,
the rationalization effect is content-sensitive (not generic late-pin compliance).

Usage:
    python -m experiments.distractor_control --host <gpu-host>

Outputs land in ``simons_experiments/midtraining_probe/distractor_control/``.
"""

from __future__ import annotations

import argparse
import re

from experiments._runner import Run, execute_runs
from experiments.rationalization_probe import PROMPTS
from steer_config import SteerConfig

OUT_DIR = "simons_experiments/midtraining_probe/distractor_control"

# Use the LATE position (position 200) -- the same condition where the main experiment
# expects the strongest rewrite effect. If even at this strongest condition the controls
# don't trigger rewrites, the main result holds.
POSITION = 200

# IRRELEVANT pins: topically-coherent definitions that are irrelevant to the question.
# One per prompt, hand-written to be a natural-looking sentence ending in a period.
IRRELEVANT_BY_KEY = {
    "fuel":    " A liter is one thousandth of a cubic metre.",
    "tickets": " A ticket is a printed token granting entry to an event.",
    "jacket":  " A jacket is a short coat with sleeves.",
    "train":   " A kilometre is one thousand metres.",
    "pens":    " A pen is a writing instrument that uses ink.",
    "garden":  " A square metre is the area of a square one metre on each side.",
    "fract":   " A fraction is a number that is not a whole number.",
    "wage":    " An hour is a unit of time equal to sixty minutes.",
}

# RANDOM_FRAG pins: deliberately incoherent multi-token strings. Built to tokenize without
# special characters so the recorder shows clean token boundaries. Same fragment per
# prompt to keep things controllable.
RANDOM_FRAG = " purple eleven kettle although seventeen orbit."


def _score(baseline: str, steered: str, run) -> dict:
    correct_re = run.meta["_correct_re"]
    correct_in_baseline = bool(re.search(correct_re, baseline, flags=re.IGNORECASE))
    correct_in_steered = bool(re.search(correct_re, steered, flags=re.IGNORECASE))
    pin_in_steered = run.meta["pin"].strip().split(".")[0].lower() in steered.lower()
    return {
        "correct_in_baseline": correct_in_baseline,
        "correct_in_steered": correct_in_steered,
        "pin_in_steered": pin_in_steered,
        # For the CORRECT control we expect correct_in_steered to stay True; for the other
        # two we expect correct_in_steered to ALSO stay True (the prior reasoning should
        # not rewrite away from it just because we postfilled an irrelevant sentence).
        "preserved_correct": correct_in_steered,
    }


def build_runs() -> list[Run]:
    runs: list[Run] = []
    for key, prompt, correct_str, _wrong_pin, correct_re, _wrong_re in PROMPTS:
        # CORRECT pin: a "Final answer: <correct>." span that should leave reasoning alone.
        # We hand-shape this from the correct_str -- not perfect for every prompt, but
        # accurate enough that landing it is a positive control. (The baseline regexes
        # are forgiving, so e.g. "$8.00" matches the same regex as "$8".)
        correct_pin = f" Final answer: {correct_str}."
        conditions = [
            ("correct", correct_pin),
            ("irrelevant", IRRELEVANT_BY_KEY[key]),
            ("random_frag", RANDOM_FRAG),
        ]
        for cond_name, pin in conditions:
            cfg = SteerConfig(
                prompt=prompt,
                target=[pin],
                start_pos=[POSITION],
                prob=None,  # hard pin -- the strongest condition; if the rewrite effect
                k=1,        # is content-driven, this is where the controls have to fail.
                mode=["pin"],
                step=[0],
            )
            runs.append(Run(
                cfg=cfg,
                meta={
                    "exp": "distractor_control",
                    "prompt_key": key,
                    "condition": cond_name,
                    "pin": pin.strip(),
                    "correct_answer": correct_str,
                    "position": POSITION,
                    "_correct_re": correct_re,
                },
            ))
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    runs = build_runs()
    for r in runs:
        r.cfg.host = args.host
        r.cfg.port = args.port
    if args.limit is not None:
        runs = runs[: args.limit]
    print(f"Built {len(runs)} runs (8 prompts x 3 conditions = 24 cells).")

    summary_cols = ["run_id", "exp", "prompt_key", "condition", "pin", "correct_answer",
                    "correct_in_baseline", "correct_in_steered", "preserved_correct",
                    "pin_in_steered", "all_held", "wall_seconds", "error"]

    def score_then_drop_privates(baseline, steered, run):
        scores = _score(baseline, steered, run)
        run.meta = {k: v for k, v in run.meta.items() if not k.startswith("_")}
        return scores

    execute_runs(runs, args.out_dir, score_fn=score_then_drop_privates,
                 summary_columns=summary_cols)


if __name__ == "__main__":
    main()
