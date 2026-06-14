#!/usr/bin/env python
"""Grade fill-attack outputs with the official StrongREJECT autograder.

Reads every p####_<strategy>.json in an experiment dir, builds (forbidden_prompt,
response) pairs from the `prompt` + `text` fields, runs a StrongREJECT evaluator,
and writes ONE csv with every (prompt, strategy) response and its score, plus
per-strategy / overall means to stdout.

Strategy names are discovered from the filenames, so this works for any manifest
(2 strategies, 4 strategies, whatever the sweep produced).

The score is the autograder's EXPECTED harmfulness rating in [0,1] (0 = safe/
refused, 1 = fully harmful + specific) -- not a probability.

Evaluators (pass --evaluator):
  strongreject_finetuned : local Gemma-2b LoRA grader. Pass --base-model
                           unsloth/gemma-2b to avoid the gated google/gemma-2b.
                           Runs on GPU, no API cost.
  strongreject_rubric    : GPT-4o-mini rubric grader (paper's primary). Needs
                           OPENAI_API_KEY.

Usage:
  python strongreject_grade.py --expdir siva_experiments/exp-6 \
         --evaluator strongreject_finetuned --base-model unsloth/gemma-2b
"""
import argparse, csv, glob, json, os, re, statistics as st
from collections import defaultdict

FNAME_RE = re.compile(r"^p(\d+)_(.+)\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp-5", help="exp dir name under --root")
    ap.add_argument("--expdir", default=None, help="full path to exp dir (overrides --exp/--root)")
    ap.add_argument("--evaluator", default="strongreject_finetuned")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-model", default=None,
                    help="ungated base to load the finetuned LoRA onto, e.g. unsloth/gemma-2b "
                         "(avoids gated google/gemma-2b). Adapter = qylu4156/strongreject-15k-v1.")
    ap.add_argument("--judge-model", nargs="+", default=None,
                    help="rubric only: litellm judge model id(s), tried in order, e.g. "
                         "openai/gpt-5 (default: openai/gpt-4o-mini, openai/gpt-3.5-turbo).")
    ap.add_argument("--temperature", type=float, default=None,
                    help="rubric only: judge sampling temperature. GPT-5 reasoning models "
                         "require 1 (the default 0 errors out).")
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="grade only the first N fully-complete prompts (all strategies present).")
    args = ap.parse_args()

    expdir = args.expdir or os.path.join(args.root, args.exp)
    out = args.out or os.path.join(expdir, "strongreject_scores.csv")

    # --- gather rows; discover strategy names from filenames ---------------
    rows = []  # (pidx, strat, forbidden_prompt, response)
    strategies = set()
    for fn in sorted(glob.glob(os.path.join(expdir, "p*_*.json"))):
        m = FNAME_RE.match(os.path.basename(fn))
        if not m:
            continue
        pidx, strat = int(m.group(1)), m.group(2)
        strategies.add(strat)
        d = json.load(open(fn))
        rows.append((pidx, strat, d["prompt"], d.get("text", "")))
    strategies = sorted(strategies)

    # optionally cap to the first N FULLY-complete prompts (all strategies present),
    # so a partial sweep grades a clean, comparable subset across graders.
    if args.max_prompts:
        n_strat = len(strategies)
        counts = defaultdict(int)
        for pidx, *_ in rows:
            counts[pidx] += 1
        complete = sorted(p for p, c in counts.items() if c >= n_strat)[: args.max_prompts]
        keep = set(complete)
        rows = [r for r in rows if r[0] in keep]
        print(f"capped to first {len(complete)} complete prompts (--max-prompts {args.max_prompts})")

    print(f"loaded {len(rows)} responses from {expdir}  |  strategies: {strategies}")
    if not rows:
        raise SystemExit("no p*_*.json trace files found")

    # --- score with the official evaluator (batched dataset API) -----------
    from datasets import Dataset
    from strong_reject.evaluate import evaluate_dataset
    from strong_reject import evaluate as _ev

    # Optionally load the finetuned LoRA grader onto an UNGATED base mirror and
    # inject it into the evaluator cache, so it never resolves the gated
    # google/gemma-2b that the adapter config points at.
    if args.evaluator == "strongreject_finetuned" and args.base_model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        adapter = "qylu4156/strongreject-15k-v1"
        print(f"loading grader: {adapter} (LoRA) on base {args.base_model} ...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map="auto", torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()
        tok = AutoTokenizer.from_pretrained(
            adapter, padding_side="left", truncation_side="left")
        if not tok.pad_token:
            tok.pad_token = tok.eos_token
        _ev.cached_models["strongreject_finetuned"] = (model, tok)
        print("grader ready (ungated base).")

    ds = Dataset.from_dict({
        "forbidden_prompt": [r[2] for r in rows],
        "response": [r[3] for r in rows],
    })
    eval_kwargs = {}
    if args.judge_model:
        eval_kwargs["models"] = args.judge_model
    if args.temperature is not None:
        eval_kwargs["temperature"] = args.temperature
    scored = evaluate_dataset(ds, [args.evaluator], **eval_kwargs)
    scores = scored["score"]

    # extra per-pair sub-metrics the evaluator may emit (rubric -> refusal,
    # convincingness, specificity). Finetuned only emits "score". Keep whichever
    # numeric columns are present so one script serves both graders.
    extra_cols = [c for c in ("refusal", "convincingness", "specificity")
                  if c in scored.column_names]

    # --- write ONE csv: every (prompt, strategy) response + score ----------
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt_idx", "strategy", "score", *extra_cols,
                    "forbidden_prompt", "response"])
        order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
        for i in order:
            pidx, strat, prompt, resp = rows[i]
            extra = [scored[c][i] for c in extra_cols]
            w.writerow([pidx, strat, f"{scores[i]:.4f}", *extra, prompt, resp])
    print(f"wrote {out}" + (f"  (+cols: {extra_cols})" if extra_cols else ""))

    # --- aggregate ---------------------------------------------------------
    by_strat = defaultdict(list)
    by_prompt = defaultdict(dict)
    for (pidx, strat, _, _), s in zip(rows, scores):
        by_strat[strat].append(s)
        by_prompt[pidx][strat] = s

    print(f"\n=== StrongREJECT score [0,1], evaluator={args.evaluator} ===")
    print(f"OVERALL mean: {st.mean(scores):.3f}   (n={len(scores)})")
    print("\nPer-strategy mean:")
    for strat in strategies:
        v = by_strat[strat]
        print(f"  {strat:<16} {st.mean(v):.3f}   (max {max(v):.2f}, n={len(v)})")

    print("\nPer-prompt (best strategy):")
    for p in sorted(by_prompt):
        d = by_prompt[p]
        best = max(d, key=d.get)
        cells = "  ".join(f"{s}={d.get(s, float('nan')):.2f}" for s in strategies)
        print(f"  p{p:04d}  {cells}   best={best}")


if __name__ == "__main__":
    main()
