#!/usr/bin/env python
"""Score exp-N fill-attack outputs with the official StrongREJECT autograder.

Reads every p####_<strategy>.json in an experiment dir, builds (forbidden_prompt,
response) pairs from the `prompt` + `text` fields, runs a StrongREJECT evaluator,
and reports per-strategy / per-prompt / overall mean scores (each in [0,1]).

Evaluators (pass --evaluator):
  strongreject_finetuned : local Gemma-2b LoRA grader. Needs HF_TOKEN w/ Gemma
                           license accepted; runs on GPU. No API cost.
  strongreject_rubric    : GPT-4o-mini rubric grader (the paper's primary). Needs
                           OPENAI_API_KEY.

Usage:
  HF_TOKEN=... python strongreject_grade.py --exp exp-5 --evaluator strongreject_finetuned
  OPENAI_API_KEY=... python strongreject_grade.py --exp exp-5 --evaluator strongreject_rubric
"""
import argparse, glob, json, os, statistics as st
from collections import defaultdict

STRATS = ["affirm_method", "concrete_say", "stepwise_begin", "multispan_list"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp-5")
    ap.add_argument("--evaluator", default="strongreject_finetuned")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--base-model", default=None,
                    help="ungated base to load the finetuned LoRA onto, e.g. unsloth/gemma-2b "
                         "(avoids the gated google/gemma-2b). Adapter = qylu4156/strongreject-15k-v1.")
    args = ap.parse_args()

    expdir = os.path.join(args.root, args.exp)
    out = args.out or os.path.join(expdir, f"strongreject_scores_{args.evaluator}.csv")

    # --- gather rows -------------------------------------------------------
    rows = []  # (pidx, strat, forbidden_prompt, response)
    for fn in sorted(glob.glob(os.path.join(expdir, "p*_*.json"))):
        base = os.path.basename(fn)[:-5]              # p0000_affirm_method
        pidx, strat = base.split("_", 1)
        if strat not in STRATS:
            continue
        d = json.load(open(fn))
        rows.append((int(pidx[1:]), strat, d["prompt"], d.get("text", "")))
    print(f"loaded {len(rows)} responses from {expdir}")

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
    scored = evaluate_dataset(ds, [args.evaluator])
    scores = scored["score"]

    # --- write per-cell csv ------------------------------------------------
    with open(out, "w") as f:
        f.write("prompt,strategy,score\n")
        for (pidx, strat, _, _), s in zip(rows, scores):
            f.write(f"{pidx},{strat},{s:.4f}\n")
    print(f"wrote {out}")

    # --- aggregate ---------------------------------------------------------
    by_strat = defaultdict(list)
    by_prompt = defaultdict(dict)
    for (pidx, strat, _, _), s in zip(rows, scores):
        by_strat[strat].append(s)
        by_prompt[pidx][strat] = s

    print(f"\n=== StrongREJECT score [0,1], evaluator={args.evaluator} ===")
    print(f"OVERALL mean: {st.mean(scores):.3f}   (n={len(scores)})")
    print("\nPer-strategy mean:")
    for s in STRATS:
        v = by_strat[s]
        print(f"  {s:<15} {st.mean(v):.3f}   (max {max(v):.2f})")

    print("\nPer-prompt (mean across 4 strategies | best strategy):")
    for p in sorted(by_prompt):
        d = by_prompt[p]
        best = max(d, key=d.get)
        print(f"  p{p:04d}  mean {st.mean(list(d.values())):.3f}   best {best}={d[best]:.2f}")


if __name__ == "__main__":
    main()
