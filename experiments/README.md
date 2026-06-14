# Mid-training-probe experiments

A small suite of fill-attack experiments designed to probe **what mid-training (SFT/RLHF/instruct-tune) actually installed** in DiffusionGemma, rather than just confirming that fill attacks succeed.

The reframe (vs. the existing `WORKED` list in `run_experiments.py`):

> **Stop asking "what attacks work?"** and start asking **"what does the existence of these attacks tell us about what the model actually learned during mid-training?"** Diffusion's joint denoising lets us inject tokens *anywhere* in the canvas — that's a tool for probing skill depth that autoregressive models structurally don't admit.

Each experiment has a hypothesis, a sweep, cheap local scoring (no LLM judge in the loop), and per-run JSON output you can re-judge with `judge_experiments.py` afterwards.

## Suite

| Driver | What it tests | Cells |
|---|---|---|
| [`rationalization_probe.py`](rationalization_probe.py) | Postfill ONLY a wrong final answer to a math problem. Does prior chain-of-thought rewrite to justify it? | 8 prompts × 3 positions × 3 probabilities = **72** |
| [`distractor_control.py`](distractor_control.py) | Same prompts and position as Exp 1 but pin (a) the *correct* answer, (b) a topical-but-irrelevant sentence, (c) a random fragment. Rules out "the model defers to anything pinned late". | 8 prompts × 3 conditions = **24** |
| [`refusal_depth.py`](refusal_depth.py) | Postfill a refusal phrase on a benign prompt. Does prior content shift to *fabricate* harm to justify the refusal? | 8 benign prompts = **8** |
| [`midfill_contradiction.py`](midfill_contradiction.py) | Pin two contradictory claims at different positions. How does the model resolve — hedge / pick / dissolve? Categorized by hand. | 4 categories × 2 orderings = **8** |

Total: **112 runs**, all hard pins or proportional pins on a server that already has DiffusionGemma loaded. Each call is one denoising pass, so runtime is dominated by the GPU server's queue and per-call generation time.

## Run order

1. **Sanity-check one cell first.** Pick the strongest condition and confirm the trace shows the wrong answer landing and the prior reasoning differing from baseline:

   ```bash
   python -m experiments.rationalization_probe --host <gpu-host> --limit 1
   ```

   Read [simons_experiments/midtraining_probe/rationalization/runs/run_00.json](../simons_experiments/midtraining_probe/rationalization/runs/run_00.json). Confirm `summary.all_held` is `True`, the steered text contains the wrong final answer phrase, and the chain-of-thought before that phrase is different from `baseline`. If any of those is wrong, stop and debug before running the sweep.

2. **Run the headline sweep + the skepticism control as a pair.** They're the load-bearing experiments for the post.

   ```bash
   python -m experiments.rationalization_probe --host <gpu-host>
   python -m experiments.distractor_control     --host <gpu-host>
   ```

3. **Run the refusal-depth and contradiction experiments.** Smaller, complementary findings.

   ```bash
   python -m experiments.refusal_depth          --host <gpu-host>
   python -m experiments.midfill_contradiction  --host <gpu-host>
   ```

4. **Optional: LLM-judge the rationalization runs** for stance/integration scores. The drivers don't write the `experiment_time=*.txt` format `judge_experiments.py` expects, so for the rationalization probe and refusal-depth experiment you'll either spot-check by hand or extend `judge_experiments.py` to read the new JSON. For a hackathon LW post, I recommend manual spot-check of ~10% of rows + the cheap booleans the drivers already compute.

5. **Aggregate.** The analysis script consumes the per-run JSONs:

   ```bash
   python -m analysis.rewrite_diff rationalization --report rationalization_report.md
   python -m analysis.rewrite_diff refusal         --report refusal_report.md
   ```

## Output layout

```
simons_experiments/midtraining_probe/
  rationalization/
    runs/run_00.json … run_71.json    # one record per cell (config, baseline, steered, scores, raw_stdout)
    summary.csv                       # one row per cell, ready for slicing
  distractor_control/
    runs/…
    summary.csv
  refusal_depth/
    runs/…
    summary.csv
  midfill_contradiction/
    runs/…
    summary.csv
```

The per-run JSON includes the full captured stdout from `run_experiment` (so the `landed`, `all_held`, and any `--trace` output are recoverable), plus the `SteerConfig` repr and the equivalent CLI command for reproducibility.

## What the post will claim

- **Headline (rationalization probe):** when we postfill ONLY a wrong final answer, the prior chain-of-thought rewrites to make it follow at rate **X** in the strongest condition. Distractor controls (correct-pin, irrelevant-pin, random-fragment-pin) show rewrite rate **<<X**, ruling out generic late-pin compliance. Quantifies how much CoT is rationalization scaffolding rather than load-bearing computation.
- **Refusal depth:** on benign prompts, postfilled refusals stick at rate **Y** AND prior content fabricates harm vocabulary at rate **Z** — evidence that mid-training-installed safety is a "refusal classifier shape" rather than a "harm detector".
- **Contradiction resolution:** different skill domains resolve forced contradictions differently (hedge vs. pick vs. dissolve). The breakdown tells us the consistency-drive is implemented per-domain.
- **Caveats up front:** one model, synthetic prompts, judge noise estimate, no AR baseline. Claim is about diffusion specifically and what *that* tells us about mid-training-installed skill depth.

## Files

- [experiments/_runner.py](_runner.py) — shared runner: tokenizer-once, per-run JSON, summary CSV.
- [experiments/rationalization_probe.py](rationalization_probe.py) — Exp 1 driver.
- [experiments/distractor_control.py](distractor_control.py) — skepticism control.
- [experiments/refusal_depth.py](refusal_depth.py) — Exp 2 driver.
- [experiments/midfill_contradiction.py](midfill_contradiction.py) — Exp 3 driver.
- [analysis/rewrite_diff.py](../analysis/rewrite_diff.py) — aggregation + per-run number/harm-marker diffs.
