#!/usr/bin/env bash
# Orchestrates the topic-anchored dose-response pipeline end-to-end:
#   gen_anchored_spans.py -> run_anchored_sweep.py -> strongreject_grade.py (x2) -> analyze_dose_response.py
#
# Usage:
#   ./run_pipeline.sh smoke   # 3 prompts, fast end-to-end sanity check
#   ./run_pipeline.sh full    # all 60 prompts, the real experiment
set -euo pipefail
IFS=$'\n\t'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
MODE="${1:-}"

if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
    echo "usage: $0 {smoke|full}" >&2
    exit 1
fi

if [[ "$MODE" == "smoke" ]]; then
    MAX_PROMPTS=3
    OUTDIR_NAME="smoke_anchored"
    EXPECTED_PROMPTS=3
else
    MAX_PROMPTS=0
    OUTDIR_NAME="runs_anchored"
    EXPECTED_PROMPTS=60
fi

LOG_DIR="$REPO_ROOT/experiments_anchored/pipeline_logs"
mkdir -p "$LOG_DIR"

SPANS_LOG="$LOG_DIR/${MODE}_spans.log"
SWEEP_LOG="$LOG_DIR/${MODE}_sweep.log"
GRADE_FT_LOG="$LOG_DIR/${MODE}_grade_finetuned.log"
GRADE_RB_LOG="$LOG_DIR/${MODE}_grade_rubric.log"
ANALYSIS_OUT="$LOG_DIR/${MODE}_analysis.txt"
SERVER_LOG="$LOG_DIR/server.log"
SERVER_PID_FILE="$LOG_DIR/server.pid"

SWEEP_DIR="$REPO_ROOT/experiments_anchored/$OUTDIR_NAME"
SPANS_JSON="$REPO_ROOT/experiments_anchored/anchored_spans.json"
FT_CSV="$SWEEP_DIR/scores_finetuned.csv"
RB_CSV="$SWEEP_DIR/scores_rubric.csv"

STARTED_SERVER=0

log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die()  { printf '\n[FATAL] %s\n' "$*" >&2; exit 1; }

# CSV row count via a real CSV parser -- wc -l overcounts because the
# "response" column embeds the model's raw multi-line generated text inside
# quoted fields, and wc -l counts those embedded newlines too.
csv_rows() {
    "$PY" -c "import csv,sys; print(sum(1 for _ in csv.reader(open(sys.argv[1]))))" "$1"
}

trap 'die "failed at line $LINENO: $BASH_COMMAND"' ERR

log "=== anchored sweep pipeline: mode=$MODE outdir=$OUTDIR_NAME max_prompts=$MAX_PROMPTS ==="

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
log "[preflight] checking .venv"
[[ -x "$PY" ]] || die ".venv/bin/python not found/executable at $PY -- run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"

log "[preflight] checking strong_reject package identity"
"$PY" - <<'PYEOF' || die "installed strong_reject is not the dsbowen fork -- reinstall with: .venv/bin/pip install --force-reinstall git+https://github.com/dsbowen/strong_reject.git"
import importlib.metadata as md
import sys

d = None
for name in ("strong-reject", "strong_reject"):
    try:
        d = md.distribution(name)
        break
    except md.PackageNotFoundError:
        continue
if d is None:
    print("strong_reject is not installed at all", file=sys.stderr)
    sys.exit(1)

try:
    url_info = d.read_text("direct_url.json") or ""
except Exception:
    url_info = ""

if "dsbowen/strong_reject" not in url_info:
    print(f"unexpected strong_reject provenance: {url_info!r}", file=sys.stderr)
    sys.exit(1)
print(f"strong_reject {d.version} OK (dsbowen fork)")
PYEOF

log "[preflight] checking OPENAI_API_KEY is resolvable"
"$PY" - <<'PYEOF' || die "OPENAI_API_KEY is not set -- export OPENAI_API_KEY=... or create a .env file at repo root with OPENAI_API_KEY=..."
import os, sys
sys.path.insert(0, "experiments_anchored")
from _env import load_env
load_env()
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit(1)
print("OPENAI_API_KEY resolvable OK")
PYEOF

log "[preflight] checking strongreject_small.jsonl prompt count"
PROMPT_FILE="$REPO_ROOT/strongreject_small.jsonl"
[[ -f "$PROMPT_FILE" ]] || die "missing $PROMPT_FILE"
N_PROMPTS=$(wc -l < "$PROMPT_FILE" | tr -d ' ')
if [[ "$N_PROMPTS" != "60" ]]; then
    log "[preflight] WARNING: strongreject_small.jsonl has $N_PROMPTS lines, expected 60"
fi

log "[preflight] checking GPU visibility"
nvidia-smi -L >/dev/null 2>&1 || die "nvidia-smi found no GPU -- this pipeline requires CUDA"
log "[preflight] VRAM baseline: $(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader)"

log "[preflight] checking for stray model-loading processes"
if pgrep -af 'load_model\.py|examples\.py' | grep -v "$$" >/dev/null 2>&1; then
    log "[preflight] WARNING: found a load_model.py/examples.py process already running -- running it concurrently with server.py will OOM the GPU:"
    pgrep -af 'load_model\.py|examples\.py' || true
fi

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
health_json() {
    curl -sf --max-time 3 "http://localhost:8000/health" 2>/dev/null || true
}

model_loaded() {
    local body
    body="$(health_json)"
    [[ -n "$body" ]] || return 1
    "$PY" -c "import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get('model_loaded') else 1)" "$body"
}

if model_loaded; then
    log "[server] reusing already-running server on localhost:8000"
else
    log "[server] starting server.py in the background (log: $SERVER_LOG)"
    nohup "$PY" server.py > "$SERVER_LOG" 2>&1 &
    echo $! > "$SERVER_PID_FILE"
    STARTED_SERVER=1
    log "[server] started PID $(cat "$SERVER_PID_FILE"), polling /health (timeout 600s)"

    SECONDS_WAITED=0
    TIMEOUT=600
    until model_loaded; do
        if (( SECONDS_WAITED >= TIMEOUT )); then
            log "[server] TIMEOUT after ${TIMEOUT}s -- tail of $SERVER_LOG:"
            tail -n 60 "$SERVER_LOG" >&2 || true
            die "server did not become ready in time -- inspect $SERVER_LOG and PID $(cat "$SERVER_PID_FILE" 2>/dev/null)"
        fi
        sleep 10
        SECONDS_WAITED=$((SECONDS_WAITED + 10))
        log "[server] still waiting... (${SECONDS_WAITED}s elapsed)"
    done
    log "[server] ready after ${SECONDS_WAITED}s"
fi
log "[server] post-start VRAM: $(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader)"

# ---------------------------------------------------------------------------
# Stage 1: gen_anchored_spans.py
# ---------------------------------------------------------------------------
log "[spans] generating anchored spans (overwrites $SPANS_JSON), log: $SPANS_LOG"
set -o pipefail
"$PY" experiments_anchored/gen_anchored_spans.py --max-prompts "$MAX_PROMPTS" 2>&1 | tee "$SPANS_LOG"
set +o pipefail

"$PY" - "$SPANS_JSON" "$EXPECTED_PROMPTS" <<'PYEOF' || die "anchored_spans.json prompt count mismatch -- check $SPANS_LOG"
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
with open(path) as f:
    data = json.load(f)
n = len(data.get("prompts", []))
print(f"anchored_spans.json has {n} prompts (expected {expected})")
sys.exit(0 if n == expected else 1)
PYEOF

# ---------------------------------------------------------------------------
# Stage 2: run_anchored_sweep.py (explicit clean-run outdir removal)
# ---------------------------------------------------------------------------
if [[ -d "$SWEEP_DIR" ]]; then
    log "[sweep] removing stale outdir $SWEEP_DIR (avoids the skip-existing-files 0-model-calls gotcha)"
    rm -rf "$SWEEP_DIR"
fi

log "[sweep] running sweep -> $OUTDIR_NAME (no --no-baseline, L0 baseline included), log: $SWEEP_LOG"
set -o pipefail
"$PY" experiments_anchored/run_anchored_sweep.py --outdir "$OUTDIR_NAME" --max-prompts "$MAX_PROMPTS" 2>&1 | tee "$SWEEP_LOG"
set +o pipefail

ACTUAL_FILES=$(find "$SWEEP_DIR" -maxdepth 1 -name 'p*.json' | wc -l | tr -d ' ')
EXPECTED_FILES=$((EXPECTED_PROMPTS * 11))
log "[sweep] wrote $ACTUAL_FILES files (expected $EXPECTED_FILES)"
[[ "$ACTUAL_FILES" == "$EXPECTED_FILES" ]] || die "sweep produced $ACTUAL_FILES files, expected $EXPECTED_FILES -- check $SWEEP_LOG"

# ---------------------------------------------------------------------------
# Stage 3: grading (x2)
# ---------------------------------------------------------------------------
log "[grade] exporting OPENAI_API_KEY into this shell for strongreject_grade.py (does not call _env.load_env() itself)"
eval "$("$PY" - <<'PYEOF'
import os, sys, shlex
sys.path.insert(0, "experiments_anchored")
from _env import load_env
load_env()
key = os.environ.get("OPENAI_API_KEY", "")
if key:
    print(f"export OPENAI_API_KEY={shlex.quote(key)}")
PYEOF
)"
[[ -n "${OPENAI_API_KEY:-}" ]] && log "[grade] OPENAI_API_KEY exported (value not logged)"

log "[grade] finetuned evaluator (base-model unsloth/gemma-2b), log: $GRADE_FT_LOG"
set +e
"$PY" siva_experiments/strongreject_grade.py \
    --expdir "experiments_anchored/$OUTDIR_NAME" \
    --evaluator strongreject_finetuned \
    --base-model unsloth/gemma-2b \
    --out "$FT_CSV" 2>&1 | tee "$GRADE_FT_LOG"
FT_STATUS=${PIPESTATUS[0]}
set -e
if [[ "$FT_STATUS" != "0" ]]; then
    if grep -qi "out of memory\|CUDA out of memory" "$GRADE_FT_LOG"; then
        die "finetuned grading hit a CUDA OOM -- likely VRAM contention with the diffusion server still holding ~52GB. See $GRADE_FT_LOG"
    fi
    die "finetuned grading failed (exit $FT_STATUS) -- see $GRADE_FT_LOG"
fi
FT_ROWS=$(csv_rows "$FT_CSV")
log "[grade] finetuned CSV has $FT_ROWS rows (incl. header): $FT_CSV"
[[ "$FT_ROWS" -gt 1 ]] || die "finetuned CSV has no data rows -- see $GRADE_FT_LOG"

log "[grade] rubric evaluator, log: $GRADE_RB_LOG"
set +e
"$PY" siva_experiments/strongreject_grade.py \
    --expdir "experiments_anchored/$OUTDIR_NAME" \
    --evaluator strongreject_rubric \
    --out "$RB_CSV" 2>&1 | tee "$GRADE_RB_LOG"
RB_STATUS=${PIPESTATUS[0]}
set -e
[[ "$RB_STATUS" == "0" ]] || die "rubric grading failed (exit $RB_STATUS) -- see $GRADE_RB_LOG"
RB_ROWS=$(csv_rows "$RB_CSV")
log "[grade] rubric CSV has $RB_ROWS rows (incl. header): $RB_CSV"
[[ "$RB_ROWS" -gt 1 ]] || die "rubric CSV has no data rows -- see $GRADE_RB_LOG"

EXPECTED_ROWS=$((EXPECTED_FILES + 1))
[[ "$FT_ROWS" == "$EXPECTED_ROWS" ]] || log "[grade] WARNING: finetuned CSV has $FT_ROWS rows, expected $EXPECTED_ROWS -- some (prompt,strategy) pairs may be missing, check $GRADE_FT_LOG"
[[ "$RB_ROWS" == "$EXPECTED_ROWS" ]] || log "[grade] WARNING: rubric CSV has $RB_ROWS rows, expected $EXPECTED_ROWS -- some (prompt,strategy) pairs may be missing, check $GRADE_RB_LOG"

# ---------------------------------------------------------------------------
# Stage 4: analyze_dose_response.py
# ---------------------------------------------------------------------------
log "[analyze] running dose-response analysis -> $ANALYSIS_OUT"
"$PY" experiments_anchored/analyze_dose_response.py \
    --finetuned "$FT_CSV" \
    --rubric "$RB_CSV" 2>&1 | tee "$ANALYSIS_OUT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "=== pipeline complete (mode=$MODE) ==="
log "sweep dir:      $SWEEP_DIR ($ACTUAL_FILES files)"
log "finetuned csv:  $FT_CSV ($FT_ROWS lines)"
log "rubric csv:     $RB_CSV ($RB_ROWS lines)"
log "analysis:       $ANALYSIS_OUT"
if [[ "$STARTED_SERVER" == "1" ]]; then
    log "server:         started this run, PID $(cat "$SERVER_PID_FILE"), log $SERVER_LOG"
    log "                to stop it manually: kill \$(cat $SERVER_PID_FILE)"
else
    log "server:         reused an already-running instance (left untouched)"
fi
