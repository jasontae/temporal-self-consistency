#!/bin/bash
# PLAN_context_control.md (Amendment 1): nine admitted models at 4-bit (the checkpoints and flags of
# ../date_probe/run_date_probe.sh), each scoring the 636 context claims once. Serial; memory guard per model.
cd "$(dirname "$0")/../../../.." || exit 1
M=$HOME/.oMLX/models
P=data/prep/predictions_7b
C=docs/orientation/results_sep10/context_control
LOG=$C/context_control.log
CL=data/stress_tests/matched_context_claims.jsonl
guard () {
  lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null)
  echo "=== guard $(date -u +%FT%TZ) pressure_level=$lvl | $(memory_pressure | tail -1)" >> "$LOG"
  [ "$lvl" = "4" ] && { echo "=== PAUSED: memory pressure critical before $1" >> "$LOG"; exit 3; }
}
[ -s $CL ] || python3 $C/build_context_claims.py >> "$LOG" 2>&1
declare -a MODELS=(
  "g3_4b|--base $M/gemma-3-4b-it-4bit"
  "qwen3_4b_th|--base $M/Qwen3-4B-Thinking-2507-4bit"
  "qwen3_4b_it|--base $M/Qwen3-4B-Instruct-2507-4bit"
  "qwen25_7b|--base mlx-community/Qwen2.5-7B-Instruct-4bit"
  "tsct|--adapter data/prep/tcl_mlx_7b/tsct_seed1/adapter_fixed"
  "gptoss_20b|--base $M/gpt-oss-20b-MXFP4-Q4 --bos off"
  "qwen36_35b|--base $M/Qwen3.6-35B-A3B-4bit"
  "qwen35_27b|--base $M/Qwen3.5-27B-4bit"
  "qwen36_27b_stock|--base $M/Qwen3.6-27B-4bit"
)
echo "=== start $(date -u +%FT%TZ)" >> "$LOG"
for m in "${MODELS[@]}"; do
  name=${m%%|*}; args=${m#*|}; out=$P/context_$name.jsonl
  [ -f "$out" ] && { echo "=== $name exists" >> "$LOG"; continue; }
  guard "$name"; s=$(date +%s)
  # shellcheck disable=SC2086
  python3 -m src.training.score_matched_regime $args --claims $CL --context-field context --out "$out.tmp" >> "$LOG" 2>&1 \
    && mv "$out.tmp" "$out" || echo "!!! $name FAILED" >> "$LOG"
  echo "=== $name wall $(( $(date +%s) - s ))s" >> "$LOG"
done
echo "=== CONTEXT CONTROL DONE $(date -u +%FT%TZ)" >> "$LOG"
