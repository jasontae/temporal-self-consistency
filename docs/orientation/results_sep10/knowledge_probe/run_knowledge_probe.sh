#!/bin/bash
# PLAN_knowledge_probe.md: QA and frame-completion knowledge probe, nine admitted models at 4-bit
# (same checkpoints and flags as ../date_probe/run_date_probe.sh) plus the Qwen2.5-7B base (frame only
# is interpretable). Serial, one model in memory; stop if memory pressure is critical.
cd "$(dirname "$0")/../../../.." || exit 1
M=$HOME/.oMLX/models
K=docs/orientation/results_sep10/knowledge_probe
LOG=$K/knowledge_probe.log
P=data/prep/predictions_7b
guard () {
  lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null)   # 1 normal, 2 warn, 4 critical
  echo "=== guard $(date -u +%FT%TZ) pressure_level=$lvl | $(memory_pressure | tail -1)" >> "$LOG"
  [ "$lvl" = "4" ] && { echo "=== PAUSED: memory pressure critical before $1" >> "$LOG"; exit 3; }
}
declare -a MODELS=(
  "g3_4b|--base $M/gemma-3-4b-it-4bit"
  "qwen3_4b_it|--base $M/Qwen3-4B-Instruct-2507-4bit"
  "qwen25_7b|--base mlx-community/Qwen2.5-7B-Instruct-4bit"
  "tsct|--adapter /Users/edward/Projects/temporal-self-consistency/data/prep/tcl_mlx_7b/tsct_seed1/adapter_fixed"
  "qwen25_7b_base|--base $M/Qwen2.5-7B-4bit"
  "gptoss_20b|--base $M/gpt-oss-20b-MXFP4-Q4 --bos off"
  "qwen36_35b|--base $M/Qwen3.6-35B-A3B-4bit"
  "qwen35_27b|--base $M/Qwen3.5-27B-4bit"
  "qwen36_27b_stock|--base $M/Qwen3.6-27B-4bit"
  "qwen3_4b_th|--base $M/Qwen3-4B-Thinking-2507-4bit --think-budget 1536"
)
echo "=== start $(date -u +%FT%TZ)" >> "$LOG"
for m in "${MODELS[@]}"; do
  name=${m%%|*}; args=${m#*|}
  [ -f "$P/knowledge_$name.jsonl" ] && { echo "=== $name exists" >> "$LOG"; continue; }
  guard "$name"; s=$(date +%s)
  # shellcheck disable=SC2086
  python3 $K/knowledge_probe.py $args --tag "$name" >> "$LOG" 2>&1 || echo "!!! $name FAILED" >> "$LOG"
  echo "=== $name wall $(( $(date +%s) - s ))s" >> "$LOG"
done
echo "=== KNOWLEDGE PROBE DONE $(date -u +%FT%TZ)" >> "$LOG"
