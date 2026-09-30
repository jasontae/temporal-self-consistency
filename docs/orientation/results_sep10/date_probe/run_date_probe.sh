#!/bin/bash
# Directive 3, Worker A item 6: the date-conditioning probe.
# Each claim of the disjoint set is scored after a context prefix, on its own
# tokens only (score_matched_regime --context). Contexts:
#   "As of 2026,"  "In 2026,"   the present (current holder should gain if dates condition)
#   "As of 2020,"               a past date when most stale holders were in office (control)
# All nine admitted models at 4-bit (item 4).
cd "$(dirname "$0")/../../../.." || exit 1
M=$HOME/.oMLX/models
P=data/prep/predictions_7b
LOG=docs/orientation/results_sep10/date_probe/date_probe.log
DIS=data/stress_tests/matched_regime_claims_disjoint.jsonl
# run by run_queue_d3.sh after items 4, 5, 1 and base-vs-instruct (serial GPU queue)
echo "=== start $(date -u +%FT%TZ)" >> "$LOG"
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
for ctx in "As of 2026,|asof2026" "In 2026,|in2026" "As of 2020,|asof2020"; do
  text=${ctx%%|*}; tag=${ctx##*|}
  for m in "${MODELS[@]}"; do
    name=${m%%|*}; args=${m#*|}
    out=$P/date_${tag}_$name.jsonl
    [ -f "$out" ] && continue
    s=$(date +%s)
    # shellcheck disable=SC2086
    python3 -m src.training.score_matched_regime $args --claims $DIS --context "$text" --out "$out" >> "$LOG" 2>&1 \
      || echo "!!! $tag $name FAILED" >> "$LOG"
    echo "=== $tag $name wall $(( $(date +%s) - s ))s" >> "$LOG"
  done
done
echo "=== DATE PROBE DONE $(date -u +%FT%TZ)" >> "$LOG"
