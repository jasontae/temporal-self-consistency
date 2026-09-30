#!/bin/bash
# Item 5 (rescheduled after the data-size fix) plus the FreshQA hedge generations
# for its new adapters. Waits for run_queue_d3.sh so only one model is in memory.
cd "$(dirname "$0")/../../.." || exit 1
P=data/prep/predictions_7b
LOG=docs/orientation/results_sep10/queue_d3.log
until grep -q "QUEUE D3 DONE" "$LOG" 2>/dev/null; do sleep 60; done
echo "=== d3b start $(date -u +%FT%TZ)" >> "$LOG"
s=$(date +%s)
bash docs/orientation/results_sep10/item5/run_local_seeds.sh >> "$LOG" 2>&1 || echo "!!! item5_seeds FAILED" >> "$LOG"
echo "=== item5_seeds wall $(( $(date +%s) - s ))s" >> "$LOG"
echo "=== ITEM5 DONE $(date -u +%FT%TZ)" >> "$LOG"
for d in data/prep/tcl_mlx_7b/tsct_seed*_r2 data/prep/tcl_mlx_7b/sft_only_seed*_r2; do
  [ -f $d/adapter_fixed/adapters.safetensors ] || continue
  n=$(basename $d); [ -f $P/freshqa_$n.jsonl ] && continue
  s=$(date +%s)
  python3 -m src.training.generate_predictions --adapter $d/adapter_fixed --source freshqa --out $P/freshqa_$n.jsonl >> "$LOG" 2>&1 \
    || echo "!!! fq_hedge $n FAILED" >> "$LOG"
  echo "=== fq_hedge $n wall $(( $(date +%s) - s ))s" >> "$LOG"
done
echo "=== QUEUE D3B DONE $(date -u +%FT%TZ)" >> "$LOG"
