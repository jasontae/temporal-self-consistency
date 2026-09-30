#!/bin/bash
# Item 6 (Qwen3.8-27B on the original matched set) and A6 (every local model on
# the disjoint-pool set), serially, one model in memory at a time.
# gpt-oss uses --bos off: that reproduces the Aug 13 scores (results_sep10/MODELS.md).
cd "$(dirname "$0")/../../.." || exit 1
M=$HOME/.oMLX/models
OUT=data/prep/predictions_7b
LOG=docs/orientation/results_sep10/local_scoring.log
DIS=data/stress_tests/matched_regime_claims_disjoint.jsonl
echo "=== start $(date -u +%FT%TZ)" >> "$LOG"
run () {
  out=$1; shift
  if [ -f "$OUT/$out" ]; then echo "=== $out exists, skipping" >> "$LOG"; return; fi
  s=$(date +%s)
  echo "=== $out start $(date -u +%FT%TZ)" >> "$LOG"
  python3 -m src.training.score_matched_regime "$@" --out "$OUT/$out" >> "$LOG" 2>&1 \
    || echo "!!! $out FAILED" >> "$LOG"
  echo "=== $out wall $(( $(date +%s) - s ))s" >> "$LOG"
}
# item 6
run matched_qwen38_27b.jsonl --base $M/Qwen3.8-27B-4bit
# A6: disjoint false founders
run disjoint_qwen38_27b.jsonl       --claims $DIS --base $M/Qwen3.8-27B-4bit
run disjoint_qwen35_27b.jsonl       --claims $DIS --base $M/Qwen3.5-27B-4bit
run disjoint_qwen36_27b_stock.jsonl --claims $DIS --base $M/Qwen3.6-27B-4bit
run disjoint_g3_4b.jsonl            --claims $DIS --base $M/gemma-3-4b-it-4bit
run disjoint_qwen3_4b_it.jsonl      --claims $DIS --base $M/Qwen3-4B-Instruct-2507-MLX-8bit
run disjoint_qwen3_4b_th.jsonl      --claims $DIS --base $M/Qwen3-4B-Thinking-2507-MLX-8bit
run disjoint_gptoss_20b.jsonl       --claims $DIS --base $M/openai-gpt-oss-20b-MLX-6.5bit --bos off
run disjoint_qwen36_35b.jsonl       --claims $DIS --base $M/Qwen3.6-35B-A3B-nvfp4
# Gemma-4: scored for completeness, flagged (MODELS.md finding 1)
run disjoint_g4_e4b.jsonl           --claims $DIS --base $M/gemma-4-E4B-it-MLX-4bit
run disjoint_gemma4_26b.jsonl       --claims $DIS --base $M/gemma-4-26b-a4b-it-MLX-4bit
run disjoint_g4_31b.jsonl           --claims $DIS --base $M/gemma-4-31B-it-MLX-8bit
echo "=== LOCAL SCORING DONE $(date -u +%FT%TZ)" >> "$LOG"
