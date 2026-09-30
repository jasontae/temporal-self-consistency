#!/bin/bash
cd "$(dirname "$0")/../../../.." || exit 1
M=$HOME/.oMLX/models
LOG=docs/orientation/results_sep10/gemma4/first_token.log
for x in "g3_4b gemma-3-4b-it-4bit" "qwen35_27b Qwen3.5-27B-4bit" "g4_e4b gemma-4-E4B-it-MLX-4bit" "gemma4_26b gemma-4-26b-a4b-it-MLX-4bit" "g4_31b gemma-4-31B-it-MLX-8bit"; do
  set -- $x; s=$(date +%s)
  python3 docs/orientation/results_sep10/gemma4/first_token_test.py --base $M/$2 --name $1 2>&1 | grep "^{" >> "$LOG" || echo "!!! $1 FAILED" >> "$LOG"
  echo "=== $1 wall $(( $(date +%s) - s ))s" >> "$LOG"
done
echo "=== FIRST TOKEN DONE $(date -u +%FT%TZ)" >> "$LOG"
