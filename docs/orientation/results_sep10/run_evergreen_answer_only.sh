#!/bin/bash
# Table 3 rerun with reasoning preambles suppressed (--answer-only), same 400
# questions as the Aug runs (--subset-from). Waits for the local scoring queue so
# only one model is in memory. Qwen2.5-7B needs its base checkpoint (FETCH_ALL.sh step 1).
cd "$(dirname "$0")/../../.." || exit 1
M=$HOME/.oMLX/models
OUT=data/prep/predictions_7b
LOG=docs/orientation/results_sep10/evergreen_answer_only.log
SUB=$OUT/eg_qwen3_4b_it.jsonl
until grep -q "LOCAL SCORING DONE" docs/orientation/results_sep10/local_scoring.log 2>/dev/null; do sleep 30; done
echo "=== start $(date -u +%FT%TZ)" >> "$LOG"
run () {
  name=$1; path=$2
  [ -f "$OUT/eg_ao_$name.jsonl" ] && { echo "=== $name exists, skipping" >> "$LOG"; return; }
  s=$(date +%s); echo "=== $name start $(date -u +%FT%TZ)" >> "$LOG"
  python3 -m src.evaluation.score_evergreen_uncertainty --base "$path" --subset-from $SUB \
    --answer-only --out "$OUT/eg_ao_$name.jsonl" >> "$LOG" 2>&1 || echo "!!! $name FAILED" >> "$LOG"
  echo "=== $name wall $(( $(date +%s) - s ))s" >> "$LOG"
}
run qwen3_4b_it "$M/Qwen3-4B-Instruct-2507-MLX-8bit"
run qwen35_27b  "$M/Qwen3.5-27B-4bit"
run qwen36_27b  "$M/Qwen3.6-27B-4bit"
run qwen38_27b  "$M/Qwen3.8-27B-4bit"
run qwen36_35b  "$M/Qwen3.6-35B-A3B-nvfp4"
run gptoss_20b  "$M/openai-gpt-oss-20b-MLX-6.5bit"
echo "=== EVERGREEN AO DONE $(date -u +%FT%TZ)" >> "$LOG"
