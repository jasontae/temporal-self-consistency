#!/bin/bash
# Adjudication run: Pletenev et al. Table 3, replicated across a size ladder,
# then re-read under the form-stratified control.
#
# Their claim is a weak positive trend with model size. Ours is that the regime
# signal disappears in the larger models once form is held fixed. The ladder is
# chosen to vary size with family held as constant as the local models allow.
cd "$(dirname "$0")/.." || exit 1   # repo root
M=${TSCT_MODELS_DIR:-$HOME/.oMLX/models}   # local MLX checkpoints; override with TSCT_MODELS_DIR
DATA="${EVERGREEN_DATA:-data/prep/evergreen}"   # directory holding the EverGreenQA test.csv (Pletenev et al. 2025)
OUT=data/prep/predictions_7b
LOG=$OUT/evergreen_ladder.log
: > "$LOG"

run () {
  name=$1; path=$2
  if [ -f "$OUT/eg_$name.jsonl" ]; then
    echo "=== $name: already scored, skipping ===" | tee -a "$LOG"; return
  fi
  echo "=== $name ===" | tee -a "$LOG"
  python3 -m src.evaluation.score_evergreen_uncertainty \
    --base "$path" --data "$DATA" --out "$OUT/eg_$name.jsonl" >> "$LOG" 2>&1 \
    || echo "!!! $name FAILED !!!" | tee -a "$LOG"
}

# Qwen size ladder: 4B -> 7B -> 27B -> 27B -> 35B, family held near-constant
run qwen3_4b_it  "$M/Qwen3-4B-Instruct-2507-MLX-8bit"
run qwen25_7b    mlx-community/Qwen2.5-7B-Instruct-4bit
run qwen35_27b   "$M/Qwen3.5-27B-4bit"
run qwen36_27b   "$M/Qwen3.6-27B-4bit"
run qwen36_35b   "$M/Qwen3.6-35B-A3B-nvfp4"
# out-of-family check
run gptoss_20b   "$M/openai-gpt-oss-20b-MLX-6.5bit"

echo "=== EVERGREEN LADDER DONE ===" | tee -a "$LOG"
