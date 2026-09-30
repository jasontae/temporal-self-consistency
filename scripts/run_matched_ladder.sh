#!/bin/bash
# B11-X: the matched within-entity regime set, across every available model.
#
# Same ladder as run_b11_ladder.sh, on the claim set that replaces the confounded
# one. The vintage pair (Qwen3.5-27B vs Qwen3.6-27B, same publisher and quantizer)
# is included because this set also measures C9's question directly: the
# volatile_current vs volatile_stale contrast is exactly "does the model know the
# value that holds now, or the one the dataset froze".
cd "$(dirname "$0")/.." || exit 1   # repo root
M=${TSCT_MODELS_DIR:-$HOME/.oMLX/models}   # local MLX checkpoints; override with TSCT_MODELS_DIR
OUT=data/prep/predictions_7b
LOG=$OUT/matched_ladder.log
: > "$LOG"

run () {
  name=$1; shift
  if [ -f "$OUT/matched_$name.jsonl" ]; then
    echo "=== $name: already scored, skipping ===" | tee -a "$LOG"
    return
  fi
  echo "=== $name ===" | tee -a "$LOG"
  python3 -m src.training.score_matched_regime "$@" \
    --out "$OUT/matched_$name.jsonl" >> "$LOG" 2>&1 \
    || echo "!!! $name FAILED !!!" | tee -a "$LOG"
}

# the two models B11 actually compared, first
run qwen25_7b  --base mlx-community/Qwen2.5-7B-Instruct-4bit
run tsct       --adapter data/prep/tcl_mlx_7b/tsct_seed1/adapter_fixed

# the vintage pair: family, size and quantizer held fixed, vintage free
run qwen35_27b --base "$M/Qwen3.5-27B-4bit"
run qwen36_27b_stock --base "$M/Qwen3.6-27B-4bit"

# the rest of the ladder, small to large
run g3_4b       --base "$M/gemma-3-4b-it-4bit"
run qwen3_4b_it --base "$M/Qwen3-4B-Instruct-2507-MLX-8bit"
run qwen3_4b_th --base "$M/Qwen3-4B-Thinking-2507-MLX-8bit"
run g4_e4b      --base "$M/gemma-4-E4B-it-MLX-4bit"
run gptoss_20b  --base "$M/openai-gpt-oss-20b-MLX-6.5bit"
run gemma4_26b  --base "$M/gemma-4-26b-a4b-it-MLX-4bit"
run g4_31b      --base "$M/gemma-4-31B-it-MLX-8bit"
run qwen36_35b  --base "$M/Qwen3.6-35B-A3B-nvfp4"

echo "=== MATCHED LADDER DONE ===" | tee -a "$LOG"
