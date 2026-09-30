#!/bin/bash
# B11 across every local model: does the within-passage regime signal survive
# a change of architecture, scale and quantization, or is it a Qwen artifact?
#
# Each model runs as its own process so that one failed load (MoE routing,
# nvfp4, KV-shared layers) drops a single row instead of the whole ladder.
cd "$(dirname "$0")/.." || exit 1   # repo root
M=${TSCT_MODELS_DIR:-$HOME/.oMLX/models}   # local MLX checkpoints; override with TSCT_MODELS_DIR
OUT=data/prep/predictions_7b
LOG=$OUT/b11_ladder.log
: > "$LOG"

run () {
  name=$1; path=$2
  if [ -f "$OUT/mixed_$name.jsonl" ]; then
    echo "=== $name: already scored, skipping ===" | tee -a "$LOG"
    return
  fi
  echo "=== $name ===" | tee -a "$LOG"
  python3 -m src.training.score_mixed_paragraphs \
    --base "$path" --out "$OUT/mixed_$name.jsonl" >> "$LOG" 2>&1 \
    || echo "!!! $name FAILED (see log above) !!!" | tee -a "$LOG"
}

# small -> large, so the cheap families report before the 31G run
run g3_4b        "$M/gemma-3-4b-it-4bit"
run qwen3_4b_it  "$M/Qwen3-4B-Instruct-2507-MLX-8bit"
run qwen3_4b_th  "$M/Qwen3-4B-Thinking-2507-MLX-8bit"
run g4_e4b       "$M/gemma-4-E4B-it-MLX-4bit"
run gptoss_20b   "$M/openai-gpt-oss-20b-MLX-6.5bit"
run qwen36_27b   "$M/Qwen3.6-27B-OBLITERATED-MLX-4bit"
run g4_31b       "$M/gemma-4-31B-it-MLX-8bit"
run qwen36_35b   "$M/Qwen3.6-35B-A3B-nvfp4"

echo "=== LADDER DONE ===" | tee -a "$LOG"
