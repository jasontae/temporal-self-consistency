#!/bin/bash
# C9's clean vintage contrast: Qwen3.5-27B vs Qwen3.6-27B.
#
# Both from mlx-community at the same 4-bit recipe, so family, parameter count
# and quantizer are held fixed and vintage is the only free variable. This is
# the pair C9 names as the control its current gemma-vs-Qwen comparison lacks.
M=${TSCT_MODELS_DIR:-$HOME/.oMLX/models}   # local MLX checkpoints; override with TSCT_MODELS_DIR
cd "$(dirname "$0")/.." || exit 1   # repo root
LOG=data/prep/predictions_7b/fetch_vintage.log
: > "$LOG"

for repo in mlx-community/Qwen3.5-27B-4bit mlx-community/Qwen3.6-27B-4bit; do
  dest="$M/$(basename "$repo")"
  echo "=== $repo -> $dest ===" | tee -a "$LOG"
  hf download "$repo" --local-dir "$dest" >> "$LOG" 2>&1 \
    && echo "OK $repo" | tee -a "$LOG" \
    || echo "!!! FAILED $repo !!!" | tee -a "$LOG"
done
echo "=== FETCH DONE ===" | tee -a "$LOG"
