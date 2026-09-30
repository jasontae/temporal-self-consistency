#!/bin/bash
# Item 5 on the team's base (BRIEF_LLAMA3.md): one arm-seed per call, serial.
#   bash run_llama3_seed.sh <tsct|sft_only> <seed>
# Base: mlx-community/Meta-Llama-3-8B-4bit @ 7f296f1d (4-bit of meta-llama/Meta-Llama-3-8B,
# the base in all 49 team adapter configs). Same data and steps as route B
# (--n-per-volatility 4000: 8,008 examples, 6,006 steps); hedges on reserved tokens
# (no spare rows); plain "Question/Answer" prompt (the base has no chat template).
cd "$(dirname "$0")/../../../.." || exit 1
arm=$1; s=$2
B=$HOME/.oMLX/models/Meta-Llama-3-8B-4bit-tokfix  # tokenizer_class -> PreTrainedTokenizerFast; weights = Meta-Llama-3-8B-4bit @ 7f296f1d
T=data/prep/tcl_mlx_llama3
P=data/prep/predictions_llama3
LOG=docs/orientation/results_sep10/item5_llama3/llama3_seeds.log
mkdir -p $T $P
d=$T/${arm}_seed$s
echo "=== $arm seed $s start $(date -u +%FT%TZ) | $(memory_pressure | tail -1) | $(sysctl -n vm.swapusage)" >> "$LOG"
if [ ! -f $d/adapter_fixed/adapters.safetensors ]; then
  if [ $arm = tsct ]; then extra=""; else extra="--lambda-over 0 --lambda-under 0 --lambda-hedge 0"; fi
  st=$(date +%s)
  python3 -m src.training.run_tcl_mlx --model $B --seed $s --n-per-volatility 4000 \
    --hedge-tokens reserved --plain-prompt --out-dir $d $extra >> "$LOG" 2>&1 \
    || { echo "!!! train $arm seed $s FAILED" >> "$LOG"; exit 1; }
  echo "=== train $arm seed $s wall $(( $(date +%s) - st ))s" >> "$LOG"
fi
for src in "temporal-delta:test|test" "freshqa|freshqa"; do
  name=${src%%|*}; tag=${src##*|}; out=$P/${arm}_seed${s}_${tag}.jsonl
  [ -f $out ] && continue
  st=$(date +%s)
  python3 -m src.training.generate_predictions --adapter $d/adapter_fixed --source $name --out $out >> "$LOG" 2>&1 \
    || echo "!!! eval $arm seed $s $tag FAILED" >> "$LOG"
  echo "=== eval $arm seed $s $tag wall $(( $(date +%s) - st ))s" >> "$LOG"
done
echo "=== $arm seed $s done $(date -u +%FT%TZ)" >> "$LOG"
