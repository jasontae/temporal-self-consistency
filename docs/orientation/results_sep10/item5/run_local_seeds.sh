#!/bin/bash
# Item 5, route B: three more local seeds per arm (2, 3, 4), giving five per arm
# with the existing seed0/seed1 adapters. About 6 x (85 min train + 15 min eval),
# roughly 10 h wall. Needs FETCH_ALL.sh steps 1 (base model) and 2 (train split);
# run_tcl_mlx now refuses to start without the train split instead of silently
# training on 49 samples. Serial; one model in memory.
# --n-per-volatility 4000 reproduces the Aug seeds' 8,008 raw examples and 6,006
# steps (run_meta.json); the script default of 2000 gives 4,008 and would not be
# data- or compute-matched with seeds 0-1.
cd "$(dirname "$0")/../../../.." || exit 1
T=data/prep/tcl_mlx_7b
P=data/prep/predictions_7b
LOG=docs/orientation/results_sep10/item5/local_seeds.log
[ -s data/prep/temporal_delta/temporal_delta_train.jsonl ] || { echo "train split missing: FETCH_ALL.sh step 2"; exit 1; }
echo "=== start $(date -u +%FT%TZ)" >> "$LOG"
for s in 2 3 4; do
  for arm in tsct sft_only; do
    d=$T/${arm}_seed${s}_r2   # _r2: new runs, never overwrite the Aug dirs
    if [ ! -f $d/adapter_fixed/adapters.safetensors ]; then
      st=$(date +%s)
      if [ $arm = tsct ]; then extra=""; else extra="--lambda-over 0 --lambda-under 0 --lambda-hedge 0"; fi
      python3 -m src.training.run_tcl_mlx --seed $s --n-per-volatility 4000 --out-dir $d $extra >> "$LOG" 2>&1 \
        || { echo "!!! train $arm seed $s FAILED" >> "$LOG"; continue; }
      echo "=== train $arm seed $s wall $(( $(date +%s) - st ))s" >> "$LOG"
    fi
    out=$P/${arm}_seed${s}_test.jsonl
    if [ ! -f $out ]; then
      st=$(date +%s)
      python3 -m src.training.generate_predictions --adapter $d/adapter_fixed --out $out >> "$LOG" 2>&1 \
        || echo "!!! eval $arm seed $s FAILED" >> "$LOG"
      echo "=== eval $arm seed $s wall $(( $(date +%s) - st ))s" >> "$LOG"
    fi
  done
done
echo "=== SEEDS DONE $(date -u +%FT%TZ)" >> "$LOG"
# analysis: python3 docs/orientation/results_sep10/item5/seed_variance.py --route B
