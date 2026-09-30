#!/bin/bash
# Serial queue for the LLaMA-3 seeds: order tsct0 (already running), sft0, tsct1, sft1, ... sft4.
# Before each arm-seed: pause (exit) only if memory pressure is critical (not on swap level).
# Commit after each seed pair (no push).
cd "$(dirname "$0")/../../../.." || exit 1
R=docs/orientation/results_sep10/item5_llama3
LOG=$R/llama3_seeds.log
guard () {
  # Edward, 2026-09-29: high swap is acceptable; pause only if memory pressure is critical.
  # (Step-time thrashing is watched separately by llama3_watchdog.py.)
  lvl=$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null)   # 1 normal, 2 warn, 4 critical
  echo "=== guard $(date -u +%FT%TZ) pressure_level=$lvl swap=$(sysctl -n vm.swapusage) | $(memory_pressure | tail -1)" >> "$LOG"
  if [ "$lvl" = "4" ]; then
    echo "=== PAUSED: memory pressure critical before $1 -- not starting" >> "$LOG"; exit 3
  fi
}
commit_pair () {
  s=$1
  git add data/prep/tcl_mlx_llama3/tsct_seed$s/run_meta.json data/prep/tcl_mlx_llama3/tsct_seed$s/loss_log.csv \
          data/prep/tcl_mlx_llama3/sft_only_seed$s/run_meta.json data/prep/tcl_mlx_llama3/sft_only_seed$s/loss_log.csv \
          data/prep/predictions_llama3/tsct_seed${s}_*.jsonl data/prep/predictions_llama3/sft_only_seed${s}_*.jsonl "$LOG" 2>/dev/null
  git commit -q -m "Item 5 LLaMA-3: seed $s pair (TSCT and plain fine-tuning), trained and evaluated" >> "$LOG" 2>&1 \
    && echo "=== committed seed $s pair $(git log --oneline -1)" >> "$LOG"
}
# v3 (after the tokenizer fix): all ten arm-seeds in order. run_llama3_seed.sh skips any
# arm-seed whose adapter and predictions already exist. First wait until no seed is
# running, so nothing co-runs with a seed started by an earlier queue.
while pgrep -f "run_llama3_seed.sh" > /dev/null; do sleep 60; done
echo "=== queue v3 start $(date -u +%FT%TZ) (tokenizer fixed; all 10 arm-seeds)" >> "$LOG"
for s in 0 1 2 3 4; do
  for arm in tsct sft_only; do
    guard "$arm seed $s"
    bash $R/run_llama3_seed.sh $arm $s || { echo "=== STOPPED: $arm seed $s failed" >> "$LOG"; exit 1; }
  done
  commit_pair $s
done
echo "=== ALL LLAMA3 SEEDS DONE $(date -u +%FT%TZ)" >> "$LOG"
