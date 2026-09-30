#!/bin/bash
# Directive 3, Worker A: serial queue after item 4 (one model in memory at a time).
#   item 5 (local seeds) -> item 1 (FreshQA) -> base vs instruct -> date probe
cd "$(dirname "$0")/../../.." || exit 1
M=$HOME/.oMLX/models
P=data/prep/predictions_7b
LOG=docs/orientation/results_sep10/queue_d3.log
FQ=data/stress_tests/freshqa_claims.jsonl
Q25=mlx-community/Qwen2.5-7B-Instruct-4bit
until grep -q "ITEM4 DONE" docs/orientation/results_sep10/item4/item4.log 2>/dev/null; do sleep 30; done
echo "=== start $(date -u +%FT%TZ)" >> "$LOG"
run () {  # label, command...
  label=$1; shift; s=$(date +%s)
  "$@" >> "$LOG" 2>&1 || echo "!!! $label FAILED" >> "$LOG"
  echo "=== $label wall $(( $(date +%s) - s ))s" >> "$LOG"
}

# ---- item 5: local seeds 2-4, both arms
run item5_seeds bash docs/orientation/results_sep10/item5/run_local_seeds.sh
echo "=== ITEM5 DONE $(date -u +%FT%TZ)" >> "$LOG"

# ---- item 1: FreshQA
declare -a ADM=(
  "g3_4b|--base $M/gemma-3-4b-it-4bit"
  "qwen3_4b_th|--base $M/Qwen3-4B-Thinking-2507-4bit"
  "qwen3_4b_it|--base $M/Qwen3-4B-Instruct-2507-4bit"
  "qwen25_7b|--base $Q25"
  "gptoss_20b|--base $M/gpt-oss-20b-MXFP4-Q4"
  "qwen36_35b|--base $M/Qwen3.6-35B-A3B-4bit"
  "qwen35_27b|--base $M/Qwen3.5-27B-4bit"
  "qwen36_27b_stock|--base $M/Qwen3.6-27B-4bit"
)
for m in "${ADM[@]}"; do
  n=${m%%|*}; a=${m#*|}; bos=""; [ "$n" = gptoss_20b ] && bos="--bos off"
  # shellcheck disable=SC2086
  [ -f $P/fq_$n.jsonl ] || run "fq_logprob $n" python3 -m src.training.score_matched_regime $a $bos --claims $FQ --out $P/fq_$n.jsonl
  # shellcheck disable=SC2086
  [ -f $P/fqgen_$n.jsonl ] || run "fq_gen $n" python3 -m src.training.generate_predictions $a --source freshqa --answer-only --max-tokens 48 --out $P/fqgen_$n.jsonl
done
[ -f $P/fq_tsct.jsonl ] || run "fq_logprob tsct" python3 -m src.training.score_matched_regime --adapter data/prep/tcl_mlx_7b/tsct_seed1/adapter_fixed --claims $FQ --out $P/fq_tsct.jsonl
for d in data/prep/tcl_mlx_7b/tsct_seed0_paired data/prep/tcl_mlx_7b/tsct_seed1 data/prep/tcl_mlx_7b/sft_only_seed0 data/prep/tcl_mlx_7b/sft_only_seed1 \
         data/prep/tcl_mlx_7b/tsct_seed*_r2 data/prep/tcl_mlx_7b/sft_only_seed*_r2; do
  [ -f $d/adapter_fixed/adapters.safetensors ] || continue
  n=$(basename $d)
  [ -f $P/freshqa_$n.jsonl ] || run "fq_hedge $n" python3 -m src.training.generate_predictions --adapter $d/adapter_fixed --source freshqa --out $P/freshqa_$n.jsonl
done
echo "=== ITEM1 DONE $(date -u +%FT%TZ)" >> "$LOG"

# ---- base vs instruct (needs FETCH_EXTRA.sh step 1)
if [ -f $M/Qwen2.5-7B-4bit/config.json ]; then
  run "audit qwen25_7b_base" python3 docs/orientation/results_sep10/model_audit/audit_models.py --name qwen25_7b_base --base $M/Qwen2.5-7B-4bit --no-generate
  [ -f $P/matched_qwen25_7b_base.jsonl ] || run "matched base" python3 -m src.training.score_matched_regime --base $M/Qwen2.5-7B-4bit --out $P/matched_qwen25_7b_base.jsonl
  [ -f $P/disjoint_qwen25_7b_base.jsonl ] || run "disjoint base" python3 -m src.training.score_matched_regime --base $M/Qwen2.5-7B-4bit --claims data/stress_tests/matched_regime_claims_disjoint.jsonl --out $P/disjoint_qwen25_7b_base.jsonl
  echo "=== BASEINSTRUCT DONE $(date -u +%FT%TZ)" >> "$LOG"
else
  echo "=== BASEINSTRUCT SKIPPED: $M/Qwen2.5-7B-4bit missing (FETCH_EXTRA.sh step 1)" >> "$LOG"
fi

# ---- date probe
run date_probe bash docs/orientation/results_sep10/date_probe/run_date_probe.sh
echo "=== QUEUE D3 DONE $(date -u +%FT%TZ)" >> "$LOG"
