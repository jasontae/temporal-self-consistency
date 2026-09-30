#!/bin/bash
# Item 4: every admitted model at one quantization (MLX 4-bit affine g64; gpt-oss
# in its native MXFP4 experts + Q4, the closest 4-bit form), plus the prompt
# control. Also finishes A6 for the two rows that needed the Qwen2.5-7B base.
# Serial, one model in memory at a time. Wall-clock per run in item4.log.
#
# 0. A6 completion: audit + disjoint rescore for Qwen2.5-7B and the TSCT adapter.
# 1. Audit each new 4-bit checkpoint (model_audit/audit_models.py).
# 2. Score the original and the disjoint claim sets -> matched4b_*, disjoint4b_*.
#    Rows already at 4-bit affine g64 reuse their existing outputs (the harness
#    reproduces those scores exactly: a6_disjoint/unchanged_arms.json).
# 3. Prompt control: the disjoint set scored as an assistant turn (--frame chat)
#    for all nine admitted models at 4-bit -> disjoint4bchat_*.
# gpt-oss runs last and waits for its download to complete.
cd "$(dirname "$0")/../../../.." || exit 1
M=$HOME/.oMLX/models
P=data/prep/predictions_7b
A=docs/orientation/results_sep10/model_audit
LOG=docs/orientation/results_sep10/item4/item4.log
DIS=data/stress_tests/matched_regime_claims_disjoint.jsonl
Q25=mlx-community/Qwen2.5-7B-Instruct-4bit
TSCT=data/prep/tcl_mlx_7b/tsct_seed1/adapter_fixed
echo "=== start $(date -u +%FT%TZ)" >> "$LOG"

audit () { s=$(date +%s); python3 $A/audit_models.py --name "$1" "${@:2}" >> "$LOG" 2>&1 || echo "!!! audit $1 FAILED" >> "$LOG"
           echo "=== audit $1 wall $(( $(date +%s) - s ))s" >> "$LOG"; }
score () {  # out, then scorer args
  out=$1; shift
  [ -f "$P/$out" ] && { echo "=== $out exists" >> "$LOG"; return; }
  s=$(date +%s)
  python3 -m src.training.score_matched_regime "$@" --out "$P/$out" >> "$LOG" 2>&1 || echo "!!! $out FAILED" >> "$LOG"
  echo "=== $out wall $(( $(date +%s) - s ))s" >> "$LOG"
}
complete () {  # dir: every indexed shard present and nothing incomplete
  python3 - "$1" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx):
    sys.exit(1)
files = set(json.load(open(idx))["weight_map"].values())
ok = all(os.path.exists(os.path.join(d, f)) for f in files) and not glob.glob(os.path.join(d, ".cache/huggingface/download/*.incomplete"))
sys.exit(0 if ok else 1)
PY
}

# 0. A6 completion
audit qwen25_7b --base $Q25
audit tsct --adapter $TSCT
score disjoint_qwen25_7b.jsonl --claims $DIS --base $Q25
score disjoint_tsct.jsonl --claims $DIS --adapter $TSCT

# 1-2. new 4-bit checkpoints
new () {  # tag dir [extra scorer args]
  tag=$1; dir=$2; shift 2
  audit "${tag}_4bit" --base "$M/$dir"
  score "matched4b_$tag.jsonl" --base "$M/$dir" "$@"
  score "disjoint4b_$tag.jsonl" --claims $DIS --base "$M/$dir" "$@"
  score "disjoint4bchat_$tag.jsonl" --claims $DIS --frame chat --base "$M/$dir" "$@"
}
new qwen3_4b_it Qwen3-4B-Instruct-2507-4bit
new qwen3_4b_th Qwen3-4B-Thinking-2507-4bit
new qwen36_35b  Qwen3.6-35B-A3B-4bit

# already 4-bit affine g64: reuse raw outputs, score the chat frame
for t in g3_4b qwen35_27b qwen36_27b_stock qwen38_27b qwen25_7b tsct; do
  for s in matched disjoint; do
    [ -f "$P/${s}_$t.jsonl" ] && [ ! -f "$P/${s}4b_$t.jsonl" ] && cp "$P/${s}_$t.jsonl" "$P/${s}4b_$t.jsonl" \
      && echo "=== reused ${s}_$t.jsonl as ${s}4b_$t.jsonl (already 4-bit affine g64)" >> "$LOG"
  done
done
score disjoint4bchat_g3_4b.jsonl            --claims $DIS --frame chat --base $M/gemma-3-4b-it-4bit
score disjoint4bchat_qwen35_27b.jsonl       --claims $DIS --frame chat --base $M/Qwen3.5-27B-4bit
score disjoint4bchat_qwen36_27b_stock.jsonl --claims $DIS --frame chat --base $M/Qwen3.6-27B-4bit
score disjoint4bchat_qwen38_27b.jsonl       --claims $DIS --frame chat --base $M/Qwen3.8-27B-4bit
score disjoint4bchat_qwen25_7b.jsonl        --claims $DIS --frame chat --base $Q25
score disjoint4bchat_tsct.jsonl             --claims $DIS --frame chat --adapter $TSCT

# gpt-oss last: wait for FETCH_ALL to finish its shards
until complete "$M/gpt-oss-20b-MXFP4-Q4"; do sleep 60; done
echo "=== gpt-oss-20b-MXFP4-Q4 complete $(date -u +%FT%TZ)" >> "$LOG"
new gptoss_20b gpt-oss-20b-MXFP4-Q4 --bos off
echo "=== ITEM4 DONE $(date -u +%FT%TZ)" >> "$LOG"
# analysis: docs/orientation/results_sep10/item4/item4_analysis.py
