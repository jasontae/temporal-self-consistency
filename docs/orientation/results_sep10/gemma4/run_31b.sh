#!/bin/bash
# Gemma-4 31B design (SEP10_PLAN.md): G1 quantization, G2 instruct vs pretrained, G4 first token.
# Same diagnostics on each checkpoint, serially.
cd "$(dirname "$0")/../../../.." || exit 1
M=$HOME/.oMLX/models
A=docs/orientation/results_sep10/model_audit
G=docs/orientation/results_sep10/gemma4
LOG=$G/run_31b.log
for x in "g4_31b_it_8bit gemma-4-31B-it-MLX-8bit" "g4_31b_it_4bit gemma-4-31b-it-4bit" "g4_31b_pt_4bit gemma-4-31b-pt-4bit"; do
  set -- $x
  for step in "audit python3 $A/audit_models.py --name $1 --base $M/$2 --no-generate" \
              "positions python3 $A/diag_positions.py --name $1 --base $M/$2" \
              "temperature python3 $A/diag_temperature.py --name $1 --base $M/$2" \
              "first_token python3 $G/first_token_test.py --name $1 --base $M/$2"; do
    label=${step%% *}; cmd=${step#* }
    [ "$label" = audit ] && [ -f $A/runs/$1.json ] && continue
    s=$(date +%s)
    $cmd >> "$LOG" 2>&1 || echo "!!! $1 $label FAILED" >> "$LOG"
    echo "=== $1 $label wall $(( $(date +%s) - s ))s" >> "$LOG"
  done
done
echo "=== 31B DONE $(date -u +%FT%TZ)" >> "$LOG"
