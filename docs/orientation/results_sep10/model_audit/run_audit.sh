#!/bin/bash
# Serial model audit, one model in memory at a time. Logs wall-clock per model.
cd "$(dirname "$0")/../../../.." || exit 1
M=$HOME/.oMLX/models
A=docs/orientation/results_sep10/model_audit
LOG=$A/audit.log
: > "$LOG"
run () {
  name=$1; shift
  s=$(date +%s)
  echo "=== $name start $(date -u +%FT%TZ)" >> "$LOG"
  python3 $A/audit_models.py --name "$name" "$@" >> "$LOG" 2>&1 || echo "!!! $name FAILED" >> "$LOG"
  echo "=== $name wall $(( $(date +%s) - s ))s" >> "$LOG"
}
run g3_4b        --base $M/gemma-3-4b-it-4bit
run g4_e4b       --base $M/gemma-4-E4B-it-MLX-4bit
run qwen3_4b_it  --base $M/Qwen3-4B-Instruct-2507-MLX-8bit
run qwen3_4b_th  --base $M/Qwen3-4B-Thinking-2507-MLX-8bit
run gptoss_20b   --base $M/openai-gpt-oss-20b-MLX-6.5bit
run gemma4_26b   --base $M/gemma-4-26b-a4b-it-MLX-4bit
run gemma4_26b   --base $M/gemma-4-26b-a4b-it-MLX-4bit --force-vlm
run qwen35_27b   --base $M/Qwen3.5-27B-4bit
run qwen36_27b   --base $M/Qwen3.6-27B-4bit
run qwen38_27b   --base $M/Qwen3.8-27B-4bit
run g4_31b       --base $M/gemma-4-31B-it-MLX-8bit
run g4_31b       --base $M/gemma-4-31B-it-MLX-8bit --force-vlm
run qwen36_35b   --base $M/Qwen3.6-35B-A3B-nvfp4
echo "=== AUDIT DONE $(date -u +%FT%TZ)" >> "$LOG"
