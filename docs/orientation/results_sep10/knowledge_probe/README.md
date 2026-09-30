# Knowledge probe: PARTIAL (stopped 2026-09-30 for the YBPA GPU reservation)

Plan: `../../PLAN_knowledge_probe.md` (committed `4b130ac` before any generation). Runner `run_knowledge_probe.sh`,
generation `knowledge_probe.py`, analysis `knowledge_analysis.py` → `knowledge_results.json`,
`audit_{qa_answer,frame_completion}.jsonl`, log `knowledge_probe.log`.

**Status.** Edward reserved the GPU for YBPA until 2026-10-01 12:00 UTC, so the queue was stopped mid-run.
Finished: gemma-3-4B, Qwen3-4B-Instruct, Qwen2.5-7B-Instruct, gpt-oss-20B, Qwen3.6-35B-A3B (5 of the 9 admitted
models), plus the Qwen2.5-7B base (frame only; not in the pooled numbers). **Missing: Qwen3.5-27B, Qwen3.6-27B,
Qwen3-4B-Thinking and the TSCT adapter** (about 15 min of GPU; the TSCT row needs the adapter path in the main
checkout, `/Users/edward/Projects/temporal-self-consistency/data/prep/tcl_mlx_7b/tsct_seed1/adapter_fixed`). The
two missing 27B models are among the most knowledgeable, so the `current` group below will grow.

**Matcher audit.** All `current`/`stale` matches were read (`audit_qa_answer.jsonl`); all were correct, one
`ambiguous` (two al-Qurashi names). 50 random `other` rows were read: none was a missed variant of either holder.
They are other people, older holders (for example Claudia Sheinbaum for Mexico City), or refusals (gpt-oss).

## Partial result (5 admitted models, 530 model-entity pairs, fixed 4-bit PMI scores)

QA probe ("Who is the current ROLE of ENTITY? Answer with only the name"):

| group (what the model names) | pairs | entities | paired win 1[current > expired] [95% entity CI] | z PMI difference [CI] |
|---|---|---|---|---|
| all | 530 | 106 | 0.443 [0.392, 0.492] | −0.165 [−0.285, −0.045] |
| names the current holder | 12 | 11 | 0.583 [0.231, 0.875] | +0.09 [−0.56, +0.65] |
| names the expired holder | 41 | 23 | 0.390 [0.213, 0.583] | −0.39 [−0.81, +0.08] |
| names neither (other/refusal) | 476 | 106 | **0.445 [0.393, 0.495]** | **−0.151 [−0.269, −0.035]** |

Frame completion gives the same picture (current 8 pairs, 0.625; expired 52 pairs, 0.346 [0.190, 0.511]; neither
470 pairs, 0.451 [0.395, 0.503]).

Per model, the models name the current holder on 0–10 of 106 entities (Qwen3.6-35B-A3B 10, the rest 0–1) and the
expired holder on 0–15.

## Reading, against the plan's decision rule

- **Primary (the `current` group): inconclusive.** 12 pairs, below the plan's minimum of 30. The point estimate
  (0.58) leans toward H-gap: where a model can name the current holder, it does not prefer the expired one.
- **What the partial data do show clearly: the preference lives almost entirely where the models can name
  neither holder.** About 90% of pairs are in that group, and it carries the full effect (0.445, CI excluding 0.5).
  So the paper's familiarity result is not "the model knows the update but still prefers the old value", and it
  is not only "the model's knowledge is outdated" either. It is a preference for the expired value on entities
  the models cannot recall at all, which is a weaker and differently worded claim than the current text implies
  ("every admitted model finds the expired office-holder more familiar").
- This is also how the result can coexist with Hossain et al. (2026), where a date prefix recovers the newer fact
  in 61–81% of cases: they condition on facts the model recalls; our matched set is dominated by facts it does not.
- Nothing here is final until the four missing models are run. The stale group (0.39) is directionally what an
  outdated-knowledge account predicts but its CI includes 0.5.
