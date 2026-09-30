# Plan (proposal, not run): in-context evidence control for the staleness instrument (worker E, 2026-09-30)

Status: **approved 2026-09-30; gated to start after 2026-10-01 12:00 UTC** (see Amendment 1). Originally a proposal. Written during the YBPA GPU reservation (no GPU until 2026-10-01 12:00 UTC). Inference
only, about 30 min of GPU. Runs after the reservation ends through the gate in `strategy_autostart/`. Commit this plan before
running; report every result.

## Why

Two things in the paper rest on the staleness instrument (name-controlled PMI of "The ROLE of ENTITY is PERSON."),
and neither has a positive control:
1. §4.4 reads a below-chance staleness AUROC as the model leaning toward the expired value. That reading assumes
   the instrument moves when the model is given evidence about which value is current. The date probe moves it by
   only 0.012–0.019, which is consistent both with a stubborn preference and with an instrument that barely
   responds to context.
2. The Discussion motivates an external solution (retrieval, provenance). Nothing in the paper shows that putting
   the current fact in context changes the model's scores.

A known-answer manipulation check answers both, in the same spirit as the truth check the paper already uses.

## Design

- Claims: the current and expired office-holder arms of `data/stress_tests/matched_regime_claims_disjoint.jsonl`
  (106 entities).
- Three per-entity contexts, each scored on the claim's own tokens only (a per-claim version of the existing
  `score_matched_regime --context`; the neutral name frame is scored without context, as in the date probe):
  - **current evidence:** "In {start year}, {current holder} became the {role} of {entity}."
  - **expired evidence:** "In {start year of the expired term}, {expired holder} became the {role} of {entity}."
  - **neutral filler** of matched length that names neither holder ("{entity} published its annual report.").
  Start years come from the P580 qualifiers already cached for `a5_prominence`.
- Models: the nine admitted checkpoints at 4-bit, as in `results_sep10/date_probe/run_date_probe.sh`.

## Metric

Paired win rate 1[PMI(current) > PMI(expired)] per context, pooled over the nine models with the entity-cluster
bootstrap used in `a5_prominence`; the difference current-evidence minus neutral.

## Decision rule

- **Instrument passes** if current evidence raises the win rate above neutral by at least 0.20 with a CI excluding
  0, and expired evidence lowers it. Then §4.4 can say the preference is what the model shows *without* evidence,
  and that evidence in context overrides it (one sentence plus an appendix row; this also supports the
  Discussion's external-solution sentence with data).
- **Instrument fails** if current evidence moves the win rate by less than 0.05. Then the staleness result is
  a property of an instrument that does not track in-context evidence, and §4.4 must be weakened; report it.
- In between: report as partial sensitivity.

## Cost and page budget

About 30 min of GPU (9 models × 3 contexts × about 1 min), 1 h of code and analysis. One sentence in §4.4 and one
appendix row; no new figure.

## Amendment 1 (2026-09-30, before any data)

- **Contexts use one template for all three conditions,** differing only in the name:
  - current evidence: "News: {current holder} has been appointed {role} of {entity}."
  - expired evidence: "News: {expired holder} has been appointed {role} of {entity}."
  - neutral: "News: {entity} has published its annual report."
- **Why:** the cached Wikidata claims give the current holder's start year but not the expired holder's. Dated
  sentences would also differ in length and specificity between the two evidence conditions. A shared template
  removes both asymmetries.
- **Implementation:** `score_matched_regime --context-field context` scores each claim's own tokens after its row's
  context. Claims are built by `results_sep10/context_control/build_context_claims.py`; the analysis is
  `context_analysis.py`. Metric and decision rule are unchanged, with the neutral context as the reference. The
  no-context 4-bit scores (`disjoint4b_*`) are reported alongside.
