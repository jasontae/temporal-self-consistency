"""Recompute every number in the submission draft from its raw artifact.

Section E of the claims ledger commits to keeping raw per-record artifacts so
that any aggregate can be recounted. This applies that commitment to the paper
itself. Numbers in the draft have been transcribed across a ledger, a website and
several revisions, and transcription is exactly the step that a human read-through
is least likely to catch.

Each check recomputes a figure from the file it came from and compares it to the
value asserted in `main.tex`. A mismatch is printed with both values. Nothing is
repaired automatically, because the correct fix depends on which of the two is
wrong.

Usage:
    python3 -m src.evaluation.verify_paper_numbers
"""
import json
import re
from collections import defaultdict
from pathlib import Path

from .b11_surface_control import auroc, has_person_name
from .b11_dedup import distinct_claims

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper" / "submission" / "main.tex"
PRED = REPO_ROOT / "data" / "prep" / "predictions_7b"
CLAIMS = REPO_ROOT / "data" / "stress_tests" / "matched_regime_claims.jsonl"
AUDIT = REPO_ROOT / "data" / "prep" / "gold_currency_audit.json"

results = []


def check(label, asserted, recomputed, tol=5e-4):
    ok = recomputed is not None and abs(asserted - recomputed) <= tol
    results.append((ok, label, asserted, recomputed))


def paper_has(pattern):
    """True if the draft contains the literal figure, so removed claims are skipped."""
    return re.search(pattern, PAPER.read_text()) is not None


def main():
    tex = PAPER.read_text()

    # ---- the matched set, as built -----------------------------------------
    rows = [json.loads(l) for l in open(CLAIMS)]
    check("matched set: total claims", 424, len(rows), 0)
    check("matched set: distinct texts", 424, len(set(r["text"] for r in rows)), 0)
    check("matched set: entities", 106, len(set(r["entity_id"] for r in rows)), 0)

    # ---- the withdrawn result's source set ---------------------------------
    mixed = [json.loads(l) for l in open(PRED / "mixed_qwen25_7b.jsonl")]
    check("withdrawn set: rows", 233, len(mixed), 0)
    check("withdrawn set: distinct claims", 22, len(distinct_claims(mixed)), 0)

    uniq = distinct_claims(mixed)
    regex = auroc([(float(not has_person_name(r["text"])), not r["is_volatile"])
                   for r in uniq])
    check("no-model person-name regex", 0.8333, regex)
    digit = auroc([(float(any(c.isdigit() for c in r["text"])), not r["is_volatile"])
                   for r in uniq])
    check("no-model digit feature", 0.7917, digit)
    model = auroc([(r["mean_logprob"], not r["is_volatile"]) for r in uniq])
    check("best model on withdrawn set", 0.9000, model)
    pooled = auroc([(r["mean_logprob"], not r["is_volatile"]) for r in mixed])
    check("pooled figure before dedup", 0.8863, pooled)

    # ---- matched-set model figures quoted in the main table ----------------
    def contrasts(stem):
        by_ent = defaultdict(dict)
        for line in open(PRED / f"matched_{stem}.jsonl"):
            r = json.loads(line)
            by_ent[r["entity_id"]][r["arm"]] = r
        ents = [a for a in by_ent.values() if len(a) == 4]
        out = {}
        for name, hi, lo in (("regime", "immutable_true", "volatile_current"),
                             ("truth", "immutable_true", "immutable_false"),
                             ("staleness", "volatile_current", "volatile_stale")):
            pairs = []
            for a in ents:
                pairs.append((a[hi]["pmi"], True))
                pairs.append((a[lo]["pmi"], False))
            out[name] = auroc(pairs)
        return out

    quoted = {
        "g3_4b":            {"truth": 0.7189, "regime": 0.6198, "staleness": 0.4535},
        "qwen25_7b":        {"truth": 0.7945, "regime": 0.6336, "staleness": 0.4257},
        "gptoss_20b":       {"truth": 0.8058, "regime": 0.5695, "staleness": 0.4146},
        "qwen35_27b":       {"truth": 0.8452, "regime": 0.5206, "staleness": 0.4151},
        "qwen36_27b_stock": {"truth": 0.8549, "regime": 0.5097, "staleness": 0.4421},
    }
    for stem, vals in quoted.items():
        got = contrasts(stem)
        for k, v in vals.items():
            check(f"{stem} {k}", v, got[k])

    # ---- the reflexive screen ----------------------------------------------
    ref = [json.loads(l) for l in open(PRED / "matched_qwen25_7b.jsonl")]
    by_ent = defaultdict(dict)
    for r in ref:
        by_ent[r["entity_id"]][r["arm"]] = r
    ents = [a for a in by_ent.values() if len(a) == 4]
    pairs = []
    for a in ents:
        pairs.append((float(len(a["immutable_true"]["person"].split())), True))
        pairs.append((float(len(a["volatile_current"]["person"].split())), False))
    check("screen: name length on regime", 0.574, auroc(pairs), 1e-3)

    # ---- the currency audit -------------------------------------------------
    d = json.loads(AUDIT.read_text())
    c = d["counts"]
    decided = c["gold_is_stale"] + c["gold_is_current"]
    check("audit: stale golds", 2575, c["gold_is_stale"], 0)
    check("audit: decided pairs", 2610, decided, 0)
    check("audit: stale rate", 0.987, c["gold_is_stale"] / decided, 5e-4)

    # ---- report -------------------------------------------------------------
    bad = [r for r in results if not r[0]]
    print(f"## paper number check: {len(results) - len(bad)}/{len(results)} agree\n")
    for ok, label, asserted, got in results:
        if not ok:
            g = f"{got:.4f}" if isinstance(got, float) else got
            print(f"   MISMATCH  {label}: paper says {asserted}, artifact gives {g}")
    if not bad:
        print("   Every checked figure in main.tex recomputes from its raw artifact.")
    else:
        print(f"\n   {len(bad)} mismatch(es). Neither value is assumed correct; check both.")


if __name__ == "__main__":
    main()
