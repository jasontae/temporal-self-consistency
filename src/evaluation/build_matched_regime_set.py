"""Build a form-matched, within-entity claim set for the regime question.

## Why this exists

B11 asked whether a model's own log-probability separates temporally volatile
claims from stable ones, and reported 0.9183 within-passage. That figure did not
survive its validity screen (`b11_surface_control`, `b11_dedup`):

  - the source artifact holds **22 distinct claims** presented as 233 rows;
  - **100%** of volatile claims are `ENTITY's ROLE is PERSON NAME` while the
    stable ones are science facts, so a no-model "is the predicate a proper
    noun" heuristic scores **0.8333**, statistically tied with every model;
  - every volatile claim is also **stale**, so low log-probability is equally
    explained by the claim being *false* rather than *volatile*.

The confound is structural, so no reanalysis fixes it. It needs a different
claim set, which is what this builds.

## The design

For a single organization, Wikidata carries both a volatile person-valued
property and an immutable one:

    volatile   P169 CEO / P6 head of government / P488 chairperson
    immutable  P112 founded by

So the same entity yields claims in an identical frame, with the same answer
type, differing only in whether the property's value can change:

    The CEO of Nike is Elliott Hill.        volatile, true now
    The CEO of Nike is John Donahoe.        volatile, stale (was true)
    The founder of Nike is Phil Knight.     immutable, true
    The founder of Nike is Steve Jobs.      immutable, false

This is a **within-entity** control, strictly stronger than the within-passage
one that failed: entity popularity, topic and tokenization of the entity string
are held exactly fixed, not merely balanced. An entity is kept only if all four
arms exist, so the design is complete by construction.

## What each contrast isolates

| contrast | isolates |
|---|---|
| volatile_current vs immutable_true | **regime** -- both true, both person, same frame |
| immutable_true vs immutable_false | **truth**, with volatility held out |
| volatile_stale vs volatile_current | **staleness** |

The headline test is the first. If log-probability tracks only truth, volatile
and immutable true claims score alike. Separation there is the regime signal
B11 wanted, measured where form cannot explain it.

## The residual confound, and the control for it

Founders skew historically famous (Phil Knight) while sitting CEOs can be
obscure (Elliott Hill), so raw surprisal partly tracks **name frequency**, not
volatility. Each record therefore carries a `name_only_text` -- the same person
in a neutral frame -- so the scorer can report a PMI-style figure
(`logprob(claim) - logprob(name alone)`) that nets out the name's marginal
frequency. Both raw and normalised results should be read together.

Usage:
    python3 -m src.evaluation.build_matched_regime_set --max-entities 400
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .verify_gold_currency import (
    current_value,
    fetch_claims,
    fetch_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "data" / "prep" / "gold_currency_audit.json"
CACHE = REPO_ROOT / "data" / "prep" / "wikidata_matched_cache"
OUT = REPO_ROOT / "data" / "stress_tests" / "matched_regime_claims.jsonl"

# Volatile, person-valued. P749/P127/P463/P36 are excluded: their values are
# organizations or places, so they cannot share a frame with a person name.
VOLATILE_PROPS = {"P169", "P6", "P488", "P1308", "P35", "P1037"}

# Immutable, person-valued. Who founded a thing is a historical fact.
IMMUTABLE_PROP = "P112"
IMMUTABLE_ROLE = "founder"

# Roles are matched against a closed list, longest first. A bare non-greedy
# `(.+?) of (.+?)` splits "head of government of Chicago" into role="head" and
# entity="government of Chicago", which reads correctly in the volatile frame
# and produces "The founder of government of Chicago" in the immutable one.
ROLES = [
    "head of government", "head of state", "director or manager",
    "officeholder", "chairperson", "chairman", "chairwoman", "president",
    "CEO", "owner",
]
QUESTION_RE = re.compile(
    r"^Who is (?:the )?(" + "|".join(re.escape(r) for r in
                                     sorted(ROLES, key=len, reverse=True))
    + r") of (.+?)\?$"
)
SEED = 20260812
HUMAN = "Q5"


def is_human(entity):
    """True if the item is `instance of` (P31) human.

    P112 admits organizations as founders, and an organization name in the
    answer slot ("The founder of X is Fastweb.") breaks the invariant the whole
    design rests on -- that every arm ends in a person's name. Answers that are
    not people are dropped rather than repaired.
    """
    for c in (entity or {}).get("claims", {}).get("P31", []):
        snak = c.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        if snak.get("datavalue", {}).get("value", {}).get("id") == HUMAN:
            return True
    return False


def parse_question(q):
    """'Who is the CEO of Nike?' -> ('CEO', 'Nike'). None if it does not fit."""
    m = QUESTION_RE.match(q.strip())
    if not m:
        return None
    role, entity = m.group(1).strip(), m.group(2).strip()
    if not role or not entity:
        return None
    return role, entity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-entities", type=int, default=400,
                    help="cap on entities kept, for runtime")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    audit = json.loads(AUDIT.read_text())

    # ---- candidate volatile claims, one per entity ---------------------------
    cands = {}
    for r in audit["rows"]:
        if r.get("verdict") != "gold_is_stale":
            continue
        if r.get("property") not in VOLATILE_PROPS:
            continue
        if not r.get("wikidata_current_label") or not r.get("dataset_gold"):
            continue
        parsed = parse_question(r.get("question", ""))
        if not parsed:
            continue
        role, entity = parsed
        # the current and stale holders must differ, or the arms collapse
        if r["wikidata_current_label"].strip() == r["dataset_gold"].strip():
            continue
        cands.setdefault(r["entity_id"], {
            "entity_id": r["entity_id"], "entity": entity, "role": role,
            "property": r["property"],
            "current": r["wikidata_current_label"].strip(),
            "current_qid": r.get("wikidata_current_qid"),
            "stale": r["dataset_gold"].strip(),
        })

    print(f"[build] {len(cands)} entities with a usable volatile pair")

    # ---- immutable arm: founder of the same entity ---------------------------
    print(f"[build] fetching {IMMUTABLE_PROP} for those entities ...")
    ents = fetch_claims(list(cands), CACHE)
    founder_qid = {}
    for eid in cands:
        q = current_value(ents.get(eid), IMMUTABLE_PROP)
        if q:
            founder_qid[eid] = q
    print(f"[build] {len(founder_qid)} entities also carry {IMMUTABLE_PROP}")

    # Every arm must end in a person's name, so both the founder and the
    # current office-holder are gated on P31 -> Q5. Without this the immutable
    # arm leaks organizations ("The founder of X is Fastweb.") and the contrast
    # measures answer *type* rather than volatility.
    # NOTE: fetch_claims names its cache files by batch index, so a second call
    # sharing CACHE would re-read the organization batches above. Person types
    # get their own directory.
    print("[build] checking founders and office-holders are people ...")
    # Only entities that already have a founder can survive, so the humanity
    # check is restricted to those -- checking all ~1,700 candidates would be
    # ten times the API traffic for the same result.
    person_qids = set(founder_qid.values())
    person_qids |= {cands[e]["current_qid"] for e in founder_qid
                    if cands[e].get("current_qid")}
    ptype_cache = CACHE / "person_types"
    ptype_cache.mkdir(parents=True, exist_ok=True)
    ptypes = fetch_claims(sorted(person_qids), ptype_cache)
    human = {q for q in person_qids if is_human(ptypes.get(q))}
    print(f"[build] {len(human)}/{len(person_qids)} answer entities are human")

    founder_qid = {e: q for e, q in founder_qid.items() if q in human}
    print(f"[build] {len(founder_qid)} entities keep a human founder")

    labels = fetch_labels(set(founder_qid.values()), CACHE)
    kept = {e: labels[q] for e, q in founder_qid.items() if labels.get(q)}
    # the volatile arm must also resolve to a human holder
    kept = {e: v for e, v in kept.items()
            if cands[e].get("current_qid") in human}
    print(f"[build] {len(kept)} entities with human founder AND human holder")

    # A founder who also held the volatile office ("Wings of the Ocean": Julien
    # Wosnitza is both founder and former chairperson) makes the arms
    # non-independent -- the stale claim is then true under the other reading.
    # Those entities are dropped rather than adjudicated.
    overlap = {e for e, f in kept.items()
               if f in (cands[e]["current"], cands[e]["stale"])}
    for e in overlap:
        del kept[e]
    print(f"[build] dropped {len(overlap)} where the founder also held the office; "
          f"{len(kept)} remain")

    # deterministic subsample, so the set is reproducible from the seed alone
    eids = sorted(kept)
    rng = random.Random(SEED)
    if len(eids) > args.max_entities:
        eids = sorted(rng.sample(eids, args.max_entities))
    print(f"[build] keeping {len(eids)} entities -> {4*len(eids)} claims")

    # ---- false-founder control: permute founders across entities -------------
    # A permuted name is a real founder of some other entity, so the false arm
    # matches the true arm in name distribution and only the pairing is wrong.
    pool = [kept[e] for e in eids]
    shuffled = pool[:]
    rng.shuffle(shuffled)
    for i, e in enumerate(eids):  # derangement: never leave a name in place
        if shuffled[i] == kept[e]:
            j = (i + 1) % len(shuffled)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    rows = []
    for i, eid in enumerate(eids):
        c = cands[eid]
        ent, role = c["entity"], c["role"]
        arms = [
            ("volatile_current", "volatile", "true", c["current"],
             f"The {role} of {ent} is {c['current']}."),
            ("volatile_stale", "volatile", "false", c["stale"],
             f"The {role} of {ent} is {c['stale']}."),
            ("immutable_true", "immutable", "true", kept[eid],
             f"The {IMMUTABLE_ROLE} of {ent} is {kept[eid]}."),
            ("immutable_false", "immutable", "false", shuffled[i],
             f"The {IMMUTABLE_ROLE} of {ent} is {shuffled[i]}."),
        ]
        for arm, vol, truth, person, text in arms:
            rows.append({
                "entity_id": eid, "entity": ent,
                "role": role if vol == "volatile" else IMMUTABLE_ROLE,
                "arm": arm, "volatility": vol, "truth": truth,
                "person": person, "text": text,
                # neutral frame, for the name-frequency control
                "name_only_text": f"{person} is a person.",
                "property": c["property"] if vol == "volatile" else IMMUTABLE_PROP,
            })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_arm = defaultdict(int)
    for r in rows:
        by_arm[r["arm"]] += 1
    print(f"\n[build] wrote {OUT}")
    print(f"[build] {len(rows)} claims, {len(set(r['text'] for r in rows))} distinct texts")
    for a, n in sorted(by_arm.items()):
        print(f"   {a:20s} {n}")
    print("\n[build] sample entity:")
    for r in rows[:4]:
        print(f"   {r['arm']:20s} {r['text']}")


if __name__ == "__main__":
    main()
