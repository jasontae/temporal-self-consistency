"""Apply our own form-validity screen to our own replacement set.

Section 4.3 of the paper proposes a screen: before accepting that a model
represents a property, compute the strongest baseline that consults no model and
report it beside the model. We applied that to `stress_mixed_paragraphs` (which
it refuted) and to EverGreenQA (which it passed). We had not applied it to
`matched_regime_claims.jsonl`, the set we built to replace the first one.

That omission is the obvious reviewer question, and it is the one case where a
failure would be fatal: if a surface feature separates the arms of a set we
designed to be form-matched, the replacement inherits the defect it was built to
remove.

The matched set should pass by construction. Every arm shares the frame
`The ROLE of ENTITY is PERSON.`, every answer is gated on `P31 -> Q5`, and the
false-founder arm is a derangement of the true-founder arm, so the same person
names appear on both sides. The screen therefore has a specific prediction: all
surface baselines should sit at chance on all three contrasts. This script checks
that rather than asserting it.

Baselines, none of which consult a model:

  n_chars, n_words        length of the claim
  n_name_tokens           length of the person's name
  has_digit               presence of a digit
  role_is_founder         whether the frame says "founder"  (see note)
  person_name_frequency   how often that person appears across the set

`role_is_founder` is expected to separate the regime contrast perfectly, because
the regime contrast *is* volatile-role versus founder-role. That is not a
confound, it is the manipulation. It is included as a positive control on the
screen itself: a screen that cannot detect a feature we know is present is not
measuring anything.

Usage:
    python3 -m src.evaluation.screen_matched_set
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .b11_surface_control import auroc
from .b11_dedup import boot_ci

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "data" / "prep" / "predictions_7b"

CONTRASTS = [
    ("regime", "immutable_true", "volatile_current"),
    ("truth", "immutable_true", "immutable_false"),
    ("staleness", "volatile_current", "volatile_stale"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(PRED_DIR / "matched_qwen25_7b.jsonl"),
                    help="scored claim file; its pmi is the model row")
    args = ap.parse_args()
    src = Path(args.src)
    rows = [json.loads(l) for l in open(src)]

    freq = Counter(r["person"] for r in rows)
    feats = {
        "n_chars": lambda r: float(len(r["text"])),
        "n_words": lambda r: float(len(r["text"].split())),
        "n_name_tokens": lambda r: float(len(r["person"].split())),
        "has_digit": lambda r: float(bool(re.search(r"\d", r["text"]))),
        "person_frequency": lambda r: float(freq[r["person"]]),
        "role_is_founder": lambda r: float(r["role"] == "founder"),
    }

    by_ent = defaultdict(dict)
    for r in rows:
        by_ent[r["entity_id"]][r["arm"]] = r
    ents = [a for a in by_ent.values() if len(a) == 4]

    print("## Form-validity screen applied to our own replacement set")
    print(f"   {len(ents)} entities, {len(rows)} claims, "
          f"{len(set(r['text'] for r in rows))} distinct texts\n")
    print("   AUROC of each no-model feature on each contrast.")
    print("   Chance is 0.5. `role_is_founder` is the manipulation, not a confound:")
    print("   it should separate `regime` and nothing else.\n")

    width = max(len(f) for f in feats)
    print(f"   {'feature':{width}s} " + "".join(f"{c[0]:>22s}" for c in CONTRASTS))
    for name, fn in feats.items():
        cells = []
        for _, hi, lo in CONTRASTS:
            pairs = []
            for a in ents:
                pairs.append((fn(a[hi]), True))
                pairs.append((fn(a[lo]), False))
            v = auroc(pairs)
            ci = boot_ci(pairs)
            cells.append(f"{v:.3f} [{ci[0]:.2f},{ci[1]:.2f}]" if ci else f"{v:.3f}")
        print(f"   {name:{width}s} " + "".join(f"{c:>22s}" for c in cells))

    print("\n   model log-probability, for comparison:")
    cells = []
    for _, hi, lo in CONTRASTS:
        pairs = []
        for a in ents:
            pairs.append((a[hi]["pmi"], True))
            pairs.append((a[lo]["pmi"], False))
        cells.append(f"{auroc(pairs):.3f}")
    print(f"   {'pmi (' + rows[0].get('model', src.stem) + ')':{width}s} " + "".join(f"{c:>22s}" for c in cells))

    print("\n## reading")
    print("   PASS if every surface feature except role_is_founder sits at chance")
    print("   on every contrast. A surface feature above chance on `truth` or")
    print("   `staleness` would mean the replacement set carries its own confound.")


if __name__ == "__main__":
    main()
