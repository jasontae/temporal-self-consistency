"""B11 validity screen: is the regime signal about time, or about proper nouns?

B11 reports that mean token log-probability separates volatile from stable
claims at 0.9183 within-passage, and reads that as the model carrying a latent
regime signal. Before that reading survives, it has to clear the same screen
that collapsed B5: does a feature with *no model in it at all* do the same job?

Inspection of `stress_mixed_paragraphs.jsonl` shows the two classes differ in
grammatical form, not only in volatility:

    fast       "The CEO of OpenAI is Sam Altman."          -> person name
    immutable  "Carbon has an atomic number of 6."          -> small integer

A person's name is high-entropy under any language model. If that alone
reproduces the AUROC, then B11 measures "the predicate is a proper noun" and
the temporal reading is unsupported.

Three screens, in order of severity:

1. **Surface baselines.** AUROC of token count, character length, digit
   presence, and a crude capitalised-name detector. No model is consulted.
   If these match the logprob AUROC, B11 is a surface artifact.

2. **Duplicate audit.** Repeated claim texts inflate the pairwise AUROC and
   are already visible in the file ("World War 2 ended in 1945." twice).

3. **Within-form contrast.** Restrict to claims whose predicate is a person
   name and re-run. If volatile *names* still separate from stable *names*,
   the temporal signal survives the confound; if the AUROC falls to chance,
   it did not.

Usage:
    python3 -m src.evaluation.b11_surface_control
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = REPO_ROOT / "data" / "prep" / "predictions_7b"

# A capitalised token that is not sentence-initial and is not a known
# non-person capitalised entity. Deliberately crude -- the point is that even a
# crude detector should not rival a language model if B11 means what it claims.
STOPCAPS = {
    "The", "A", "An", "In", "It", "Its", "This", "That", "World", "War",
    "Earth", "Jupiter", "Mars", "Carbon", "Water", "United", "Nations",
    "States", "America", "Germany", "France", "China", "Japan", "Mount",
    "Everest", "Pacific", "Atlantic", "Ocean", "Sun", "Moon",
}


def has_person_name(text):
    """True if a non-initial capitalised token looks like a personal name."""
    toks = re.findall(r"[A-Z][a-zà-ÿÀ-ÿ'\-]+", text)
    cands = [t for t in toks if t not in STOPCAPS]
    # a name is usually two adjacent capitalised tokens ("Sam Altman")
    return len(cands) >= 2


def auroc(pairs):
    pos = [s for s, y in pairs if y]
    neg = [s for s, y in pairs if not y]
    if not pos or not neg:
        return None
    w = t = 0
    for a in pos:
        for b in neg:
            if a > b:
                w += 1
            elif a == b:
                t += 1
    return (w + 0.5 * t) / (len(pos) * len(neg))


def within_passage(rows, score_fn):
    """AUROC(score -> claim is STABLE), averaged over passages, weighted by n."""
    by_p = defaultdict(list)
    for r in rows:
        by_p[r["passage_idx"]].append(r)
    scored = []
    for g in by_p.values():
        a = auroc([(score_fn(r), not r["is_volatile"]) for r in g])
        if a is not None:
            scored.append((a, len(g)))
    tot = sum(n for _, n in scored)
    if not tot:
        return float("nan"), 0, 0
    return sum(a * n for a, n in scored) / tot, len(scored), tot


def main():
    files = sorted(PRED_DIR.glob("mixed_*.jsonl"))
    if not files:
        raise SystemExit("no mixed_*.jsonl found")
    ref = [json.loads(l) for l in open(files[0])]

    # ---- screen 2: duplicates ------------------------------------------------
    counts = Counter(r["text"] for r in ref)
    dupes = {t: c for t, c in counts.items() if c > 1}
    n_dup_rows = sum(c for c in dupes.values()) - len(dupes)
    print(f"## duplicate audit ({files[0].stem} as reference)")
    print(f"   {len(ref)} rows, {len(counts)} distinct texts, "
          f"{len(dupes)} texts repeated, {n_dup_rows} redundant rows")
    for t, c in sorted(dupes.items(), key=lambda kv: -kv[1])[:5]:
        print(f"     x{c}  {t[:78]}")

    # ---- screen 1: surface baselines ----------------------------------------
    # No model is consulted for any of these.
    print("\n## screen 1 -- surface baselines (no model consulted)")
    print("   AUROC(feature -> claim is STABLE), within passage\n")
    baselines = {
        "-n_tokens (shorter=stable)": lambda r: -r["n_tokens"],
        "-char_length": lambda r: -len(r["text"]),
        "has_digit": lambda r: float(bool(re.search(r"\d", r["text"]))),
        "NOT has_person_name": lambda r: float(not has_person_name(r["text"])),
    }
    for name, fn in baselines.items():
        a, np_, n = within_passage(ref, fn)
        print(f"   {name:32s} {a:.4f}   ({np_} passages, n={n})")

    lp, np_, n = within_passage(ref, lambda r: r["mean_logprob"])
    print(f"   {'mean_logprob (the B11 claim)':32s} {lp:.4f}   ({np_} passages, n={n})")

    # ---- class composition ---------------------------------------------------
    print("\n## class composition -- is form confounded with volatility?")
    tab = defaultdict(lambda: [0, 0])
    for r in ref:
        tab[r["volatility"]][int(has_person_name(r["text"]))] += 1
    print(f"   {'volatility':12s} {'no name':>9s} {'has name':>9s}  {'% name':>7s}")
    for v, (no, yes) in sorted(tab.items()):
        pct = 100 * yes / (no + yes) if (no + yes) else 0
        print(f"   {v:12s} {no:9d} {yes:9d}  {pct:6.1f}%")

    # ---- screen 3: within-form contrast, every model -------------------------
    print("\n## screen 3 -- restrict to person-name claims only")
    print("   If volatility is real, volatile names still separate from stable names.\n")
    print(f"   {'model':22s} {'all claims':>11s} {'names only':>11s} {'n':>6s}")
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        allа, _, _ = within_passage(rows, lambda r: r["mean_logprob"])
        named = [r for r in rows if has_person_name(r["text"])]
        na, _, nn = within_passage(named, lambda r: r["mean_logprob"])
        na_s = f"{na:.4f}" if nn else "  n/a"
        print(f"   {f.stem.replace('mixed_',''):22s} {allа:11.4f} {na_s:>11s} {nn:6d}")


if __name__ == "__main__":
    main()
