"""Validity screen on EverGreenQA (Pletenev et al., EMNLP 2025).

We ran this screen on our own regime probe and it refuted it: a no-model regex
for "is the predicate a proper noun" scored 0.8333 where the best model scored
0.9000, because 100% of volatile claims were `ENTITY's ROLE is PERSON` and the
stable ones were science facts (B11-X). The screen is not specific to our data,
so it should be run against the published benchmark that asks the same question.

EverGreenQA labels *questions*, not claims, so answer form cannot be the
confound. The analogue is question form:

    "And who is the king of the United Kingdom?"      -> mutable
    "When did the Second World War end?"              -> evergreen

If the wh-word and tense alone separate the classes, then a classifier trained
on this data can score well without representing temporality at all, and the
reported numbers partly measure surface regularity. Their Appendix F controls
question *complexity*; it does not control question *form*.

Three baselines, none of which consult a language model:

1. **Single surface features** -- length, wh-word, tense marker, deixis
   ("current", "now", "today"). Reported as AUROC and as weighted F1 so they
   sit on the same scale as the paper's Table 2.
2. **Lexical logistic regression** -- bag of words, no embeddings, no
   pretraining. This is the strongest "no semantics" baseline.
3. **First-token rule** -- the single most predictive opening word, fit on
   train and applied to test.

Reference points from the paper (weighted F1, English, test):
    random               0.637
    EG-E5 (their SoTA)   0.913
    best few-shot LLM    0.885   (LLaMA 3.1 70B)
    GPT-4.1              0.794

Usage:
    python3 -m src.evaluation.evergreen_surface_screen --data <dir>
"""
import argparse
import re
from collections import Counter
from pathlib import Path

from .b11_surface_control import auroc

# label 1 == evergreen (stable), 0 == mutable. Scores are oriented so that a
# higher score predicts EVERGREEN, matching `auroc(... , y=is_evergreen)`.
WH = ["who", "when", "where", "what", "which", "how", "why", "is", "does", "did"]
PAST = re.compile(r"\b(did|was|were|had|ended|founded|discovered|invented|born)\b", re.I)
PRESENT = re.compile(r"\b(is|are|currently|now|today|current|latest|present)\b", re.I)
DEIXIS = re.compile(r"\b(current|currently|now|today|latest|recent|nowadays|these days)\b", re.I)


def weighted_f1(y_true, y_pred):
    """Weighted-average F1 over both classes, as reported in the paper."""
    total = len(y_true)
    out = 0.0
    for cls in (0, 1):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out += f1 * sum(1 for t in y_true if t == cls) / total
    return out


def first_word(q):
    m = re.findall(r"[A-Za-z']+", q.lower())
    return m[0] if m else ""


def wh_word(q):
    """First interrogative in the question, else ''. Handles 'And who is ...'."""
    toks = re.findall(r"[a-z']+", q.lower())
    for t in toks:
        if t in WH:
            return t
    return ""


def fit_lexical_lr(train_q, train_y, epochs=60, lr=0.5, l2=1e-4):
    """Bag-of-words logistic regression, plain Python, no libraries.

    Deliberately the weakest possible learner: unigram presence, no embeddings,
    no pretraining, no word order. If this approaches the paper's numbers, the
    task is substantially lexical.
    """
    vocab = {}
    for q in train_q:
        for w in set(re.findall(r"[a-z']+", q.lower())):
            if w not in vocab:
                vocab[w] = len(vocab)
    w = [0.0] * len(vocab)
    b = 0.0
    feats = [[vocab[t] for t in set(re.findall(r"[a-z']+", q.lower())) if t in vocab]
             for q in train_q]
    n = len(feats)
    for _ in range(epochs):
        gw = [0.0] * len(vocab)
        gb = 0.0
        for idx, y in zip(feats, train_y):
            z = b + sum(w[i] for i in idx)
            p = 1.0 / (1.0 + pow(2.718281828459045, -max(-30.0, min(30.0, z))))
            d = p - y
            gb += d
            for i in idx:
                gw[i] += d
        b -= lr * gb / n
        for i in range(len(w)):
            w[i] -= lr * (gw[i] / n + l2 * w[i])
    return vocab, w, b


def lr_score(q, vocab, w, b):
    idx = [vocab[t] for t in set(re.findall(r"[a-z']+", q.lower())) if t in vocab]
    return b + sum(w[i] for i in idx)


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir holding the parquet files")
    args = ap.parse_args()
    d = Path(args.data)
    tr = pd.read_parquet(d / "train-00000-of-00001.parquet")
    te = pd.read_parquet(d / "test-00000-of-00001.parquet")

    trq = [str(x) for x in tr["English"]]
    trY = [int(x) for x in tr["label"]]
    teq = [str(x) for x in te["English"]]
    teY = [int(x) for x in te["label"]]

    print("## EverGreenQA, English split")
    print(f"   train {len(trq)} rows, {len(set(trq))} distinct")
    print(f"   test  {len(teq)} rows, {len(set(teq))} distinct")
    print(f"   test class balance: evergreen={sum(teY)}, mutable={len(teY)-sum(teY)}")
    overlap = set(q.strip().lower() for q in trq) & set(q.strip().lower() for q in teq)
    print(f"   train/test text overlap: {len(overlap)} questions")

    print("\n## screen 1 -- single surface features on TEST (no model consulted)")
    print(f"   {'feature':30s} {'AUROC':>7s} {'wF1':>7s}")
    feats = {
        "length in words": lambda q: float(len(q.split())),
        "-length in words": lambda q: -float(len(q.split())),
        "has past-tense marker": lambda q: float(bool(PAST.search(q))),
        "NOT has present-tense marker": lambda q: float(not PRESENT.search(q)),
        "NOT has deixis (current/now)": lambda q: float(not DEIXIS.search(q)),
        "wh != 'who'": lambda q: float(wh_word(q) != "who"),
        "wh in {when,how,why}": lambda q: float(wh_word(q) in {"when", "how", "why"}),
    }
    for name, fn in feats.items():
        s = [fn(q) for q in teq]
        a = auroc(list(zip(s, [bool(y) for y in teY])))
        thr = sorted(set(s))
        best = max(weighted_f1(teY, [1 if v >= t else 0 for v in s]) for t in thr)
        print(f"   {name:30s} {a:7.4f} {best:7.4f}")

    print("\n## screen 2 -- bag-of-words logistic regression, fit on TRAIN only")
    vocab, w, b = fit_lexical_lr(trq, trY)
    sc = [lr_score(q, vocab, w, b) for q in teq]
    a = auroc(list(zip(sc, [bool(y) for y in teY])))
    f1 = weighted_f1(teY, [1 if v >= 0 else 0 for v in sc])
    print(f"   vocabulary {len(vocab)} unigrams, no embeddings, no pretraining")
    print(f"   AUROC {a:.4f}   weighted F1 {f1:.4f}")

    print("\n## screen 3 -- most predictive opening words (fit on TRAIN)")
    cnt = Counter()
    tot = Counter()
    for q, y in zip(trq, trY):
        fw = first_word(q)
        tot[fw] += 1
        cnt[fw] += y
    rows = [(fw, cnt[fw] / tot[fw], tot[fw]) for fw in tot if tot[fw] >= 25]
    rows.sort(key=lambda r: -r[1])
    print(f"   {'opening word':14s} {'P(evergreen)':>13s} {'n':>6s}")
    for fw, p, n in rows[:6]:
        print(f"   {fw:14s} {p:13.3f} {n:6d}")
    print("   ...")
    for fw, p, n in rows[-5:]:
        print(f"   {fw:14s} {p:13.3f} {n:6d}")

    print("\n## reference points from the paper (weighted F1, English test)")
    print("   random 0.637 | GPT-4.1 0.794 | best few-shot LLM 0.885 | EG-E5 0.913")


if __name__ == "__main__":
    main()
