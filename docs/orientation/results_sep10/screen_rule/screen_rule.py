"""One decision rule for the form-validity screen, applied to all four sets.

Implements docs/orientation/PLAN_screen_rule.md (2026-09-30) exactly. CPU only;
reads existing per-item score files, runs no model.

Primary rule: the screen FIRES when the model's discrimination does not survive
conditioning on the strongest no-model feature, i.e. the cue-stratified AUROC
(pairs compared only within a cue stratum, pooled by within-stratum pair count)
has a 95% bootstrap CI that includes 0.5 (or excludes it on the opposite side).
Precondition: if the unconditional model AUROC CI includes 0.5 the verdict is
"no signal". If the cue separates the classes perfectly, the stratified AUROC is
undefined and the screen fires.

Run from the repo root:
    python3 docs/orientation/results_sep10/screen_rule/screen_rule.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.evaluation.b11_dedup import distinct_claims  # noqa: E402
from src.evaluation.b11_surface_control import has_person_name  # noqa: E402
from src.evaluation.evergreen_surface_screen import PAST  # noqa: E402

P = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
SEED = 20260930
NBOOT = 2000
TEMPORAL = re.compile(r"\b(current|currently|latest|most recent|now|this year|today|recent)\b", re.I)
PANEL = ["qwen3_4b_it", "qwen25_7b", "qwen35_27b", "qwen36_27b", "qwen36_35b"]
LABEL = {"qwen3_4b_it": "Qwen3-4B-Instruct", "qwen25_7b": "Qwen2.5-7B", "qwen35_27b": "Qwen3.5-27B",
         "qwen36_27b": "Qwen3.6-27B", "qwen36_35b": "Qwen3.6-35B-A3B"}
FEATURES = ["n_chars", "n_words", "has_digit", "name_len", "person_name", "past_tense", "temporal_cue"]
BINARY = {"has_digit", "person_name", "past_tense", "temporal_cue"}


# ------------------------------------------------------------------ features
def cap_run(text):
    """Longest run of consecutive capitalised words, excluding the first word."""
    best = cur = 0
    for w in text.split()[1:]:
        if re.match(r"[A-Z]", w):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def features(text, person=None):
    return {"n_chars": float(len(text)), "n_words": float(len(text.split())),
            "has_digit": float(bool(re.search(r"\d", text))),
            "name_len": float(len(person.split())) if person is not None else float(cap_run(text)),
            "person_name": float(has_person_name(text)), "past_tense": float(bool(PAST.search(text))),
            "temporal_cue": float(bool(TEMPORAL.search(text)))}


def strata(values, binary):
    v = np.asarray(values, float)
    if binary:
        return (v > 0.5).astype(int)
    edges = np.unique(np.quantile(v, [0.2, 0.4, 0.6, 0.8]))
    return np.digitize(v, edges, right=True)


# ------------------------------------------------------------------ AUROC
def u_pairs(pos, neg):
    """(wins + 0.5 ties, n pairs) for pos > neg."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return 0.0, 0
    r = rankdata(np.concatenate([pos, neg]))
    return float(r[:n1].sum() - n1 * (n1 + 1) / 2), n1 * n0


def auroc(pos, neg):
    u, n = u_pairs(pos, neg)
    return u / n if n else None


def strat_auroc(pos, neg, spos, sneg):
    U = N = 0
    for s in np.union1d(spos, sneg):
        u, n = u_pairs(pos[spos == s], neg[sneg == s])
        U += u
        N += n
    return (U / N if N else None), N


def dirfree(a):
    return None if a is None else max(a, 1 - a)


# ------------------------------------------------------------------ set loaders
# Each loader returns a list of "contrasts"; each contrast has
#   pos_items / neg_items: list of item keys; feats[key] -> dict; scores[model][key] -> float
#   cluster: True if pos_items[i] and neg_items[i] belong to one cluster (entity bootstrap)
def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def set_stress():
    files = {"qwen3_4b_it": "mixed_qwen3_4b_it", "qwen25_7b": "mixed_qwen25_7b",
             "qwen36_27b": "mixed_qwen36_27b", "qwen36_35b": "mixed_qwen36_35b"}
    scores, ref = {}, None
    for m, stem in files.items():
        u = distinct_claims(load_jsonl(P / f"{stem}.jsonl"))
        scores[m] = {r["text"]: r["mean_logprob"] for r in u}
        ref = ref or u
    pos = [r["text"] for r in ref if not r["is_volatile"]]
    neg = [r["text"] for r in ref if r["is_volatile"]]
    feats = {r["text"]: features(r["text"]) for r in ref}
    return [{"set": "stress", "contrast": "stable vs volatile", "pos_items": pos, "neg_items": neg,
             "feats": feats, "scores": scores, "cluster": False,
             "missing_panel": ["qwen35_27b (never scored on this set)"]}]


def set_evergreen():
    files = {"qwen3_4b_it": "eg_ao_qwen3_4b_it", "qwen25_7b": "eg_qwen25_7b", "qwen35_27b": "eg_ao_qwen35_27b",
             "qwen36_27b": "eg_ao_qwen36_27b", "qwen36_35b": "eg_ao_qwen36_35b"}
    scores, q, lab = {}, {}, {}
    for m, stem in files.items():
        rows = [r for r in load_jsonl(P / f"{stem}.jsonl") if r.get("ppl") and r.get("entropy") is not None]
        scores[m] = {r["idx"]: -r["ppl"] for r in rows}
        for r in rows:
            q[r["idx"]], lab[r["idx"]] = r["question"], bool(r["is_evergreen"])
    keys = sorted(set.intersection(*(set(s) for s in scores.values())))
    return [{"set": "evergreen", "contrast": "evergreen vs mutable",
             "pos_items": [k for k in keys if lab[k]], "neg_items": [k for k in keys if not lab[k]],
             "feats": {k: features(q[k]) for k in keys}, "scores": scores, "cluster": False, "missing_panel": []}]


def set_freshqa():
    files = {"qwen3_4b_it": "fq_qwen3_4b_it", "qwen25_7b": "fq_qwen25_7b", "qwen35_27b": "fq_qwen35_27b",
             "qwen36_27b": "fq_qwen36_27b_stock", "qwen36_35b": "fq_qwen36_35b"}
    built = {r["id"]: r for r in load_jsonl(ROOT / "data/prep/freshqa/freshqa_built.jsonl")}
    scores, alt = {}, {}
    for m, stem in files.items():
        by = defaultdict(dict)
        for r in load_jsonl(P / f"{stem}.jsonl"):
            by[r["entity_id"]][r["arm"]] = r
        s, a = {}, {}
        for e, arms in by.items():
            ft = built[e]["fact_type"] if e in built else None
            r = arms.get("fq_current") if ft == "fast-changing" else arms.get("fq_true") if ft == "never-changing" else None
            if r is not None:
                s[e], a[e] = r["pmi"], r["mean_logprob"]
        scores[m], alt[m] = s, a
    keys = sorted(set.intersection(*(set(s) for s in scores.values())))
    pos = [k for k in keys if built[k]["fact_type"] == "never-changing"]
    neg = [k for k in keys if built[k]["fact_type"] == "fast-changing"]
    # reference: the paper's full-set screen (all fast vs all never questions)
    allf = [r for r in built.values() if r["fact_type"] == "fast-changing"]
    alln = [r for r in built.values() if r["fact_type"] == "never-changing"]
    ref = {f: dirfree(auroc(np.array([features(r["question"])[f] for r in alln]),
                            np.array([features(r["question"])[f] for r in allf]))) for f in FEATURES}
    return [{"set": "freshqa", "contrast": "never-changing vs fast-changing (matched subset)",
             "pos_items": pos, "neg_items": neg, "feats": {k: features(built[k]["question"]) for k in keys},
             "scores": scores, "alt_scores_mean_logprob": alt, "cluster": False, "missing_panel": [],
             "full_set_feature_auroc": ref, "full_set_n": {"never": len(alln), "fast": len(allf)}}]


def set_matched():
    files = {"qwen3_4b_it": "disjoint4b_qwen3_4b_it", "qwen25_7b": "disjoint4b_qwen25_7b",
             "qwen35_27b": "disjoint4b_qwen35_27b", "qwen36_27b": "disjoint4b_qwen36_27b_stock",
             "qwen36_35b": "disjoint4b_qwen36_35b"}
    scores, rows = {}, {}
    for m, stem in files.items():
        scores[m] = {}
        for r in load_jsonl(P / f"{stem}.jsonl"):
            k = (r["entity_id"], r["arm"])
            scores[m][k] = r["pmi"]
            rows[k] = r
    ents = sorted({e for e, _ in rows if all((e, a) in rows for a in
                  ("volatile_current", "volatile_stale", "immutable_true", "immutable_false"))})
    feats = {k: features(r["text"], person=r["person"]) for k, r in rows.items()}
    out = []
    for name, hi, lo in (("staleness", "volatile_current", "volatile_stale"),
                         ("truth", "immutable_true", "immutable_false"),
                         ("regime", "immutable_true", "volatile_current")):
        out.append({"set": "matched", "contrast": name, "pos_items": [(e, hi) for e in ents],
                    "neg_items": [(e, lo) for e in ents], "feats": feats, "scores": scores,
                    "cluster": True, "missing_panel": [], "claimed_direction": -1 if name == "staleness" else 1})
    return out


# ------------------------------------------------------------------ analysis
def verdict(unc_ci, unc_pt, st_ci, st_pt):
    if unc_ci[0] <= 0.5 <= unc_ci[1]:
        return "no signal"
    if st_pt is None or st_ci is None:
        return "fires (cue separates classes perfectly)"
    side = 1 if unc_pt > 0.5 else -1
    if side == 1 and st_ci[0] > 0.5:
        return "passes"
    if side == -1 and st_ci[1] < 0.5:
        return "passes"
    return "fires"


def pct(x):
    x = [v for v in x if v is not None]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) >= 100 else None


def analyse(c, rng):
    models = [m for m in PANEL if m in c["scores"]]
    pos_k, neg_k = c["pos_items"], c["neg_items"]
    n1, n0 = len(pos_k), len(neg_k)
    allk = pos_k + neg_k
    F = {f: np.array([c["feats"][k][f] for k in allk]) for f in FEATURES}
    S = {m: np.array([c["scores"][m][k] for k in allk]) for m in models}
    ip, ineg = np.arange(n1), np.arange(n1, n1 + n0)

    feat_auc = {f: dirfree(auroc(F[f][ip], F[f][ineg])) for f in FEATURES}
    feat_dir = {f: (1 if (auroc(F[f][ip], F[f][ineg]) or 0.5) >= 0.5 else -1) for f in FEATURES}
    nonconst = [f for f in FEATURES if len(np.unique(F[f])) > 1]
    strongest = max(nonconst, key=lambda f: (feat_auc[f], -FEATURES.index(f)))
    ST = {f: strata(F[f], f in BINARY) for f in nonconst}

    # bootstrap draws: index arrays into ip / ineg
    draws = []
    for _ in range(NBOOT):
        if c["cluster"]:
            j = rng.integers(0, n1, n1)
            draws.append((ip[j], ineg[j]))
        else:
            draws.append((ip[rng.integers(0, n1, n1)], ineg[rng.integers(0, n0, n0)]))

    def stat_unc(m, a, b):
        return auroc(S[m][a], S[m][b])

    def stat_st(m, f, a, b):
        return strat_auroc(S[m][a], S[m][b], ST[f][a], ST[f][b])[0]

    cd = c.get("claimed_direction", 1)

    def m_or(x):
        return x if cd == 1 else 1 - x

    def cue_or(f, a, b):
        x = auroc(F[f][a], F[f][b])
        return x if feat_dir[f] == 1 else 1 - x

    res = {"set": c["set"], "contrast": c["contrast"], "claimed_direction": c.get("claimed_direction", 1), "n_pos": n1, "n_neg": n0,
           "unit": "entity (cluster bootstrap)" if c["cluster"] else "item (within-class bootstrap)",
           "models": {m: LABEL[m] for m in models}, "missing_panel": c["missing_panel"],
           "feature_auroc_dirfree": feat_auc, "feature_direction": feat_dir,
           "constant_features": [f for f in FEATURES if f not in nonconst],
           "strongest_feature": strongest, "by_feature": {}}
    if "full_set_feature_auroc" in c:
        res["full_set_feature_auroc_reference"] = c["full_set_feature_auroc"]
        res["full_set_n"] = c["full_set_n"]

    # unconditional model AUROC (+ panel mean), bootstrapped once
    unc_b = {m: np.array([stat_unc(m, a, b) for a, b in draws]) for m in models}
    unc_pt = {m: stat_unc(m, ip, ineg) for m in models}
    unc_mean_b = np.mean([unc_b[m] for m in models], axis=0)
    res["model_unconditional"] = {m: {"auroc": unc_pt[m], "ci95": pct(unc_b[m])} for m in models}
    res["model_unconditional"]["panel_mean"] = {"auroc": float(np.mean(list(unc_pt.values()))),
                                                "ci95": pct(unc_mean_b)}
    if "alt_scores_mean_logprob" in c:
        A = c["alt_scores_mean_logprob"]
        res["model_unconditional_mean_logprob_sensitivity"] = {
            m: auroc(np.array([A[m][k] for k in pos_k]), np.array([A[m][k] for k in neg_k])) for m in models}

    for f in nonconst:
        st_pt, npairs = {}, {}
        st_b = {}
        for m in models:
            st_pt[m], npairs[m] = strat_auroc(S[m][ip], S[m][ineg], ST[f][ip], ST[f][ineg])
            st_b[m] = [stat_st(m, f, a, b) for a, b in draws]
        dropped = sum(x is None for x in st_b[models[0]])
        valid = [i for i in range(NBOOT) if all(st_b[m][i] is not None for m in models)]
        st_mean_b = [float(np.mean([st_b[m][i] for m in models])) for i in valid]
        st_mean_pt = None if any(st_pt[m] is None for m in models) else float(np.mean(list(st_pt.values())))
        cue_b = np.array([cue_or(f, a, b) for a, b in draws])
        cue_pt = cue_or(f, ip, ineg)
        per = {}
        for m in models:
            ci = pct(st_b[m])
            d = cue_b - m_or(unc_b[m])
            per[m] = {"stratified_auroc": st_pt[m], "stratified_ci95": ci, "within_stratum_pairs": npairs[m],
                      "verdict": verdict(res["model_unconditional"][m]["ci95"], unc_pt[m], ci, st_pt[m]),
                      "cue_minus_model": cue_pt - m_or(unc_pt[m]), "cue_minus_model_ci95": pct(d),
                      "cue_minus_model_p": float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean())))}
        mci = pct(st_mean_b)
        dm = cue_b - m_or(unc_mean_b)
        pm = {"stratified_auroc": st_mean_pt, "stratified_ci95": mci,
              "verdict": verdict(res["model_unconditional"]["panel_mean"]["ci95"],
                                 res["model_unconditional"]["panel_mean"]["auroc"], mci, st_mean_pt),
              "cue_minus_model": cue_pt - m_or(res["model_unconditional"]["panel_mean"]["auroc"]),
              "cue_minus_model_ci95": pct(dm),
              "cue_minus_model_p": float(min(1, 2 * min((dm <= 0).mean(), (dm >= 0).mean())))}
        per_stratum = []
        for st in np.unique(ST[f]):
            a_, b_ = ip[ST[f][ip] == st], ineg[ST[f][ineg] == st]
            per_stratum.append({"stratum": int(st), "n_pos": int(len(a_)), "n_neg": int(len(b_)),
                                "model_auroc": {m: auroc(S[m][a_], S[m][b_]) for m in models},
                                "mean_score_pos": {m: float(S[m][a_].mean()) if len(a_) else None for m in models},
                                "mean_score_neg": {m: float(S[m][b_].mean()) if len(b_) else None for m in models}})
        res["by_feature"][f] = {"per_stratum": per_stratum, "cue_auroc_oriented": cue_pt, "cue_ci95": pct(cue_b),
                                "n_strata": int(len(np.unique(ST[f]))), "boot_draws_dropped": int(dropped),
                                "per_model": per, "panel_mean": pm,
                                "models_fired": sum(per[m]["verdict"].startswith("fires") for m in models),
                                "n_models": len(models)}
    res["primary"] = res["by_feature"][strongest]
    return res


def fmt(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def ci(c):
    return "—" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"


def main():
    rng = np.random.default_rng(SEED)
    contrasts = set_stress() + set_evergreen() + set_freshqa() + set_matched()
    results = [analyse(c, rng) for c in contrasts]
    # sensitivity (not in the plan): FreshQA scored by mean_logprob instead of pmi
    fqc = dict(next(c for c in contrasts if c["set"] == "freshqa"))
    fqc["scores"], fqc["contrast"] = fqc.pop("alt_scores_mean_logprob"), "never vs fast, mean_logprob (sensitivity)"
    fqc.pop("full_set_feature_auroc")
    results.append(analyse(fqc, np.random.default_rng(SEED)))
    json.dump(results, open(OUT / "screen_rule.json", "w"), indent=2)

    L = ["# Screen rule: one decision rule, four sets", "",
         "Generated by `screen_rule.py` (plan: `../../PLAN_screen_rule.md`). AUROC oriented so higher score predicts the positive class; 95% bootstrap CIs, 2,000 draws.", ""]
    L += ["## 1. Verdicts under the primary rule (panel mean, conditioning on the strongest feature)", "",
          "| set | contrast | n+ / n− | strongest feature (AUROC) | model AUROC, panel mean | cue-stratified, panel mean | verdict | models fired | cue − model [CI], p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        p = r["primary"]
        mu = r["model_unconditional"]["panel_mean"]
        L.append(f"| {r['set']} | {r['contrast']} | {r['n_pos']} / {r['n_neg']} | {r['strongest_feature']} ({fmt(r['feature_auroc_dirfree'][r['strongest_feature']])}) "
                 f"| {fmt(mu['auroc'])} {ci(mu['ci95'])} | {fmt(p['panel_mean']['stratified_auroc'])} {ci(p['panel_mean']['stratified_ci95'])} "
                 f"| **{p['panel_mean']['verdict']}** | {p['models_fired']}/{p['n_models']} "
                 f"| {p['panel_mean']['cue_minus_model']:+.3f} {ci(p['panel_mean']['cue_minus_model_ci95'])}, {p['panel_mean']['cue_minus_model_p']:.3f} |")
    L += ["", "## 2. Per model, strongest feature", ""]
    for r in results:
        p = r["primary"]
        L += [f"### {r['set']}: {r['contrast']} (conditioning on `{r['strongest_feature']}`, {p['n_strata']} strata; unit: {r['unit']})", ""]
        if r["missing_panel"]:
            L += [f"Missing from panel: {', '.join(r['missing_panel'])}.", ""]
        L += ["| model | model AUROC [CI] | cue-stratified [CI] | within-stratum pairs | verdict | cue − model [CI], p |", "|---|---|---|---|---|---|"]
        for m, lab in r["models"].items():
            u, q = r["model_unconditional"][m], p["per_model"][m]
            L.append(f"| {lab} | {fmt(u['auroc'])} {ci(u['ci95'])} | {fmt(q['stratified_auroc'])} {ci(q['stratified_ci95'])} | {q['within_stratum_pairs']} "
                     f"| {q['verdict']} | {q['cue_minus_model']:+.3f} {ci(q['cue_minus_model_ci95'])}, {q['cue_minus_model_p']:.3f} |")
        L.append("")
    L += ["## 3. Robustness: the rule with every non-constant feature (panel mean)", "",
          "| set | contrast | feature | feature AUROC | cue-stratified, panel mean [CI] | verdict | models fired |", "|---|---|---|---|---|---|---|"]
    for r in results:
        for f, b in r["by_feature"].items():
            L.append(f"| {r['set']} | {r['contrast']} | {f}{' (strongest)' if f == r['strongest_feature'] else ''} | {fmt(r['feature_auroc_dirfree'][f])} "
                     f"| {fmt(b['panel_mean']['stratified_auroc'])} {ci(b['panel_mean']['stratified_ci95'])} | {b['panel_mean']['verdict']} | {b['models_fired']}/{b['n_models']} |")
    fq = results[2]
    L += ["", "## 4a. Per-stratum breakdown, strongest feature (panel models)", ""]
    for r in results:
        if r["set"] not in ("evergreen", "freshqa"):
            continue
        L += [f"**{r['set']}: {r['contrast']}**, stratified on `{r['strongest_feature']}`", "",
              "| stratum | n+ | n− | " + " | ".join(r["models"].values()) + " |", "|---|---|---|" + "---|" * len(r["models"])]
        for q in r["primary"]["per_stratum"]:
            L.append(f"| {q['stratum']} | {q['n_pos']} | {q['n_neg']} | " + " | ".join(fmt(q["model_auroc"][m]) for m in r["models"]) + " |")
        L.append("")
    L += ["## 4. FreshQA: full set vs matched subset (feature AUROC, direction-free)", "",
          f"Full set: {fq['full_set_n']['never']} never vs {fq['full_set_n']['fast']} fast questions. Matched subset: {fq['n_pos']} never vs {fq['n_neg']} fast (questions scored by every panel model).", "",
          "| feature | full set | matched subset |", "|---|---|---|"]
    for f in FEATURES:
        L.append(f"| {f} | {fmt(fq['full_set_feature_auroc_reference'][f])} | {fmt(fq['feature_auroc_dirfree'][f])} |")
    L += ["", "FreshQA model AUROC with `mean_logprob` instead of `pmi` (sensitivity): " +
          ", ".join(f"{LABEL[m]} {fmt(v)}" for m, v in fq["model_unconditional_mean_logprob_sensitivity"].items()), ""]
    (OUT / "screen_rule.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
