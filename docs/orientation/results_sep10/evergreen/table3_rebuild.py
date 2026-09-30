"""Table 3 / Appendix A rebuilt on answers instead of reasoning preambles.

The Aug evergreen runs (eg_*.jsonl) scored 32 greedy tokens after the chat
prompt. For gpt-oss, Qwen3.5-27B, Qwen3.6-27B and Qwen3.6-35B, 99-100% of those
tokens were a reasoning preamble ("Thinking Process:", "analysis..."), so the
perplexity and entropy measured the preamble (results_sep10/MODELS.md). The
answer-only rerun (eg_ao_*.jsonl, --answer-only, same 400 questions) starts each
model on its answer. Qwen2.5-7B and Qwen3-4B-Instruct produced answers in the
Aug run; Qwen3-4B's rerun is identical (a control), and Qwen2.5-7B keeps its Aug
row (its base checkpoint was not on disk for the rerun).

Reports, for the Aug table and the rebuilt one: per-model r(PPL), pooled AUROC,
form-stratified AUROC; the "form stratification removes about a third of the
above-chance excess" statement; the <=10B vs >=20B size contrast; the vintage-pair
range. The rebuilt table is shown with and without gpt-oss, because suppressing
gpt-oss's analysis channel runs it outside its intended mode.

Run from the repo root:
    python3 docs/orientation/results_sep10/evergreen/table3_rebuild.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.evaluation.analyze_evergreen_uncertainty import SIZE, pearson, within_form  # noqa: E402
from src.evaluation.b11_surface_control import auroc  # noqa: E402

P = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
LABEL = {"qwen3_4b_it": "Qwen3-4B-Instruct", "qwen25_7b": "Qwen2.5-7B", "gptoss_20b": "gpt-oss-20B",
         "qwen35_27b": "Qwen3.5-27B", "qwen36_27b": "Qwen3.6-27B", "qwen38_27b": "Qwen3.8-27B",
         "qwen36_35b": "Qwen3.6-35B-A3B"}


def row(path):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("ppl") and r.get("entropy") is not None]
    y = [float(r["is_evergreen"]) for r in rows]
    return {"n": len(rows), "r_ppl": pearson([-r["ppl"] for r in rows], y),
            "auroc": auroc([(-r["ppl"], bool(r["is_evergreen"])) for r in rows]),
            "within": within_form(rows, lambda r: -r["ppl"])[0], "file": path.name}


def summary(tab, pair=("qwen35_27b", "qwen36_27b")):
    ks = list(tab)
    pooled = sum(tab[k]["auroc"] for k in ks) / len(ks)
    within = sum(tab[k]["within"] for k in ks) / len(ks)
    small = [tab[k]["within"] for k in ks if SIZE[k] <= 10]
    large = [tab[k]["within"] for k in ks if SIZE[k] >= 20]
    out = {"models": ks, "mean_pooled_auroc": pooled, "mean_within_auroc": within,
           "share_of_excess_removed_by_form": (pooled - within) / (pooled - 0.5) if pooled > 0.5 else None,
           "r_ppl_range": [min(tab[k]["r_ppl"] for k in ks), max(tab[k]["r_ppl"] for k in ks)],
           "within_small_mean": sum(small) / len(small), "within_large_mean": sum(large) / len(large)}
    out["size_difference_large_minus_small"] = out["within_large_mean"] - out["within_small_mean"]
    if all(p in tab for p in pair):
        out["vintage_pair_range"] = abs(tab[pair[0]]["within"] - tab[pair[1]]["within"])
    return out


def main():
    aug = {k: row(P / f"eg_{k}.jsonl") for k in LABEL if (P / f"eg_{k}.jsonl").exists()}
    new = {}
    for k in LABEL:
        f = P / f"eg_ao_{k}.jsonl"
        if f.exists():
            new[k] = row(f)
        elif k == "qwen25_7b":
            new[k] = aug[k]  # answered in the Aug run; no preamble to remove
    six = [k for k in aug]  # the Sep 2 table's six models
    res = {"aug_table": aug, "rebuilt_table": new,
           "aug_summary": summary({k: aug[k] for k in six}),
           "rebuilt_summary_same_six": summary({k: new[k] for k in six}),
           "rebuilt_summary_six_without_gptoss": summary({k: new[k] for k in six if k != "gptoss_20b"}),
           "rebuilt_summary_with_qwen38": summary({k: new[k] for k in new})}
    json.dump(res, open(OUT / "table3_rebuild.json", "w"), indent=2)

    lines = ["| model | Aug r(PPL) | Aug pooled | Aug within-form | rebuilt r(PPL) | rebuilt pooled | rebuilt within-form | rebuilt source |",
             "|---|---|---|---|---|---|---|---|"]
    for k in LABEL:
        if k not in new:
            continue
        a = aug.get(k)
        n = new[k]
        af = f"{a['r_ppl']:+.3f} | {a['auroc']:.4f} | {a['within']:.4f}" if a else "— | — | —"
        lines.append(f"| {LABEL[k]} | {af} | {n['r_ppl']:+.3f} | {n['auroc']:.4f} | {n['within']:.4f} | {n['file']} |")
    lines += ["", "| summary | Aug (Sep 2) | rebuilt, same six | rebuilt, without gpt-oss | rebuilt, + Qwen3.8 |", "|---|---|---|---|---|"]
    S = [res["aug_summary"], res["rebuilt_summary_same_six"], res["rebuilt_summary_six_without_gptoss"],
         res["rebuilt_summary_with_qwen38"]]
    for key, fmt in (("mean_pooled_auroc", "{:.4f}"), ("mean_within_auroc", "{:.4f}"),
                     ("share_of_excess_removed_by_form", "{:.2f}"), ("within_small_mean", "{:.4f}"),
                     ("within_large_mean", "{:.4f}"), ("size_difference_large_minus_small", "{:+.4f}"),
                     ("vintage_pair_range", "{:.4f}")):
        lines.append(f"| {key} | " + " | ".join(fmt.format(s[key]) if s.get(key) is not None else "—" for s in S) + " |")
    lines.append("| r(PPL) range | " + " | ".join(f"{s['r_ppl_range'][0]:+.3f} to {s['r_ppl_range'][1]:+.3f}" for s in S) + " |")
    (OUT / "table3_rebuild.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
