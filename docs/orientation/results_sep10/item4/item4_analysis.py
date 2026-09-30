"""Item 4: the fixed 4-bit ladder, with tokenizer, prompt and answer-length controls.

Inputs (from run_4bit_ladder.sh):
  disjoint_<m>.jsonl        Aug checkpoints (mixed quantization), disjoint claim set
  disjoint4b_<m>.jsonl      the same nine models at 4-bit
  disjoint4bchat_<m>.jsonl  4-bit, claims scored as an assistant turn (prompt control)

Per model and contrast (truth, regime, staleness; PMI):
  quantization   AUROC at mixed quantization vs 4-bit, entity-paired bootstrap on the change
  tokenizer      AUROC after regressing PMI on the claim's and the name's token counts
                 (that model's tokenizer), within model
  answer length  paired win rate restricted to entity pairs whose two names have the
                 same token count
  prompt         AUROC with the chat frame

and the inverse result (inverse_truth.py) is rerun on disjoint4b_ and disjoint4bchat_.

Run from the repo root:
    python3 docs/orientation/results_sep10/item4/item4_analysis.py
"""
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
P = ROOT / "data" / "prep" / "predictions_7b"
OUT = Path(__file__).resolve().parent
M = Path.home() / ".oMLX" / "models"
Q25 = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-7B-Instruct-4bit/snapshots/c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed"
TOK = {"g3_4b": M / "gemma-3-4b-it-4bit", "qwen3_4b_th": M / "Qwen3-4B-Thinking-2507-4bit",
       "qwen3_4b_it": M / "Qwen3-4B-Instruct-2507-4bit", "qwen25_7b": Q25, "tsct": Q25,
       "gptoss_20b": M / "gpt-oss-20b-MXFP4-Q4", "qwen36_35b": M / "Qwen3.6-35B-A3B-4bit",
       "qwen35_27b": M / "Qwen3.5-27B-4bit", "qwen36_27b_stock": M / "Qwen3.6-27B-4bit"}
REQUANT = {"qwen3_4b_th": "8-bit", "qwen3_4b_it": "8-bit", "gptoss_20b": "6.5-bit", "qwen36_35b": "nvfp4"}
CONTRASTS = {"truth": ("immutable_true", "immutable_false"), "regime": ("immutable_true", "volatile_current"),
             "staleness": ("volatile_current", "volatile_stale")}
SEED = 20260928


def load(path):
    by = defaultdict(dict)
    for r in map(json.loads, open(path)):
        by[r["entity_id"]][r["arm"]] = r
    return {e: a for e, a in by.items() if len(a) == 4}


def auroc(pos, neg):
    r = stats.rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    from transformers import AutoTokenizer
    rng = np.random.default_rng(SEED)
    res = {}
    for m, tokdir in TOK.items():
        f4 = P / f"disjoint4b_{m}.jsonl"
        if not f4.exists():
            res[m] = {"missing": True}
            continue
        a4 = load(f4)
        ents = sorted(a4)
        boots = rng.integers(0, len(ents), (4000, len(ents)))
        tok = AutoTokenizer.from_pretrained(str(tokdir))
        ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
        r = {}
        fa = P / f"disjoint_{m}.jsonl"
        fc = P / f"disjoint4bchat_{m}.jsonl"
        aa = load(fa) if fa.exists() else None
        ac = load(fc) if fc.exists() else None
        for c, (hi, lo) in CONTRASTS.items():
            x4h = np.array([a4[e][hi]["pmi"] for e in ents])
            x4l = np.array([a4[e][lo]["pmi"] for e in ents])
            row = {"auroc_4bit": float(auroc(x4h, x4l))}
            if aa is not None:
                xah = np.array([aa[e][hi]["pmi"] for e in ents])
                xal = np.array([aa[e][lo]["pmi"] for e in ents])
                row["auroc_aug"] = float(auroc(xah, xal))
                d = [auroc(x4h[b], x4l[b]) - auroc(xah[b], xal[b]) for b in boots]
                row["delta_4bit_minus_aug"] = row["auroc_4bit"] - row["auroc_aug"]
                row["delta_ci95"] = np.percentile(d, [2.5, 97.5]).tolist()
            # tokenizer control: residualize PMI on claim and name token counts
            feats, ys = [], []
            for e in ents:
                for arm in (hi, lo):
                    feats.append([1.0, ntok(a4[e][arm]["text"]), ntok(a4[e][arm]["name_only_text"])])
                    ys.append(a4[e][arm]["pmi"])
            X, y = np.array(feats), np.array(ys)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res_ = (y - X @ beta).reshape(len(ents), 2)
            row["auroc_token_residualized"] = float(auroc(res_[:, 0], res_[:, 1]))
            # answer-length control: pairs whose two names have equal token counts
            eq = [i for i, e in enumerate(ents) if ntok(a4[e][hi]["person"]) == ntok(a4[e][lo]["person"])]
            row["n_equal_name_length_pairs"] = len(eq)
            if len(eq) >= 10:
                row["paired_win_equal_length"] = float(np.mean([x4h[i] > x4l[i] for i in eq]))
                row["paired_win_all"] = float(np.mean(x4h > x4l))
            if ac is not None:
                row["auroc_chat_frame"] = float(auroc(np.array([ac[e][hi]["pmi"] for e in ents]),
                                                      np.array([ac[e][lo]["pmi"] for e in ents])))
            r[c] = row
        r["requantized_from"] = REQUANT.get(m, "already 4-bit affine g64 (reused)")
        res[m] = r
    json.dump(res, open(OUT / "item4_controls.json", "w"), indent=2)

    # inverse result under 4-bit and under the chat frame
    inv = {}
    for pre in ("disjoint4b_", "disjoint4bchat_", "matched4b_"):
        o = subprocess.run([sys.executable, str(ROOT / "docs/orientation/results_sep10/inverse/inverse_truth.py"),
                            "--prefix", pre], capture_output=True, text=True, cwd=ROOT)
        (OUT / f"inverse_{pre.rstrip('_')}.out").write_text(o.stdout + o.stderr)
        j = ROOT / f"docs/orientation/results_sep10/inverse/inverse_{pre.rstrip('_')}.json"
        if j.exists():
            d = json.load(open(j))
            inv[pre] = {k: d["model_level"][k] for k in ("ols_slope", "exact_permutation_p", "spearman_rho",
                                                          "slope_entity_bootstrap_ci")} | {
                "n": len(d["admitted"]), "lmm_paired_win_b1": d["lmm"]["paired win 1[T > V]"]["b1"],
                "lmm_paired_win_p": d["lmm"]["paired win 1[T > V]"]["p_t"],
                "split_half_frac_negative": d["split_half"]["frac_slope_negative"],
                "staleness_mean": d["staleness_mean_admitted"], "staleness_ci": d["staleness_mean_ci"]}
    json.dump(inv, open(OUT / "item4_inverse.json", "w"), indent=2)

    lines = ["| model | quant (Aug → 4-bit) | contrast | Aug | 4-bit | Δ [95% CI] | token-residualized | equal-name-length win (n) / all | chat frame |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, r in res.items():
        if r.get("missing"):
            lines.append(f"| {m} | missing | | | | | | | |")
            continue
        for c in CONTRASTS:
            x = r[c]
            dl = f"{x['delta_4bit_minus_aug']:+.3f} [{x['delta_ci95'][0]:+.3f}, {x['delta_ci95'][1]:+.3f}]" if "delta_ci95" in x else ""
            eqs = (f"{x['paired_win_equal_length']:.3f} ({x['n_equal_name_length_pairs']}) / {x['paired_win_all']:.3f}"
                   if "paired_win_equal_length" in x else f"n = {x['n_equal_name_length_pairs']}")
            lines.append(f"| {m} | {r['requantized_from']} | {c} | {x.get('auroc_aug', float('nan')):.3f} | {x['auroc_4bit']:.3f} | {dl} | "
                         f"{x['auroc_token_residualized']:.3f} | {eqs} | {x.get('auroc_chat_frame', float('nan')):.3f} |")
    lines += ["", "| inverse result | n | slope [entity-bootstrap CI] | permutation p | Spearman | LMM paired-win b1, p | split-half negative | staleness mean [CI] |",
              "|---|---|---|---|---|---|---|---|"]
    for k, v in inv.items():
        lines.append(f"| {k} | {v['n']} | {v['ols_slope']:.3f} [{v['slope_entity_bootstrap_ci'][0]:.2f}, {v['slope_entity_bootstrap_ci'][1]:.2f}] | "
                     f"{v['exact_permutation_p']:.4f} | {v['spearman_rho']:.3f} | {v['lmm_paired_win_b1']:.2f}, {v['lmm_paired_win_p']:.4f} | "
                     f"{100 * v['split_half_frac_negative']:.1f}% | {v['staleness_mean']:.3f} [{v['staleness_ci'][0]:.3f}, {v['staleness_ci'][1]:.3f}] |")
    (OUT / "item4_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
