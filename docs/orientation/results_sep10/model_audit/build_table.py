"""Collect runs/*.json into models_table.md (so MODELS.md numbers are not hand-copied)."""
import json
from pathlib import Path

R = Path(__file__).parent / "runs"
ORDER = ["g3_4b", "g4_e4b", "qwen3_4b_th", "qwen3_4b_it", "gptoss_20b", "qwen36_35b", "qwen35_27b",
         "qwen36_27b", "qwen38_27b", "g4_31b", "gemma4_26b", "gemma4_26b_vlm"]
lines = ["| tag | repo id | revision | converter | quant | loader | BOS prepended | known-answer raw / PMI (n=36) | same, no BOS | NLL/token (T=1) | best T | generation 5/5 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in ORDER:
    p = R / f"{t}.json"
    if not p.exists():
        continue
    d = json.load(open(p))
    m = d["meta"]
    if not m.get("repo_id") and m.get("path"):
        readme = Path(m["path"]) / "README.md"
        txt = readme.read_text() if readme.exists() else ""
        if "LM Studio" in txt:  # LM Studio community conversions name the repo after the dir
            m["repo_id"] = "lmstudio-community/" + Path(m["path"]).name
            m["converter"] = "LM Studio (" + ("mlx_vlm" if "using [mlx_vlm]" in txt else "mlx_lm") + ")"
        else:
            m["repo_id"] = "unverified: " + Path(m["path"]).name
    q = m.get("quantization", {})
    qs = f"{q.get('bits')}-bit {q.get('mode') or 'affine'} g{q.get('group_size')}"
    nb = d.get("known_answer_no_bos")
    nbs = f"{nb['acc_raw']:.2f} / {nb['acc_pmi']:.2f}" if nb else "n/a (no BOS)"
    tp = R / f"temperature_{t}.json"
    bt = json.load(open(tp))["best_T"] if tp.exists() else ""
    gen = sum(g.get("correct", False) for g in d.get("generation", []))
    lines.append(f"| {t} | {m.get('repo_id')} | {str(m.get('revision_sha'))[:10]} | {m.get('converter')} | {qs} | "
                 f"{d['loader']} | {d['tokenizer']['bos_id_scorer_uses']} | "
                 f"{d['known_answer']['acc_raw']:.2f} / {d['known_answer']['acc_pmi']:.2f} | {nbs} | "
                 f"{d['nll_paragraph']:.2f} | {bt} | {gen} |")
(Path(__file__).parent / "models_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
