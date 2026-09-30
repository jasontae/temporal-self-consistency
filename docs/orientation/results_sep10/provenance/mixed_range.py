"""Sep 2 section 4.3: "Extending the measurement to twelve models ... produced a
range from 0.5414 to 0.9183". Recompute the within-passage AUROC (the metric of
the 0.9183 figure, b11_surface_control.within_passage) for every mixed_*.jsonl,
with and without the Gemma-4 rows (dropped 2026-09-28).

Run from the repo root:
    python3 docs/orientation/results_sep10/provenance/mixed_range.py
"""
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.evaluation.b11_surface_control import within_passage  # noqa: E402

G4 = {"g4_31b", "g4_e4b", "gemma4_26b"}
res = {}
for f in sorted(glob.glob(str(ROOT / "data/prep/predictions_7b/mixed_*.jsonl"))):
    rows = [json.loads(l) for l in open(f)]
    res[Path(f).stem.replace("mixed_", "")] = within_passage(rows, lambda r: r["mean_logprob"])[0]
out = {"per_model_within_passage": res}
for name, keep in (("all", res), ("no_gemma4", {k: v for k, v in res.items() if k not in G4})):
    out[name] = {"n_models": len(keep), "min": min(keep.values()), "max": max(keep.values())}
json.dump(out, open(Path(__file__).with_suffix(".json"), "w"), indent=2)
print(json.dumps(out, indent=1))
