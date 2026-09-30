"""Apply PLAN_knowledge_probe.md's decision rule to knowledge_results.json (QA probe = primary)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.load(open(HERE / "knowledge_results.json"))
lines = []
for field in ("qa_answer", "frame_completion"):
    g = r[field]["groups"]
    cur = g.get("current")
    models = [m for m, c in r[field]["per_model_counts"].items() if c]
    if not cur or cur["n_pairs"] < 30:
        v = f"INCONCLUSIVE (current group {cur['n_pairs'] if cur else 0} pairs < 30)"
    elif cur["win_ci"][0] >= 0.5 or (cur["win_ci"][0] < 0.5 <= cur["win_ci"][1] and cur["win_rate"] >= 0.5):
        v = "H-GAP (where a model names the current holder, it does not prefer the expired one)"
    elif cur["win_ci"][1] < 0.5:
        v = "H-PERSIST (the preference survives knowing the current holder)"
    else:
        v = "DIRECTION ONLY (point estimate < 0.5, CI includes 0.5)"
    lines.append(f"## {field} ({'primary' if field == 'qa_answer' else 'secondary'}): {v}")
    lines.append(f"models: {len(models)} ({', '.join(models)})")
    for k, x in g.items():
        lines.append(f"- {k}: {x['n_pairs']} pairs, {x['n_entities']} entities, win {x['win_rate']:.3f} "
                     f"[{x['win_ci'][0]:.3f}, {x['win_ci'][1]:.3f}], z {x['z_diff']:+.3f} [{x['z_ci'][0]:+.3f}, {x['z_ci'][1]:+.3f}]")
    lines.append("")
(HERE / "VERDICT.md").write_text("# Knowledge probe: verdict under the plan's decision rule\n\n" + "\n".join(lines))
print("\n".join(lines))
