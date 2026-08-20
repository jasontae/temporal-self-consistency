import json,sys,os,glob,hashlib,numpy as np
from collections import Counter, defaultdict
TK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, TK)
from eval_pipeline import compute_ece, compute_accuracy_metrics, HEDGE_TO_CONFIDENCE, bonferroni_ece_comparison
from hedge_quality_rubric import GOLD_HEDGE_FOR_VOLATILITY
import argparse
_ap=argparse.ArgumentParser(description="Report how many DISTINCT models a prediction folder actually contains.")
_ap.add_argument("--pred-dir", required=True)
D=_ap.parse_args().pred_dir
HED=["[CONFIDENT]","[COND_CONFIDENT]","[TEMPORAL_HEDGE]","[UNKNOWN]"]
md5=lambda f: hashlib.md5(open(f,'rb').read()).hexdigest()

print("="*112)
print("DISTINCT MODELS IN predictions_v8  (temporal_delta; files sharing a hash are one model)")
print("="*112)
groups={}
for pfx,lab in (("exp2_v2","SFT"),("exp3_v7","TSCT")):
    h=defaultdict(list)
    for f in sorted(glob.glob(os.path.join(D,f"{pfx}_seed*_temporal_delta_predictions.jsonl"))):
        s=int(f.split("seed")[1].split("_")[0]); h[md5(f)].append((s,f))
    groups[lab]=sorted(h.values(), key=lambda v:-len(v))
    print(f"\n{lab}: {len(h)} distinct model(s) across {sum(len(v) for v in h.values())} seed files")
    print(f"{'  seeds':<34}{'ECE':>8}{'EM':>8}{'corr':>8}   hedge distribution %")
    for v in groups[lab]:
        R=[json.loads(l) for l in open(v[0][1])]; n=len(R)
        c=Counter(r["predicted_hedge"] for r in R)
        hd=" ".join(f"{k.strip('[]')[:4]}={100*x/n:5.1f}" for k,x in c.most_common())
        seeds=str(sorted(s for s,_ in v))
        print(f"  {seeds:<32}{compute_ece(R)['ece']:>8.4f}{compute_accuracy_metrics(R)['em']:>8.4f}"
              f"{sum(1 for r in R if r.get('correct'))/n:>8.4f}   {hd}")
R0=[json.loads(l) for l in open(groups['SFT'][0][0][1])]
g=Counter(GOLD_HEDGE_FOR_VOLATILITY.get(r.get("volatility")) for r in R0); gn=sum(g.values())
print(f"  {'GOLD target':<32}{'':>8}{'':>8}{'':>8}   "+" ".join(f"{k.strip('[]')[:4]}={100*v/gn:5.1f}" for k,v in g.most_common()))

print()
print("="*112)
print("SIGNIFICANCE — using one representative per DISTINCT model (temporal_delta)")
print("="*112)
def eces(lab):
    out=[]
    for v in groups[lab]:
        R=[json.loads(l) for l in open(v[0][1])]
        out.append(compute_ece(R)["ece"])
    return out
s,t=eces("SFT"),eces("TSCT")
print(f"  SFT  n={len(s)} distinct: {[round(x,4) for x in s]}")
print(f"  TSCT n={len(t)} distinct: {[round(x,4) for x in t]}")
if len(t)>=2 and len(s)>=2:
    c=bonferroni_ece_comparison(t,{"SFT":s})["comparisons"]["SFT"]
    print(f"\n  reduction={c['ece_reduction']:+.4f}  p={c['p_value']:.4f}  d={c['cohens_d']:+.3f}  sig={c['significant_bonferroni']}")
print("\n  NOTE: naive per-seed stats would report n=7 vs n=8 and understate the variance,")
print("  because most of those files are copies of the same checkpoint.")

print()
print("="*112); print("REFERENCE CEILINGS (v8 SFT seed 42)"); print("="*112)
base=[json.loads(l) for l in open(os.path.join(D,"exp2_v2_seed42_temporal_delta_predictions.jsonl"))]
print(f"  accuracy(correct) = {np.mean([bool(r.get('correct')) for r in base]):.4f}")
for h in HED:
    print(f"  constant {h:<18} ECE = {compute_ece([{**r,'predicted_hedge':h} for r in base])['ece']:.4f}")
orc=compute_ece([{**r,'predicted_hedge':GOLD_HEDGE_FOR_VOLATILITY.get(r.get('volatility'),'[TEMPORAL_HEDGE]')} for r in base])['ece']
unk=compute_ece([{**r,'predicted_hedge':'[UNKNOWN]'} for r in base])['ece']
print(f"  gold-hedge oracle          ECE = {orc:.4f}")
print(f"  --> constant [UNKNOWN] beats the oracle by {orc/unk:.1f}x")
