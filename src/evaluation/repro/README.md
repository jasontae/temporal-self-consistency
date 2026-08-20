# Reproduction and validity checks

CPU-only. Each script answers one question about whether a result means what it
appears to mean.

| script | question |
|---|---|
| `seed_duplication_check.py` | How many *distinct* models does a prediction folder actually contain? Files sharing an md5 are one model reported many times. |
| `ece_strategy_probe.py` | Which constant hedging policy minimises ECE at a given accuracy? Shows the regime where refusal outscores a perfect oracle. |
| `verify_v6_seed_split.py` | Do the reported v6 numbers reproduce across the full seed set, or only on a subset? |

```bash
python3 seed_duplication_check.py --pred-dir /path/to/predictions
python3 ece_strategy_probe.py
python3 verify_v6_seed_split.py --pred-dir /path/to/predictions_v6
```

`seed_duplication_check.py` exists because a released prediction folder was
found to contain eight seed-labelled files that resolved to two distinct
checkpoints. Hash your checkpoints before reporting a standard deviation over
seeds.
