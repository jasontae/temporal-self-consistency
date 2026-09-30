"""Appendix M: the pre-registered frequency test (null).

Per entity (n = 106): x = ln((c_expired + 0.5) / (c_current + 0.5)), the log
ratio of expired to current entity-holder co-occurrence in Dolma v1.7 (form
F1), against the expired-value preference (PMI stale - current, z-scored
within model, mean of the nine models). Line: the primary slope.

Source: results_sep10/frequency/frequency_results.json (analyze_frequency.py).

    python3 figures/paper/scripts/fig_frequency.py
"""
import json

import numpy as np
from matplotlib import pyplot as plt

from style import C, COL, RES, SIZE, apply, save, signed


def main():
    apply()
    d = json.load(open(RES / "frequency" / "frequency_results.json"))
    P = d["primary"]
    rows = [r for r in d["per_entity"] if r["x"] is not None]
    x = np.array([r["x"] for r in rows])
    y = np.array([r["ybar"] for r in rows])
    assert len(rows) == P["n_entities"]
    # the primary slope is OLS of ybar on x; check it before drawing it
    b_ols = np.polyfit(x, y, 1)[0]
    assert abs(b_ols - P["slope"]) < 1e-9, (b_ols, P["slope"])

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    ax.axhline(0, color=C["light"], lw=0.6)
    ax.axvline(0, color=C["light"], lw=0.6)
    ax.plot(x, y, "o", color=C["ink"], ms=2.4, alpha=0.75, mew=0)
    g = np.linspace(x.min(), x.max(), 20)
    ax.plot(g, y.mean() + P["slope"] * (g - x.mean()), color=C["constant"], lw=1.1)
    ax.set_title(f"primary slope {signed(P['slope'], '+.3f')}, permutation p = {P['perm_p_two_sided']:.2f}",
                 fontsize=SIZE["small"], color=C["constant_text"])
    ax.set_xlabel("log co-occurrence ratio, expired / current (Dolma)")
    ax.set_ylabel("expired-value preference (z)")
    print("\n".join(map(str, save(fig, "figM_frequency"))))


if __name__ == "__main__":
    main()
