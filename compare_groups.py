"""
Compares colour/texture features across Healthy / Hypertension / Diabetes.
Unit of analysis = subject (left and right ear averaged).

Run after extract_features.py:
    python scripts/compare_groups.py

Needs: pip install statsmodels matplotlib
"""

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

GROUPS = ["healthy", "hypertension", "diabetes"]
POINTS = ["ShenMen", "Endocrine", "Pancreas", "Heart", "Kidney",
          "Sympathetic", "SuperiorTriangularFossa"]
FEATURES = ["a", "a_rel", "L", "saturation", "texture"]
SIDE = None          # None = average L and R; or "L" / "R"
ALPHA = 0.05

df = pd.read_csv(RES / "features.csv")
if SIDE:
    df = df[df["side"] == SIDE]
df = df[df["point"].isin(POINTS)]

# one value per subject, per point, per feature
subj = df.groupby(["group", "subject", "point"])[FEATURES].mean().reset_index()

# ---------- omnibus test: 3 groups ----------
rows = []
for p in POINTS:
    d = subj[subj["point"] == p]
    for f in FEATURES:
        samples = [d.loc[d["group"] == g, f].dropna().values for g in GROUPS]
        if min(len(s) for s in samples) < 3:
            continue
        H, p_kw = stats.kruskal(*samples)
        _, p_anova = stats.f_oneway(*samples)
        n = sum(len(s) for s in samples)
        row = {"point": p, "feature": f, "p_kruskal": p_kw, "p_anova": p_anova,
               "epsilon_sq": H / (n - 1)}
        for g, s in zip(GROUPS, samples):
            row[f"n_{g}"] = len(s)
            row[f"median_{g}"] = np.median(s)
        rows.append(row)

omni = pd.DataFrame(rows)
omni["q_fdr"] = multipletests(omni["p_kruskal"], method="fdr_bh")[1]
omni = omni.sort_values("p_kruskal").reset_index(drop=True)
omni.to_csv(RES / "stats_omnibus.csv", index=False)

print(f"{len(omni)} omnibus tests run. Top 10 by p-value:")
print(omni[["point", "feature", "p_kruskal", "q_fdr", "epsilon_sq"]].head(10).round(4).to_string())
print(f"\nSignificant after FDR (q < {ALPHA}): {(omni['q_fdr'] < ALPHA).sum()}")

# ---------- post-hoc pairwise (only where uncorrected omnibus p < 0.05) ----------
post = []
for _, r in omni[omni["p_kruskal"] < 0.05].iterrows():
    d = subj[subj["point"] == r["point"]]
    pairs, pvals = [], []
    for g1, g2 in combinations(GROUPS, 2):
        a = d.loc[d["group"] == g1, r["feature"]].dropna()
        b = d.loc[d["group"] == g2, r["feature"]].dropna()
        pvals.append(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        pairs.append(f"{g1} vs {g2}")
    p_holm = multipletests(pvals, method="holm")[1]
    for pair, pv, ph in zip(pairs, pvals, p_holm):
        post.append({"point": r["point"], "feature": r["feature"], "pair": pair,
                     "p_raw": pv, "p_holm": ph, "omnibus_q_fdr": r["q_fdr"]})

post = pd.DataFrame(post)
if len(post):
    post.to_csv(RES / "stats_posthoc.csv", index=False)
    print("\nPost-hoc (Mann-Whitney, Holm within each trio):")
    print(post.round(4).to_string(index=False))
else:
    print("\nNo omnibus result reached p < 0.05, so no post-hoc tests were run.")

# ---------- plot the top 6 ----------
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, (_, r) in zip(axes.ravel(), omni.head(6).iterrows()):
    d = subj[subj["point"] == r["point"]]
    data = [d.loc[d["group"] == g, r["feature"]].dropna().values for g in GROUPS]
    ax.boxplot(data, showfliers=False)
    for i, v in enumerate(data, 1):
        ax.scatter(np.random.normal(i, 0.05, len(v)), v, s=12, alpha=0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(GROUPS, fontsize=8)
    ax.set_title(f"{r['point']} - {r['feature']}\np={r['p_kruskal']:.3f}, q={r['q_fdr']:.3f}",
                 fontsize=9)
plt.tight_layout()
plt.savefig(RES / "top_features.png", dpi=200)
print(f"\nSaved tables and top_features.png in {RES}")