"""Figures 6-8 of the revised manuscript: single 75/25 split (test_size=0.25, random_state=42).

Fig. 6 measured vs predicted E_total and Fig. 7 residuals use the v1.0.0 column order (spatial predictors
first), which reproduces the headline metrics R2 = 0.806, MAE = 0.231 V/m, RMSE = 0.321 V/m (see 01_validation.py).
Fig. 8 shows the Table 2 importances (results/table2_feature_importance.csv, band-first order).
Run 01_validation.py first.
"""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from common import BANDS, SP, FIGURES, RESULTS, load_dataset, metrics

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
d = load_dataset(); y = d.E_total.values
FEAT_V1 = SP + BANDS
tr, te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42)
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_depth=None, random_state=42, n_jobs=-1)
p = rf.fit(d.loc[tr, FEAT_V1].values, y[tr]).predict(d.loc[te, FEAT_V1].values)
m = metrics(y[te], p)

# Fig. 6
fig, ax = plt.subplots(figsize=(3.65, 3.55))
ax.scatter(y[te], p, s=32, color="#2f7bb6", edgecolors="#1a3f5c", linewidths=0.6, alpha=0.9, zorder=3)
lo, hi = 2.7, 6.1
ax.plot([lo, hi], [lo, hi], "r--", lw=1.4, zorder=2)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Measured $E_{total}$ (V/m)"); ax.set_ylabel("Predicted $E_{total}$ (V/m)")
ax.text(0.04, 0.96, f"R² = {m['R2']:.3f}\nMAE = {m['MAE']:.3f} V/m\nRMSE = {m['RMSE']:.3f} V/m", transform=ax.transAxes,
        va="top", fontsize=9.5, bbox=dict(boxstyle="round", fc="white", ec="0.5"))
fig.tight_layout(); fig.savefig(FIGURES / "Figure_06_Measured_vs_Predicted.png", dpi=300); plt.close(fig)

# Fig. 7
res = y[te] - p
fig, ax = plt.subplots(figsize=(3.6, 2.55))
ax.hist(res, bins=12, color="#1f77b4", edgecolor="black", linewidth=0.7)
ax.axvline(0, color="red", ls="--", lw=1.4)
ax.set_xlabel("Residual (measured − predicted) (V/m)"); ax.set_ylabel("Frequency")
fig.tight_layout(); fig.savefig(FIGURES / "Figure_07_Residual_Distribution.png", dpi=300); plt.close(fig)

# Fig. 8
labels = {"Distance_to_nearest_antenna_m": "Distance to antenna", "Nearest_Antenna_Height_m": "Antenna height"}
t = pd.read_csv(RESULTS / "table2_feature_importance.csv").sort_values("Importance_pct")
fig, ax = plt.subplots(figsize=(3.6, 2.55))
ax.barh([labels.get(f, f) for f in t.Feature], t.Importance_pct, color="#2ca02c", edgecolor="black", linewidth=0.7)
ax.set_xlabel("Relative importance (%)")
fig.tight_layout(); fig.savefig(FIGURES / "Figure_08_Feature_Importance.png", dpi=300); plt.close(fig)
print({k: round(v, 3) for k, v in m.items()})
