"""Table 5 and Section 3.5: single 75/25 split, repeated 5- and 10-fold CV, spatial-only RF,
MLR baseline and percentile bootstrap of the single-split test set; also Table 2 (importances)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from common import BANDS, FEAT, SP, RESULTS, load_dataset, metrics

d = load_dataset()
y = d.E_total.values
# Column order matters for Random Forest (random feature permutation at each split). The single split
# uses the column order of the v1.0.0 script (spatial first), which reproduces the headline metrics
# (R2 = 0.806, MAE = 0.231, RMSE = 0.321); the CV protocols and Table 2 use band-first order (FEAT).
FEAT_V1 = SP + BANDS
MODELS = {
    "Random Forest": (FEAT, lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_depth=None, random_state=42, n_jobs=-1)),
    "RF spatial-only": (SP, lambda: RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_depth=None, random_state=42, n_jobs=-1)),
    "Linear regression": (FEAT, LinearRegression),
}
rows, fold_rows = [], []
# single split (identical to the v1 script: test_size=0.25, random_state=42)
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42)
single_pred = {}
for name, (cols, mk) in MODELS.items():
    if name == "Random Forest":
        # Table 2 importances (band-first order)
        imp_m = mk().fit(d.loc[idx_tr, cols].values, y[idx_tr])
        pd.DataFrame(dict(Feature=cols, Importance_pct=100 * imp_m.feature_importances_)).sort_values(
            "Importance_pct", ascending=False).to_csv(RESULTS / "table2_feature_importance.csv", index=False, float_format="%.2f")
        cols = FEAT_V1
    m = mk().fit(d.loc[idx_tr, cols].values, y[idx_tr])
    p = m.predict(d.loc[idx_te, cols].values); single_pred[name] = p
    rows.append(dict(Protocol="Single 75/25 split", Model=name, **{k: f"{v:.3f}" for k, v in metrics(y[idx_te], p).items() if k != "Bias"}))
# spatial-only importances (model fitted to all 171 points, as reported in Section 3.5)
_m = MODELS["RF spatial-only"][1]().fit(d[SP].values, y)
pd.DataFrame(dict(Feature=SP, Importance_pct=100 * _m.feature_importances_)).sort_values("Importance_pct", ascending=False).to_csv(
    RESULTS / "spatial_only_importance.csv", index=False, float_format="%.1f")
# repeated k-fold
for k in (5, 10):
    for seed in range(10):
        for f, (tr, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(y)):
            for name, (cols, mk) in MODELS.items():
                p = mk().fit(d.loc[tr, cols].values, y[tr]).predict(d.loc[te, cols].values)
                fold_rows.append(dict(k=k, seed=seed, fold=f, Model=name, **{a: b for a, b in metrics(y[te], p).items() if a != "Bias"}))
fr = pd.DataFrame(fold_rows); fr.to_csv(RESULTS / "cv_fold_metrics.csv", index=False, float_format="%.5f")
for k in (5, 10):
    for name in MODELS:
        g = fr[(fr.k == k) & (fr.Model == name)]
        rows.append(dict(Protocol=f"{k}-fold x 10 repeats", Model=name,
                         **{m: f"{g[m].mean():.3f} ± {g[m].std(ddof=0):.3f}" for m in ("R2", "MAE", "RMSE")}))
t5 = pd.DataFrame(rows)
t5.to_csv(RESULTS / "table5_validation.csv", index=False, float_format="%.3f")
print(t5.to_string())
# bootstrap of single-split RF test predictions
rng = np.random.default_rng(42)
yt, pt = y[idx_te], single_pred["Random Forest"]
sq = d.Square.values[idx_te]
B = []
for b in range(5000):
    i = rng.integers(0, len(yt), len(yt)); B.append(metrics(yt[i], pt[i]))
B = pd.DataFrame(B)
ci = [dict(Subset="Pooled", n=len(yt), **{f"{m}": metrics(yt, pt)[m] for m in ("R2", "MAE", "RMSE")},
           **{f"{m}_CI95": f"[{B[m].quantile(.025):.2f}, {B[m].quantile(.975):.2f}]" for m in ("R2", "MAE", "RMSE")})]
for s in ("KT", "MKA"):
    m_ = sq == s; ys, ps = yt[m_], pt[m_]; Bs = []
    for b in range(5000):
        i = rng.integers(0, len(ys), len(ys)); Bs.append(metrics(ys[i], ps[i]))
    Bs = pd.DataFrame(Bs)
    ci.append(dict(Subset=s, n=len(ys), **{m: metrics(ys, ps)[m] for m in ("R2", "MAE", "RMSE")},
                   **{f"{m}_CI95": f"[{Bs[m].quantile(.025):.2f}, {Bs[m].quantile(.975):.2f}]" for m in ("R2", "MAE", "RMSE")}))
ci = pd.DataFrame(ci); ci.to_csv(RESULTS / "bootstrap_ci_single_split.csv", index=False, float_format="%.3f")
print(ci.to_string())
