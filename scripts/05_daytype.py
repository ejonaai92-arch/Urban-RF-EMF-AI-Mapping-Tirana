"""Sections 2.5 and 3.1: weekday vs weekend sensitivity of E_total and of the models."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from common import DATA, FEAT, RESULTS, load_dataset

d = load_dataset(); X = d[FEAT].values; y = d.E_total.values
lines = []
def log(s): print(s); lines.append(s)
log("Schedule (points per date):\n" + d.groupby(["Square", "Date", "Weekday"]).size().to_string())
meta = pd.read_csv(DATA / "point_metadata.csv")
log(f"Weekday points {(d.DayType=='WD').sum()}, weekend points {(d.DayType=='WE').sum()}; "
    f"measurement sessions (field sheets) {meta.Session_start.min()[:5]}-{meta.Session_end.max()[:5]}; instrument save times {d.Time.min()}-{d.Time.max()}")
# (a) pooled Mann-Whitney
a, b = d[d.DayType == "WD"].E_total, d[d.DayType == "WE"].E_total
log(f"(a) E_total median WD {a.median():.2f} V/m vs WE {b.median():.2f} V/m, Mann-Whitney p = {stats.mannwhitneyu(a, b).pvalue:.4f}")
# (b) paired: locations within 10 m (complete linkage) measured on both day types
rows = []
for sq in ("KT", "MKA"):
    g = d[d.Square == sq].copy()
    g["cl"] = fcluster(linkage(g[["X", "Y"]].values, "complete"), 10, "distance")
    for c, x in g.groupby("cl"):
        if set(x.DayType) == {"WD", "WE"}:
            rows.append(dict(Square=sq, cluster=c, n_points=len(x), WD=x[x.DayType == "WD"].E_total.mean(), WE=x[x.DayType == "WE"].E_total.mean()))
c = pd.DataFrame(rows); c.to_csv(RESULTS / "daytype_paired_locations.csv", index=False, float_format="%.4f")
log(f"(b) {len(c)} locations ({c.n_points.sum()} points) measured on both day types; WD higher at {(c.WD>c.WE).sum()}; "
    f"median diff {(c.WD-c.WE).median():.2f} V/m ({100*((c.WD/c.WE)-1).median():.1f}%); Wilcoxon p = {stats.wilcoxon(c.WD, c.WE).pvalue:.4f}")
# (c) model sensitivity
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
oof = cross_val_predict(rf, X, y, cv=KFold(10, shuffle=True, random_state=0)); res = y - oof
log(f"(c) OOF residual median WD {np.median(res[d.DayType=='WD']):.3f} vs WE {np.median(res[d.DayType=='WE']):.3f}, "
    f"Mann-Whitney p = {stats.mannwhitneyu(res[d.DayType=='WD'], res[d.DayType=='WE']).pvalue:.3f}")
Xd = np.column_stack([X, (d.DayType == "WD").astype(int)])
for name, XX in (("11 features", X), ("11 features + day type", Xd)):
    for mname, m in (("RF", rf), ("MLR", LinearRegression())):
        r2 = [r2_score(y, cross_val_predict(m, XX, y, cv=KFold(10, shuffle=True, random_state=s))) for s in range(10)]
        log(f"    {name:24s} {mname}: 10x10 pooled-OOF R2 = {np.mean(r2):.3f}")
for dt in ("WD", "WE"):
    m_ = (d.DayType == dt).values
    for mname, mm in (("RF", rf), ("MLR", LinearRegression())):
        p = cross_val_predict(mm, X[m_], y[m_], cv=KFold(10, shuffle=True, random_state=0))
        log(f"    separate model {dt} (n={m_.sum()}) {mname}: OOF R2 = {r2_score(y[m_], p):.3f}")
for a_, b_ in (("WD", "WE"), ("WE", "WD")):
    ma, mb = (d.DayType == a_).values, (d.DayType == b_).values
    for mname, mm in (("RF", rf), ("MLR", LinearRegression())):
        p = mm.fit(X[ma], y[ma]).predict(X[mb]); log(f"    train {a_} -> test {b_} {mname}: R2 = {r2_score(y[mb], p):.3f}")
(RESULTS / "daytype_sensitivity.txt").write_text("\n".join(lines) + "\n")
