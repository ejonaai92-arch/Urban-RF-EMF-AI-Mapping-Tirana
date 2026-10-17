"""Fig. 12 / Section 3.5: learning curves (15-128 training points, 30 repetitions). In each repetition the data are
split anew into a 43-point held-out test set and a 128-point training pool; nested training subsets of increasing
size are drawn from that pool and all sizes are evaluated on the same test set of that repetition."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from common import FEAT, SP, RESULTS, load_dataset

d = load_dataset(); X = d[FEAT].values; Xs = d[SP].values; y = d.E_total.values
rng = np.random.default_rng(1); sizes = [15, 25, 40, 60, 80, 100, 128]; R = []
for rep in range(30):
    perm = rng.permutation(len(y)); te, pool = perm[:43], perm[43:]
    for n in sizes:
        tr = pool[:n]
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
        R.append(dict(n=n, rep=rep, model="RF (11 features)", r2=r2_score(y[te], rf.fit(X[tr], y[tr]).predict(X[te]))))
        R.append(dict(n=n, rep=rep, model="MLR (11 features)", r2=r2_score(y[te], LinearRegression().fit(X[tr], y[tr]).predict(X[te]))))
        R.append(dict(n=n, rep=rep, model="RF (spatial only)", r2=r2_score(y[te], rf.fit(Xs[tr], y[tr]).predict(Xs[te]))))
r = pd.DataFrame(R); r.to_csv(RESULTS / "learning_curves_raw.csv", index=False, float_format="%.5f")
s = r.groupby(["model", "n"]).r2.agg(median="median", q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75))
s.to_csv(RESULTS / "learning_curves_summary.csv", float_format="%.3f"); print(s.round(3))
