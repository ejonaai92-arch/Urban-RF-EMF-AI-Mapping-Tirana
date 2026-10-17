"""Table 3 / Section 2.3: leave-one-out comparison of interpolators for measured E_total, plus fitted variograms."""
import warnings
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import Rbf, griddata
from common import RESULTS, load_dataset, metrics

warnings.filterwarnings("ignore")
d = load_dataset()
out, vg = [], []
rng = np.random.default_rng(0)  # 5 cm jitter separates exactly co-located (re-measured) points
for sq in ("KT", "MKA"):
    g = d[d.Square == sq].reset_index(drop=True)
    X = g.X.values + rng.normal(0, 0.05, len(g)); Y = g.Y.values + rng.normal(0, 0.05, len(g)); z = g.E_total.values
    lon, lat = g.Longitude.values, g.Latitude.values
    P = {k: np.zeros(len(g)) for k in ["Linear (Delaunay)", "IDW (power 2)", "RBF multiquadric", "OK, exponential", "OK, spherical", "Global mean (reference)"]}
    nfb = 0
    for i in range(len(g)):
        m = np.arange(len(g)) != i
        v = griddata(np.c_[X[m], Y[m]], z[m], (X[i], Y[i]), method="linear")
        if np.isnan(v):  # outside the convex hull of the remaining points -> nearest neighbour
            v = griddata(np.c_[X[m], Y[m]], z[m], (X[i], Y[i]), method="nearest"); nfb += 1
        P["Linear (Delaunay)"][i] = v
        w = 1 / np.maximum(np.hypot(X[m] - X[i], Y[m] - Y[i]), 0.5) ** 2
        P["IDW (power 2)"][i] = np.sum(w * z[m]) / w.sum()
        P["RBF multiquadric"][i] = Rbf(lon[m], lat[m], z[m], function="multiquadric", smooth=0.0006)(lon[i], lat[i])
        for vm in ("exponential", "spherical"):
            ok = OrdinaryKriging(X[m], Y[m], z[m], variogram_model=vm, nlags=10, enable_plotting=False)
            P[f"OK, {vm}"][i] = ok.execute("points", [X[i]], [Y[i]])[0][0]
        P["Global mean (reference)"][i] = z[m].mean()
    for k, p in P.items():
        out.append(dict(Area=sq, Method=k, **metrics(z, p)))
    for vm in ("exponential", "spherical"):
        psill, rang, nug = OrdinaryKriging(X, Y, z, variogram_model=vm, nlags=10).variogram_model_parameters
        vg.append(dict(Area=sq, Model=vm, Partial_sill=psill, Range_m=rang, Nugget=nug, Nugget_to_sill=nug / (psill + nug)))
    print(sq, "linear-interpolation fallbacks (outside hull):", nfb)
r = pd.DataFrame(out); r.to_csv(RESULTS / "table3_interpolation_loocv.csv", index=False, float_format="%.4f")
v = pd.DataFrame(vg); v.to_csv(RESULTS / "variogram_parameters.csv", index=False, float_format="%.4f")
print(r.round(3).to_string()); print(v.round(3).to_string())
