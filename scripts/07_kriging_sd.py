"""Section 3.4: range of the ordinary-kriging standard deviation within the convex hull of the points
(spherical variogram, 2 m grid, same settings as the kriged surfaces of Figs. 9-10 in 20_figures.py)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib.path import Path
from pykrige.ok import OrdinaryKriging
from scipy.spatial import ConvexHull
from common import RESULTS, load_dataset

d = load_dataset(); rows = []
for sq in ("KT", "MKA"):
    g = d[d.Square == sq]
    gx = np.arange(g.X.min(), g.X.max() + 2.0, 2.0); gy = np.arange(g.Y.min(), g.Y.max() + 2.0, 2.0)
    rng = np.random.default_rng(0)
    ok = OrdinaryKriging(g.X.values + rng.normal(0, 0.05, len(g)), g.Y.values + rng.normal(0, 0.05, len(g)), g.E_total.values,
                         variogram_model="spherical", nlags=10)
    _, ss = ok.execute("grid", gx, gy); XX, YY = np.meshgrid(gx, gy)
    pts = g[["X", "Y"]].values; h = ConvexHull(pts)
    m = Path(pts[h.vertices]).contains_points(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    sd = np.sqrt(np.asarray(ss))[m]
    rows.append(dict(Square=sq, SD_min=sd.min(), SD_median=np.median(sd), SD_max=sd.max()))
r = pd.DataFrame(rows); r.to_csv(RESULTS / "kriging_sd_range.csv", index=False, float_format="%.3f"); print(r.round(3))
