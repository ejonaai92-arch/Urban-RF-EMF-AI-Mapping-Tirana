"""Section 2.1: point spacing, extents, convex-hull areas and densities (Reviewer 1, Comment 6)."""
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist
from common import RESULTS, load_dataset

d = load_dataset(); rows = []
for sq in ("MKA", "KT"):
    g = d[d.Square == sq]; xy = g[["X", "Y"]].values
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]; A = ConvexHull(xy).volume / 1e4; ext = xy.max(0) - xy.min(0)
    rows.append(dict(Square=sq, n=len(g), NN_mean_m=nn.mean(), NN_median_m=np.median(nn), Pairwise_mean_m=pdist(xy).mean(),
                     Extent_E_m=ext[0], Extent_N_m=ext[1], Hull_area_ha=A, Points_per_ha=len(g) / A,
                     Dist_BS_min_m=g.Distance_to_nearest_antenna_m.min(), Dist_BS_max_m=g.Distance_to_nearest_antenna_m.max(),
                     UTM_500m_grid_cells_touched=len(set(map(tuple, (xy // 500).astype(int))))))
r = pd.DataFrame(rows); r.to_csv(RESULTS / "sampling_geometry.csv", index=False, float_format="%.2f"); print(r.round(2).T)
