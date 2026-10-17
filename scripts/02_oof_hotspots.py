"""Section 3.4 / Table 4 / Figs. 9 and 11: 10-fold out-of-fold RF predictions and P85 hotspot agreement."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from common import FEAT, RESULTS, load_dataset, metrics

d = load_dataset(); y = d.E_total.values
rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=42, n_jobs=-1)
oof = cross_val_predict(rf, d[FEAT].values, y, cv=KFold(10, shuffle=True, random_state=42))
print("OOF", {k: round(v, 3) for k, v in metrics(y, oof).items()})
out = d[["Point_ID", "Square", "Latitude", "Longitude", "X_UTM34N", "Y_UTM34N", "E_total"]].copy()
out["E_total_RF_OOF"] = oof
rows = []
for sq in ("KT", "MKA"):
    m = (d.Square == sq).values
    p85m, p85p = np.percentile(y[m], 85), np.percentile(oof[m], 85)
    mt, pt = y[m] >= p85m, oof[m] >= p85p
    cls = np.select([mt & pt, mt & ~pt, ~mt & pt], ["TP", "FN", "FP"], "TN")
    out.loc[m, "Hotspot_class"] = cls
    tp, fp, fn = (mt & pt).sum(), (~mt & pt).sum(), (mt & ~pt).sum()
    rows.append(dict(Square=sq, Measured_P85=p85m, Predicted_P85=p85p, n_measured_hotspots=mt.sum(), TP=tp, FP=fp, FN=fn,
                     Jaccard=tp / (tp + fp + fn), Precision=tp / (tp + fp), Recall=tp / (tp + fn)))
out.to_csv(RESULTS / "oof_predictions_hotspots.csv", index=False, float_format="%.6f")
t4 = pd.DataFrame(rows); t4.to_csv(RESULTS / "table4_hotspot_agreement.csv", index=False, float_format="%.3f")
print(t4.round(3).to_string())
