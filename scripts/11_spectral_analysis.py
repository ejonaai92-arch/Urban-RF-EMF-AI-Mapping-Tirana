"""Sections 2.2, 3.1, 4: noise- and RBW-corrected total field, FDD uplink share, NR3500 share and
ICNIRP 2020 exposure ratios, from data/spectra_Eact_531bins.csv.
Output: data/point_spectral_summary.csv and results/spectral_summary.txt"""
import numpy as np
import pandas as pd
from common import BANDS, DATA, RESULTS, load_dataset

d = load_dataset()
sp = pd.read_csv(DATA / "spectra_Eact_531bins.csv"); assert (sp.Point_ID.values == d.Point_ID.values).all()
F = sp.columns[1:].astype(float).values; S = sp.iloc[:, 1:].values; P = S ** 2
lines = []
def log(s): print(s); lines.append(s)
# frequency-dependent noise floor fitted on bins without cellular allocations
quiet = ((F >= 1000) & (F <= 1400)) | ((F >= 2200) & (F <= 2380)) | ((F >= 2720) & (F <= 3280)) | ((F >= 3820) & (F <= 4900)) | (F >= 5900)
c = np.polyfit(np.log10(F[quiet]), np.log10(np.median(P[:, quiet], 0)), 1); N = 10 ** np.polyval(c, np.log10(F))
log(f"Noise floor per bin: {N[0]**.5:.3f} V/m at {F[0]:.0f} MHz, {N[-1]**.5:.3f} V/m at {F[-1]:.0f} MHz")
Psig = np.clip(P - N, 0, None)
d["E_corr"] = np.sqrt(0.5 * Psig.sum(1))       # x (10 MHz step / 20 MHz RBW)
for sq, g in d.groupby("Square"):
    log(f"{sq}: E_corr mean {g.E_corr.mean():.2f} V/m (range {g.E_corr.min():.2f}-{g.E_corr.max():.2f})")
# FDD bands as in Section 3.1 (LTE700 excluded: its uplink window overlaps non-cellular emissions)
UL = {"LTE800": (832, 862), "GSM900": (880, 915), "GSM1800": (1710, 1785), "LTE2100": (1920, 1980), "LTE2600": (2500, 2570)}
DL = {"LTE800": (791, 821), "GSM900": (925, 960), "GSM1800": (1805, 1880), "LTE2100": (2110, 2170), "LTE2600": (2620, 2690)}
inside = lambda lo, hi: (F - 10 >= lo - 1) & (F + 10 <= hi + 1)
ul = sum(0.5 * Psig[:, inside(*UL[b])].sum(1) for b in UL); dl = sum(0.5 * Psig[:, inside(*DL[b])].sum(1) for b in DL)
d["FDD_uplink_share"] = ul / (ul + dl)
top5 = d.nlargest(5, "E_total")
log(f"FDD uplink share: median {d.FDD_uplink_share.median():.3f}, P90 {d.FDD_uplink_share.quantile(.9):.3f}, max {d.FDD_uplink_share.max():.3f}; "
    f"five highest-E_total points: max {top5.FDD_uplink_share.max():.4f}")
nr = (F >= 3300) & (F <= 3800); d["NR3500_share_corrected"] = Psig[:, nr].sum(1) / Psig.sum(1)
log(f"NR3500 share of corrected power: median {d.NR3500_share_corrected.median():.3f}")
EL = np.where(F < 2000, 1.375 * np.sqrt(F), 61.0)   # ICNIRP 2020 general-public E reference level (V/m)
d["ER_raw"] = (P / EL ** 2).sum(1); d["ER_corrected"] = (0.5 * Psig / EL ** 2).sum(1)
log(f"ICNIRP exposure ratio (sum E^2/E_L^2): raw max {d.ER_raw.max():.4f}, corrected max {d.ER_corrected.max():.4f}")
r7 = np.sqrt((d[BANDS] ** 2).sum(axis=1))
log(f"RSS of the seven bands relative to E_total: mean {100*(1-(r7/d.E_total).mean()):.1f}% lower")
log(f"Max E_total {d.E_total.max():.2f} V/m = {100*d.E_total.max()/41:.1f}% of 41 V/m and {100*d.E_total.max()/(1.375*700**.5):.1f}% of {1.375*700**.5:.1f} V/m")
d[["Point_ID", "Square", "E_total", "E_corr", "FDD_uplink_share", "NR3500_share_corrected", "ER_raw", "ER_corrected"]].to_csv(
    DATA / "point_spectral_summary.csv", index=False, float_format="%.6f")
(RESULTS / "spectral_summary.txt").write_text("\n".join(lines) + "\n")
