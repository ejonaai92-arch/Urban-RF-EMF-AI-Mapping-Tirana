"""Reads the original Narda SRM-3006 spectrum exports and writes the 171 x 531 Eact matrix aligned to the
rows of data/urban_rf_emf_ai_dataset.csv.

Input : data/raw_spectra/<Zone>/Dt-<day>/P<nn>/P-0482_<index>_00001.csv  (one folder per measurement point;
        Export_from_Narda/ holds the same files as exported from the instrument)
Output: data/spectra_Eact_531bins.csv   Point_ID + 531 bins (V/m), header = bin frequency in MHz
        data/spectra_source_files.csv  Point_ID -> source file, Narda index, instrument date/time, settings
Matching: by the instrument date and time stored in each file; a dataset row whose time differs from the
instrument time is matched by an identical RSS on the same date (reported).
"""
import glob
import os
import numpy as np
import pandas as pd
from common import DATA, load_dataset

RAW = DATA / "raw_spectra"
recs = []
for f in sorted(glob.glob(str(RAW / "*" / "Dt-*" / "P*" / "*_00001.csv"))):
    L = open(f, encoding="latin-1").read().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("Frequency [Hz]"))
    kv = {l.split(",")[0]: l.split(",", 1)[1].strip() if "," in l else "" for l in L[:i]}
    a = np.array([[float(x) for x in l.split(",")[:4]] for l in L[i + 1:] if l.strip()])
    recs.append(dict(file=os.path.relpath(f, DATA), Narda_index=int(os.path.basename(f).split("_")[1]),
                     Date=kv["Date"], Time=kv["Time"], F=a[:, 0], act=a[:, 1], RSS=np.sqrt((a[:, 1] ** 2).sum()),
                     RBW_Hz=kv.get("RBW [Hz]"), df_Hz=kv.get("df [Hz]"), Tavg_s=kv.get("Time Avg. [s]"),
                     Sweep_ms=kv.get("Sweep Time [ms]"), MR_Vm=kv.get("MR [V/m]"), Antenna=kv.get("Antenna"),
                     GPS_Quality=kv.get("GPS Quality"), GPS_Satellites=kv.get("GPS Satellites in Use"), Overdriven=kv.get("Overdriven")))
r = pd.DataFrame(recs)
F = r.F.iloc[0]; assert all(np.array_equal(x, F) for x in r.F) and len(F) == 531
d = load_dataset()
match = []
for _, row in d.iterrows():
    c = r[(r.Date == row.Date) & (r.Time == row.Time)]
    how = "date+time"
    if len(c) != 1:
        c = r[(r.Date == row.Date) & (np.abs(r.RSS - row.E_total) < 1e-9)]; how = "date+identical RSS"
        print(f"{row.Point_ID}: dataset time {row.Time} -> instrument time {c.Time.iloc[0]} ({c.file.iloc[0]})")
    assert len(c) == 1, row.Point_ID
    match.append((c.index[0], how))
idx = [m[0] for m in match]; S = np.vstack(r.act.loc[idx].values)
diff = np.abs(np.sqrt((S ** 2).sum(1)) - d.E_total.values); print("max |RSS - E_total| =", diff.max()); assert diff.max() < 1e-6
unused = r.drop(index=idx); print("spectra not used (repeated measurements):", list(unused.file))
out = pd.DataFrame(S, columns=[f"{x/1e6:.0f}" for x in F]); out.insert(0, "Point_ID", d.Point_ID.values)
out.to_csv(DATA / "spectra_Eact_531bins.csv", index=False, float_format="%.6g")
src = r.loc[idx].drop(columns=["F", "act"]).reset_index(drop=True)
src.insert(0, "Point_ID", d.Point_ID.values); src.insert(1, "Match", [m[1] for m in match])
src.rename(columns={"Date": "Instrument_date", "Time": "Instrument_time"}).to_csv(DATA / "spectra_source_files.csv", index=False, float_format="%.6f")
