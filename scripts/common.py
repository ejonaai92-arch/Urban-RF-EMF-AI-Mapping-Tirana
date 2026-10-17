"""Shared paths, constants and loaders for the v2.0.0 analyses.

All scripts are run from the repository root, e.g. `python scripts/01_validation.py`.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
WORK = ROOT / "work"          # intermediate files (not archived)
for p in (RESULTS, FIGURES, WORK):
    p.mkdir(exist_ok=True)

BANDS = ["LTE700", "LTE800", "GSM900", "GSM1800", "LTE2100", "LTE2600", "NR3500"]
SP = ["Latitude", "Longitude", "Distance_to_nearest_antenna_m", "Nearest_Antenna_Height_m"]
FEAT = BANDS + SP

# Zone labels in the dataset -> square names used in the manuscript
ZONE_TO_SQUARE = {"21_Dhjetori": "MKA", "Zogu_i_Zi": "KT"}
SQUARE_NAMES = {"MKA": "Mustafa Kemal Atatürk Square", "KT": "Karl Topia Square"}

_UTM = Transformer.from_crs(4326, 32634, always_xy=True)


def load_dataset():
    """Point-level dataset (171 rows) with square, timestamp, day type and UTM 34N coordinates."""
    d = pd.read_csv(DATA / "urban_rf_emf_ai_dataset.csv")
    d["Square"] = d.Zone.map(ZONE_TO_SQUARE)
    # Date is stored as MM/DD/YYYY
    d["dt"] = pd.to_datetime(d.Date + " " + d.Time, format="%m/%d/%Y %H:%M:%S")
    d["Weekday"] = d.dt.dt.day_name()
    d["DayType"] = np.where(d.dt.dt.dayofweek >= 5, "WE", "WD")
    d["X_UTM34N"], d["Y_UTM34N"] = _UTM.transform(d.Longitude.values, d.Latitude.values)
    d["X"], d["Y"] = d.X_UTM34N, d.Y_UTM34N
    return d


def load_base_stations():
    bs = pd.read_csv(DATA / "base_stations.csv")
    bs["X"], bs["Y"] = _UTM.transform(bs.Longitude.values, bs.Latitude.values)
    return bs


def dms_to_deg(text):
    d, m, s = map(float, re.findall(r"[\d.]+", text)[:3])
    return d + m / 60 + s / 3600


def metrics(y, p):
    e = np.asarray(p) - np.asarray(y)
    return dict(R2=1 - np.sum(e**2) / np.sum((y - np.mean(y)) ** 2),
                MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e**2)), Bias=e.mean())
