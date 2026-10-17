"""Writes data/base_stations.csv: the five surveyed rooftop base stations (Fig. 1).

Coordinates are given in WGS84 (DMS from the field survey); antenna elevations above sea level were
recorded in feet and are converted to metres exactly as in the predictor Nearest_Antenna_Height_m.
"""
import pandas as pd
from common import DATA, dms_to_deg

SURVEY = [  # label, square, Nearest_Antenna_ID in dataset, lat, lon, elevation a.s.l. (ft)
    ("BS-K1", "KT", "Zogu_Zi", "41°20'01.27''", "19°48'17.07''", 463),
    ("BS-K2", "KT", "Zogu_Zi", "41°19'55.49''", "19°48'17.69''", 443),
    ("BS-M1", "MKA", "21_Djetori", "41°19'34.07''", "19°48'13.01''", 375),
    ("BS-M2", "MKA", "21_Djetori", "41°19'33.64''", "19°48'11.01''", 400),
    ("BS-M3", "MKA", "21_Djetori", "41°19'33.78''", "19°48'10.60''", 400),
]
rows = [dict(BS_label=l, Square=s, Nearest_Antenna_ID=i, Latitude_DMS=la, Longitude_DMS=lo,
             Latitude=dms_to_deg(la), Longitude=dms_to_deg(lo), Elevation_asl_ft=h, Elevation_asl_m=h * 0.3048)
        for l, s, i, la, lo, h in SURVEY]
pd.DataFrame(rows).to_csv(DATA / "base_stations.csv", index=False, float_format="%.6f")
print(pd.DataFrame(rows))
