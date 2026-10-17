from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
DATASET_CSV = PROJECT_ROOT / "data" / "urban_rf_emf_ai_dataset_corrected.csv"


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==================================================
# PATHS
# ==================================================

BASE_DIR = str(PROJECT_ROOT)

INPUT_CSV = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "all",
    "urban_rf_emf_ai_dataset.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "tables_and_metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

features = [
    "Latitude",
    "Longitude",
    "Distance_to_nearest_antenna_m",
    "Nearest_Antenna_Height_m",
    "LTE700",
    "LTE800",
    "GSM900",
    "GSM1800",
    "LTE2100",
    "LTE2600",
    "NR3500"
]

target = "E_total"

# ==================================================
# HELPER FUNCTION TO SAVE TABLE AS IMAGE
# ==================================================

def save_table_image(table_df, title, filename):
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.35 * len(table_df)))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    png_path = os.path.join(OUTPUT_DIR, filename + ".png")
    tif_path = os.path.join(OUTPUT_DIR, filename + ".tif")

    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    img = Image.open(png_path)
    img.save(png_path, optimize=True, compress_level=9)

    img = Image.open(png_path).convert("RGB")
    img.save(tif_path, compression="tiff_lzw", dpi=(600, 600))

    print("Saved:", png_path)
    print("Saved:", tif_path)

# ==================================================
# TABLE 1: DESCRIPTIVE STATISTICS
# ==================================================

bands = [
    "LTE700",
    "LTE800",
    "GSM900",
    "GSM1800",
    "LTE2100",
    "LTE2600",
    "NR3500",
    "E_total"
]

desc = df.groupby("Zone")[bands].agg(["mean", "median", "std", "min", "max"])

desc_table = []

for zone in df["Zone"].unique():
    zone_label = {
        "21_Dhjetori": "Mustafa Q. Atatürk Square",
        "Zogu_i_Zi": "Karl Topia Square"
    }.get(zone, zone)

    for band in bands:
        row = {
            "Zone": zone_label,
            "Band": band.replace("E_total", "Total E"),
            "Mean": round(desc.loc[zone, (band, "mean")], 3),
            "Median": round(desc.loc[zone, (band, "median")], 3),
            "Std": round(desc.loc[zone, (band, "std")], 3),
            "Min": round(desc.loc[zone, (band, "min")], 3),
            "Max": round(desc.loc[zone, (band, "max")], 3),
        }
        desc_table.append(row)

desc_table = pd.DataFrame(desc_table)

desc_csv = os.path.join(OUTPUT_DIR, "Table_01_Descriptive_Statistics.csv")
desc_table.to_csv(desc_csv, index=False)

save_table_image(
    desc_table,
    "Descriptive statistics of measured RF-EMF exposure",
    "Table_01_Descriptive_Statistics"
)

# ==================================================
# RANDOM FOREST MODEL
# ==================================================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# ==================================================
# TABLE 2: MODEL PERFORMANCE
# ==================================================

performance_table = pd.DataFrame({
    "Metric": ["R²", "MAE", "RMSE"],
    "Value": [
        round(r2, 3),
        round(mae, 3),
        round(rmse, 3)
    ],
    "Unit": ["-", "V/m", "V/m"]
})

performance_csv = os.path.join(OUTPUT_DIR, "Table_02_Model_Performance.csv")
performance_table.to_csv(performance_csv, index=False)

save_table_image(
    performance_table,
    "Random Forest model performance",
    "Table_02_Model_Performance"
)

# ==================================================
# TABLE 3: FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance (%)": rf.feature_importances_ * 100
})

label_map = {
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "Distance_to_nearest_antenna_m": "Distance to antenna",
    "Nearest_Antenna_Height_m": "Antenna height",
    "LTE700": "LTE700",
    "LTE800": "LTE800",
    "GSM900": "GSM900",
    "GSM1800": "GSM1800",
    "LTE2100": "LTE2100",
    "LTE2600": "LTE2600",
    "NR3500": "NR3500"
}

importance["Feature"] = importance["Feature"].replace(label_map)

importance = importance.sort_values(
    by="Importance (%)",
    ascending=False
)

importance["Importance (%)"] = importance["Importance (%)"].round(2)

importance_csv = os.path.join(OUTPUT_DIR, "Table_03_Feature_Importance.csv")
importance.to_csv(importance_csv, index=False)

save_table_image(
    importance,
    "Random Forest feature importance",
    "Table_03_Feature_Importance"
)

# ==================================================
# FIGURE: RESIDUAL DISTRIBUTION
# ==================================================

residuals = y_test - y_pred

fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(
    residuals,
    bins=12,
    edgecolor="black",
    alpha=0.8
)

ax.axvline(
    0,
    linestyle="--",
    linewidth=1.5
)

ax.set_title(
    "Distribution of prediction residuals",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Residual error, measured - predicted (V/m)", fontsize=12)
ax.set_ylabel("Number of samples", fontsize=12)

ax.grid(axis="y", linestyle="--", alpha=0.25)

plt.tight_layout()

res_png = os.path.join(OUTPUT_DIR, "Figure_Residual_Distribution.png")
res_tif = os.path.join(OUTPUT_DIR, "Figure_Residual_Distribution.tif")

plt.savefig(res_png, dpi=600, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

img = Image.open(res_png)
img.save(res_png, optimize=True, compress_level=9)

img = Image.open(res_png).convert("RGB")
img.save(res_tif, compression="tiff_lzw", dpi=(600, 600))

print("\nMODEL PERFORMANCE")
print("------------------")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f} V/m")
print(f"RMSE = {rmse:.4f} V/m")

print("\nAll tables and figures saved in:")
print(OUTPUT_DIR)