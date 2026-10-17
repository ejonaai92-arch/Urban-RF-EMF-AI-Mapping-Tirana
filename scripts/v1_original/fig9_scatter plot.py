from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
DATASET_CSV = PROJECT_ROOT / "data" / "urban_rf_emf_ai_dataset_corrected.csv"


# ============================================================
# Figure - Measured vs Predicted Total RF-EMF Exposure
# Random Forest regression scatter plot
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# PATHS
# =========================

BASE_DIR = str(PROJECT_ROOT)

INPUT_CSV = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "all",
    "urban_rf_emf_ai_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "figures_publication"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(
    OUT_DIR,
    "Figure_Measured_vs_Predicted_Total_RF_EMF_Exposure.png"
)

OUT_TIF = os.path.join(
    OUT_DIR,
    "Figure_Measured_vs_Predicted_Total_RF_EMF_Exposure.tif"
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

# =========================
# FEATURES AND TARGET
# =========================

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

# =========================
# TRAIN / TEST SPLIT
# =========================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# =========================
# RANDOM FOREST MODEL
# =========================

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# =========================
# MODEL METRICS
# =========================

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# =========================
# PLOT
# =========================

fig, ax = plt.subplots(figsize=(7.0, 6.8))

ax.scatter(
    y_test,
    y_pred,
    s=65,
    alpha=0.75,
    edgecolor="black",
    linewidth=0.5
)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

padding = (line_max - line_min) * 0.08
line_min -= padding
line_max += padding

ax.plot(
    [line_min, line_max],
    [line_min, line_max],
    "--",
    linewidth=2.5,
    color="dimgray",
    label="Ideal agreement (y = x)"
)

ax.set_xlim(line_min, line_max)
ax.set_ylim(line_min, line_max)

ax.set_xlabel(
    "Measured total RF-EMF exposure (V/m)",
    fontsize=13
)

ax.set_ylabel(
    "Predicted total RF-EMF exposure (V/m)",
    fontsize=13
)

ax.set_title(
    "Measured vs Predicted Total RF-EMF Exposure",
    fontsize=17,
    fontweight="bold",
    pad=14
)

metric_text = (
    f"R² = {r2:.3f}\n"
    f"MAE = {mae:.3f} V/m\n"
    f"RMSE = {rmse:.3f} V/m"
)

ax.text(
    0.04,
    0.95,
    metric_text,
    transform=ax.transAxes,
    fontsize=11,
    va="top",
    ha="left",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        boxstyle="round,pad=0.35",
        alpha=0.95
    )
)

ax.legend(
    loc="lower right",
    fontsize=10,
    frameon=True
)

ax.grid(
    True,
    linestyle="--",
    alpha=0.25
)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

fig.tight_layout()

# =========================
# SAVE PNG + TIF
# =========================

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)

plt.close(fig)

img = Image.open(OUT_PNG)
img.save(
    OUT_PNG,
    optimize=True,
    compress_level=9
)

img = Image.open(OUT_PNG).convert("RGB")
img.save(
    OUT_TIF,
    compression="tiff_lzw",
    dpi=(600, 600)
)

print("DONE.")
print("Saved:")
print(OUT_PNG)
print(OUT_TIF)

print("\nMODEL PERFORMANCE")
print("------------------")
print(f"R²   = {r2:.4f}")
print(f"MAE  = {mae:.4f} V/m")
print(f"RMSE = {rmse:.4f} V/m")