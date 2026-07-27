import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==================================================
# PATHS
# ==================================================

INPUT_CSV = r"C:\Users\Dori\Desktop\AI_EMF\data\processed\urban_rf_emf_ai_dataset.csv"
OUTPUT_DIR = r"C:\Users\Dori\Desktop\AI_EMF\outputs\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_CSV)

# ==================================================
# FEATURES
# ==================================================

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

# Cleaner labels for figures
label_map = {
    "Distance_to_nearest_antenna_m": "Distance to antenna",
    "Nearest_Antenna_Height_m": "Antenna height",
    "E_total": "Total E",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "LTE700": "LTE700",
    "LTE800": "LTE800",
    "GSM900": "GSM900",
    "GSM1800": "GSM1800",
    "LTE2100": "LTE2100",
    "LTE2600": "LTE2600",
    "NR3500": "NR3500"
}

# ==================================================
# HELPER: SAVE PNG + TIF
# ==================================================

def save_publication_figure(fig, base_name):
    png_path = os.path.join(OUTPUT_DIR, base_name + ".png")
    tif_path = os.path.join(OUTPUT_DIR, base_name + ".tif")

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08
    )

    plt.close(fig)

    img = Image.open(png_path)
    img.save(
        png_path,
        optimize=True,
        compress_level=9
    )

    img = Image.open(png_path).convert("RGB")
    img.save(
        tif_path,
        compression="tiff_lzw",
        dpi=(600, 600)
    )

    print("Saved:", png_path)
    print("Saved:", tif_path)

# ==================================================
# FIGURE 4: CLEAN CORRELATION MATRIX
# ==================================================

corr_cols = [
    "Distance_to_nearest_antenna_m",
    "LTE700",
    "LTE800",
    "GSM900",
    "GSM1800",
    "LTE2100",
    "LTE2600",
    "NR3500",
    "E_total"
]

corr = df[corr_cols].corr(numeric_only=True)

corr = corr.rename(
    index=label_map,
    columns=label_map
)

fig, ax = plt.subplots(figsize=(8.8, 7.2))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0,
    square=True,
    linewidths=0.4,
    linecolor="white",
    cbar_kws={"shrink": 0.85},
    annot_kws={"fontsize": 9},
    ax=ax
)

ax.set_title(
    "Correlation Matrix of RF-EMF Variables",
    fontsize=15,
    fontweight="bold",
    pad=12
)

ax.tick_params(axis="x", labelrotation=45, labelsize=9)
ax.tick_params(axis="y", labelrotation=0, labelsize=9)

fig.tight_layout()

save_publication_figure(
    fig,
    "Figure_04_Correlation_Matrix_Clean"
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
# FIGURE 5A: MEASURED VS PREDICTED
# ==================================================

fig, ax = plt.subplots(figsize=(7.2, 7.0))

ax.scatter(
    y_test,
    y_pred,
    s=52,
    alpha=0.75,
    edgecolor="black",
    linewidth=0.35
)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())

padding = (line_max - line_min) * 0.08
line_min -= padding
line_max += padding

ax.plot(
    [line_min, line_max],
    [line_min, line_max],
    linestyle="--",
    linewidth=2
)

ax.set_xlim(line_min, line_max)
ax.set_ylim(line_min, line_max)

ax.set_xlabel(
    "Measured total E-field (V/m)",
    fontsize=12
)

ax.set_ylabel(
    "Predicted total E-field (V/m)",
    fontsize=12
)

ax.set_title(
    "Measured vs Predicted RF-EMF Exposure",
    fontsize=15,
    fontweight="bold",
    pad=12
)

metric_text = (
    f"R² = {r2:.3f}\n"
    f"MAE = {mae:.3f} V/m\n"
    f"RMSE = {rmse:.3f} V/m"
)

ax.text(
    0.05,
    0.95,
    metric_text,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=11,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black",
        alpha=0.9
    )
)

ax.grid(True, linestyle="--", alpha=0.25)

fig.tight_layout()

save_publication_figure(
    fig,
    "Figure_05a_Measured_vs_Predicted_RF"
)

# ==================================================
# FIGURE 5B: CLEAN FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
})

importance["Importance (%)"] = importance["Importance"] * 100
importance["Feature label"] = importance["Feature"].replace(label_map)

# Keep only meaningful features >= 1%
importance_clean = importance[
    importance["Importance (%)"] >= 1.0
].copy()

importance_clean = importance_clean.sort_values(
    by="Importance (%)",
    ascending=True
)

fig, ax = plt.subplots(figsize=(8.2, 5.8))

ax.barh(
    importance_clean["Feature label"],
    importance_clean["Importance (%)"]
)

ax.set_xlabel(
    "Importance (%)",
    fontsize=12
)

ax.set_ylabel("")

ax.set_title(
    "Random Forest Feature Importance",
    fontsize=15,
    fontweight="bold",
    pad=12
)

ax.grid(axis="x", linestyle="--", alpha=0.25)

for i, value in enumerate(importance_clean["Importance (%)"]):
    ax.text(
        value + 0.4,
        i,
        f"{value:.1f}%",
        va="center",
        fontsize=10
    )

fig.tight_layout()

save_publication_figure(
    fig,
    "Figure_05b_Random_Forest_Feature_Importance_Clean"
)

# ==================================================
# SAVE NUMERIC RESULTS
# ==================================================

results_txt = os.path.join(
    OUTPUT_DIR,
    "Random_Forest_Model_Performance.txt"
)

with open(results_txt, "w", encoding="utf-8") as f:
    f.write("Random Forest Model Performance\n")
    f.write("--------------------------------\n")
    f.write(f"R2   : {r2:.4f}\n")
    f.write(f"MAE  : {mae:.4f} V/m\n")
    f.write(f"RMSE : {rmse:.4f} V/m\n")

print("\nMODEL PERFORMANCE")
print("------------------------")
print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f} V/m")
print(f"RMSE : {rmse:.4f} V/m")
print("Saved:", results_txt)