from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
DATASET_CSV = PROJECT_ROOT / "data" / "urban_rf_emf_ai_dataset_corrected.csv"


# ============================================================
# Figure - Measured vs AI-predicted RF-EMF exposure maps
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import Rbf
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from PIL import Image

# =========================
# PATHS
# =========================

BASE_DIR = str(PROJECT_ROOT)

DATASET_FILE = os.path.join(
    BASE_DIR, "data", "processed", "all", "urban_rf_emf_ai_dataset.csv"
)

PRED_FILE = os.path.join(
    BASE_DIR, "data", "processed", "all", "real_vs_ai_predictions.csv"
)

OUT_DIR = os.path.join(BASE_DIR, "outputs", "figures_publication")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(
    OUT_DIR,
    "Figure_12_Measured_vs_AI_Predicted_RF_EMF.png"
)

OUT_TIF = os.path.join(
    OUT_DIR,
    "Figure_12_Measured_vs_AI_Predicted_RF_EMF.tif"
)

# =========================
# SETTINGS
# =========================

ZONE_21 = "21_Dhjetori"
ZONE_ZOGU = "Zogu_i_Zi"

ZONE_LABELS = {
    "21_Dhjetori": "Mustafa Q. Atatürk Square",
    "Zogu_i_Zi": "Karl Topia Square"
}

LAT_COL = "Latitude"
LON_COL = "Longitude"
ZONE_COL = "Zone"
MEASURED_COL = "E_total"

GRID_RES = 260
PADDING = 0.00018

cmap = LinearSegmentedColormap.from_list(
    "publication_emf",
    [
        "#fff8b0",
        "#fee08b",
        "#fdae61",
        "#f46d43",
        "#d53e4f",
        "#9e0142",
        "#5e3c99",
        "#313695",
    ],
    N=256
).reversed()

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(DATASET_FILE)
pred = pd.read_csv(PRED_FILE)

df.columns = df.columns.str.strip()
pred.columns = pred.columns.str.strip()

df[ZONE_COL] = df[ZONE_COL].astype(str).str.strip()
pred[ZONE_COL] = pred[ZONE_COL].astype(str).str.strip()

possible_ai_cols = [
    "AI_predicted_E_total",
    "AI_predicted",
    "Predicted_E_total",
    "Predicted",
    "AI_Predicted_E_total",
    "AI prediction",
]

AI_COL = None
for c in possible_ai_cols:
    if c in pred.columns:
        AI_COL = c
        break

if AI_COL is None:
    raise ValueError(
        "Nuk u gjet kolona e AI prediction në real_vs_ai_predictions.csv"
    )

data = df[[ZONE_COL, LAT_COL, LON_COL, MEASURED_COL]].copy()
pred_small = pred[[ZONE_COL, LAT_COL, LON_COL, AI_COL]].copy()

merged = pd.merge(
    data,
    pred_small,
    on=[ZONE_COL, LAT_COL, LON_COL],
    how="inner"
)

if merged.empty:
    raise ValueError(
        "Merge doli bosh. Kontrollo Latitude, Longitude dhe Zone."
    )

print("Merged rows:", len(merged))

# =========================
# COMMON COLOR SCALE
# =========================

vmin = min(merged[MEASURED_COL].min(), merged[AI_COL].min())
vmax = max(merged[MEASURED_COL].max(), merged[AI_COL].max())

levels = np.linspace(vmin, vmax, 18)
contour_levels = np.linspace(vmin, vmax, 7)

# =========================
# FUNCTIONS
# =========================

def make_grid(zone_df):
    lon = zone_df[LON_COL].values
    lat = zone_df[LAT_COL].values

    lon_min = lon.min() - PADDING
    lon_max = lon.max() + PADDING
    lat_min = lat.min() - PADDING
    lat_max = lat.max() + PADDING

    xi = np.linspace(lon_min, lon_max, GRID_RES)
    yi = np.linspace(lat_min, lat_max, GRID_RES)

    return np.meshgrid(xi, yi)


def interpolate_surface(zone_df, value_col):
    lon = zone_df[LON_COL].values
    lat = zone_df[LAT_COL].values
    val = zone_df[value_col].values

    X, Y = make_grid(zone_df)

    rbf = Rbf(
        lon,
        lat,
        val,
        function="multiquadric",
        smooth=0.0006
    )

    Z = rbf(X, Y)

    points = np.column_stack([lon, lat])
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])

    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    mask = hull_path.contains_points(grid_points).reshape(X.shape)

    Z = np.where(mask, Z, np.nan)
    Z = np.clip(Z, vmin, vmax)

    return X, Y, Z


def add_panel_label(ax, label):
    ax.text(
        0.035,
        0.945,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=3
        )
    )


def plot_panel(ax, zone_name, value_col, title, label):
    zdf = merged[merged[ZONE_COL] == zone_name].copy()

    if zdf.empty:
        raise ValueError(f"Nuk u gjetën të dhëna për zonën: {zone_name}")

    X, Y, Z = interpolate_surface(zdf, value_col)

    cf = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
        extend="both"
    )

    ax.contour(
        X,
        Y,
        Z,
        levels=contour_levels,
        colors="black",
        linewidths=0.35,
        alpha=0.35
    )

    ax.scatter(
        zdf[LON_COL],
        zdf[LAT_COL],
        s=16,
        c="black",
        edgecolors="white",
        linewidths=0.35,
        zorder=5
    )

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=8
    )

    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)

    ax.ticklabel_format(
        useOffset=False,
        style="plain"
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    add_panel_label(ax, label)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    return cf


# =========================
# PLOT 2x2
# =========================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.5, 11),
    constrained_layout=False
)

fig.suptitle(
    "Measured and AI-predicted RF-EMF exposure",
    fontsize=17,
    fontweight="bold",
    y=0.965
)

cf1 = plot_panel(
    axes[0, 0],
    ZONE_21,
    MEASURED_COL,
    f"Measured — {ZONE_LABELS[ZONE_21]}",
    "(a)"
)

plot_panel(
    axes[0, 1],
    ZONE_ZOGU,
    MEASURED_COL,
    f"Measured — {ZONE_LABELS[ZONE_ZOGU]}",
    "(b)"
)

plot_panel(
    axes[1, 0],
    ZONE_21,
    AI_COL,
    f"AI predicted — {ZONE_LABELS[ZONE_21]}",
    "(c)"
)

plot_panel(
    axes[1, 1],
    ZONE_ZOGU,
    AI_COL,
    f"AI predicted — {ZONE_LABELS[ZONE_ZOGU]}",
    "(d)"
)

plt.subplots_adjust(
    left=0.07,
    right=0.88,
    bottom=0.085,
    top=0.91,
    wspace=0.16,
    hspace=0.24
)

# =========================
# COMMON COLORBAR
# =========================

cbar_ax = fig.add_axes([0.905, 0.22, 0.025, 0.58])
cbar = fig.colorbar(cf1, cax=cbar_ax)

cbar.set_label(
    "Electric field strength, total E-field (V/m)",
    fontsize=13,
    rotation=90,
    labelpad=16
)

cbar.ax.tick_params(labelsize=10)

fig.text(
    0.5,
    0.035,
    "Black points indicate measurement locations. The same color scale is used for all panels.",
    ha="center",
    fontsize=11,
    color="dimgray"
)

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