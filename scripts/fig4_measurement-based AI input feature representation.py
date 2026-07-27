from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
DATASET_CSV = PROJECT_ROOT / "data" / "urban_rf_emf_ai_dataset_corrected.csv"


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

INPUT_CSV = str(DATASET_CSV)
OUTPUT_DIR = r"str(PROJECT_ROOT)\outputs\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

zone_names = {
    "21_Dhjetori": "Mustafa Q. Atatürk Square",
    "Zogu_i_Zi": "Karl Topia Square"
}

output_names = {
    "21_Dhjetori": "Figure_04a_AI_Input_Feature_Maps_Mustafa_Q_Ataturk_IDW",
    "Zogu_i_Zi": "Figure_04b_AI_Input_Feature_Maps_Karl_Topia_IDW"
}

features = [
    ("Measurement points", None, ""),
    ("Distance to nearest antenna", "Distance_to_nearest_antenna_m", "m"),
    ("Antenna height", "Nearest_Antenna_Height_m", "m"),
    ("LTE700", "LTE700", "V/m"),
    ("LTE800", "LTE800", "V/m"),
    ("GSM900", "GSM900", "V/m"),
    ("GSM1800", "GSM1800", "V/m"),
    ("LTE2100", "LTE2100", "V/m"),
    ("LTE2600", "LTE2600", "V/m"),
    ("NR3500", "NR3500", "V/m"),
    ("Total E", "E_total", "V/m")
]

# --------------------------------------------------
# Global color limits per feature
# same scale for both zones
# --------------------------------------------------

global_limits = {}

for title, col, unit in features:
    if col is not None:
        vmin = df[col].min()
        vmax = df[col].max()

        if unit == "V/m":
            vmin = 0

        global_limits[col] = (vmin, vmax)

# --------------------------------------------------
# IDW interpolation function
# --------------------------------------------------

def idw_interpolation(x, y, values, grid_x, grid_y, power=2):
    x = x.reshape(-1, 1, 1)
    y = y.reshape(-1, 1, 1)
    values = values.reshape(-1, 1, 1)

    dist = np.sqrt((grid_x - x) ** 2 + (grid_y - y) ** 2)
    dist[dist == 0] = 1e-12

    weights = 1 / (dist ** power)
    interpolated = np.sum(weights * values, axis=0) / np.sum(weights, axis=0)

    return interpolated

# --------------------------------------------------
# Make figure per zone
# --------------------------------------------------

def make_feature_maps(zone_key):
    zone_df = df[df["Zone"] == zone_key].copy()

    lon = zone_df["Longitude"].to_numpy()
    lat = zone_df["Latitude"].to_numpy()

    lon_margin = (lon.max() - lon.min()) * 0.10
    lat_margin = (lat.max() - lat.min()) * 0.10

    lon_min = lon.min() - lon_margin
    lon_max = lon.max() + lon_margin
    lat_min = lat.min() - lat_margin
    lat_max = lat.max() + lat_margin

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, 220),
        np.linspace(lat_min, lat_max, 220)
    )

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.ravel()

    for ax, (title, col, unit) in zip(axes, features):

        if col is None:
            ax.scatter(
                lon,
                lat,
                s=28,
                color="#1f77b4",
                edgecolor="black",
                linewidth=0.4
            )

            ax.set_title(title, fontsize=11, fontweight="bold")

        else:
            values = zone_df[col].to_numpy()

            grid_values = idw_interpolation(
                lon,
                lat,
                values,
                grid_lon,
                grid_lat,
                power=2
            )

            if unit == "V/m":
                grid_values = np.clip(grid_values, 0, None)

            vmin, vmax = global_limits[col]

            im = ax.imshow(
                grid_values,
                extent=(lon_min, lon_max, lat_min, lat_max),
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="auto"
            )

            ax.scatter(
                lon,
                lat,
                s=6,
                color="black",
                alpha=0.45
            )

            cbar = plt.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04
            )
            cbar.set_label(unit, fontsize=8)
            cbar.ax.tick_params(labelsize=8)

            ax.set_title(title, fontsize=11, fontweight="bold")

        ax.set_xticks([])
        ax.set_yticks([])

    axes[-1].axis("off")

    fig.suptitle(
        f"AI Input Feature Maps - {zone_names[zone_key]}",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = os.path.join(OUTPUT_DIR, output_names[zone_key] + ".png")
    tif_path = os.path.join(OUTPUT_DIR, output_names[zone_key] + ".tif")

    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08
    )

    plt.close()

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

    print("Saved:")
    print(png_path)
    print(tif_path)

# --------------------------------------------------
# Run
# --------------------------------------------------

make_feature_maps("21_Dhjetori")
make_feature_maps("Zogu_i_Zi")