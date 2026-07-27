from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
DATASET_CSV = PROJECT_ROOT / "data" / "urban_rf_emf_ai_dataset_corrected.csv"


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

INPUT_CSV = str(DATASET_CSV)
OUTPUT_DIR = r"str(PROJECT_ROOT)\outputs\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(OUTPUT_DIR, "Figure_03_Boxplot_RF_EMF_Distribution.png")
OUTPUT_TIF = os.path.join(OUTPUT_DIR, "Figure_03_Boxplot_RF_EMF_Distribution.tif")

df = pd.read_csv(INPUT_CSV)

df["Zone"] = df["Zone"].replace({
    "21_Dhjetori": "Mustafa Q. Atatürk Sq.",
    "21 Dhjetori": "Mustafa Q. Atatürk Sq.",
    "Sheshi Mustafa Qemal Atatürk": "Mustafa Q. Atatürk Sq.",
    "Zogu_i_Zi": "Karl Topia Sq.",
    "Zogu i Zi": "Karl Topia Sq.",
    "Sheshi Karl Topia": "Karl Topia Sq."
})

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

band_labels = {
    "E_total": "Total E"
}

df_long = df.melt(
    id_vars=["Zone"],
    value_vars=bands,
    var_name="Frequency band",
    value_name="Electric field strength (V/m)"
)

df_long["Frequency band"] = df_long["Frequency band"].replace(band_labels)

plt.figure(figsize=(13.5, 6.2))

sns.boxplot(
    data=df_long,
    x="Frequency band",
    y="Electric field strength (V/m)",
    hue="Zone",
    width=0.62,
    linewidth=1.1,
    showfliers=True,
    fliersize=3
)

plt.xlabel("Frequency bands", fontsize=13)
plt.ylabel("Electric field strength (V/m)", fontsize=13)

plt.title(
    "Distribution of RF-EMF Exposure Across Frequency Bands",
    fontsize=15,
    fontweight="bold",
    pad=12
)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.legend(
    title="Urban area",
    fontsize=10.5,
    title_fontsize=11,
    loc="upper left",
    frameon=True
)

plt.grid(axis="y", linestyle="--", alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)

plt.close()

img = Image.open(OUTPUT_PNG)
img.save(
    OUTPUT_PNG,
    optimize=True,
    compress_level=9
)

img = Image.open(OUTPUT_PNG).convert("RGB")
img.save(
    OUTPUT_TIF,
    compression="tiff_lzw",
    dpi=(600, 600)
)

print("Done.")
print("PNG saved at:", OUTPUT_PNG)
print("TIF saved at:", OUTPUT_TIF)