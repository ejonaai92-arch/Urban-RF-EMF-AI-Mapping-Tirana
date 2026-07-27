import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

# ==================================================
# PATHS
# ==================================================

OUTPUT_DIR = r"C:\Users\Dori\Desktop\AI_EMF\outputs\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure_06_Dataset_Construction_Workflow.png"
)

OUTPUT_TIF = os.path.join(
    OUTPUT_DIR,
    "Figure_06_Dataset_Construction_Workflow.tif"
)

# ==================================================
# STYLE
# ==================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11

# ==================================================
# FIGURE SETUP
# ==================================================

fig, ax = plt.subplots(figsize=(8.2, 9.2))

ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")

# ==================================================
# BOX DEFINITIONS
# ==================================================

boxes = [
    {
        "title": "Raw RF-EMF Measurements",
        "body": "171 measurement points\nMustafa Q. Atatürk Square and Karl Topia Square",
        "y": 12.0
    },
    {
        "title": "Frequency-Band Features",
        "body": "LTE700, LTE800, GSM900,\nGSM1800, LTE2100,\nLTE2600, NR3500",
        "y": 9.4
    },
    {
        "title": "Spatial Features",
        "body": "Latitude, longitude,\ndistance to nearest antenna,\nnearest antenna height",
        "y": 6.8
    },
    {
        "title": "Total Exposure Computation",
        "body": "Total RF-EMF exposure\ncalculated using the RSS method",
        "y": 4.2
    },
    {
        "title": "Final RF-EMF Dataset",
        "body": "171 measurement points\n11 input features\nTarget: total RF-EMF exposure",
        "y": 1.6
    }
]

box_x = 1.6
box_w = 6.8
box_h = 1.55

# ==================================================
# DRAW BOXES
# ==================================================

for box in boxes:
    y = box["y"]

    rect = FancyBboxPatch(
        (box_x, y),
        box_w,
        box_h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.4,
        edgecolor="black",
        facecolor="#e6e6e6"
    )

    ax.add_patch(rect)

    ax.text(
        box_x + box_w / 2,
        y + 1.05,
        box["title"],
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold"
    )

    ax.text(
        box_x + box_w / 2,
        y + 0.45,
        box["body"],
        ha="center",
        va="center",
        fontsize=10.5
    )

# ==================================================
# DRAW ARROWS
# ==================================================

for i in range(len(boxes) - 1):
    y_start = boxes[i]["y"]
    y_end = boxes[i + 1]["y"] + box_h

    arrow = FancyArrowPatch(
        (5, y_start),
        (5, y_end),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.4,
        color="black"
    )

    ax.add_patch(arrow)

# ==================================================
# TITLE
# ==================================================

ax.set_title(
    "Dataset Construction Workflow for RF-EMF Exposure Prediction",
    fontsize=15,
    fontweight="bold",
    pad=18
)

# ==================================================
# FOOTNOTE
# ==================================================

ax.text(
    5,
    0.55,
    "The final dataset was used for machine-learning-based total RF-EMF exposure prediction.",
    ha="center",
    va="center",
    fontsize=9.5,
    style="italic"
)

plt.tight_layout()

# ==================================================
# SAVE PNG
# ==================================================

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.08
)

plt.close(fig)

img = Image.open(OUTPUT_PNG)
img.save(
    OUTPUT_PNG,
    optimize=True,
    compress_level=9
)

# ==================================================
# SAVE TIF USING PIL
# ==================================================

img = Image.open(OUTPUT_PNG).convert("RGB")
img.save(
    OUTPUT_TIF,
    compression="tiff_lzw",
    dpi=(600, 600)
)

print("Saved:")
print(OUTPUT_PNG)
print(OUTPUT_TIF)