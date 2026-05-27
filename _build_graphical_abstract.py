"""
_build_graphical_abstract.py
============================
Render the JCP/Elsevier graphical abstract from existing paper figures.

Spec (Elsevier):
  - Single landscape image, minimum 531 px high x 1328 px wide
  - Captures the central methodological contribution at a glance
  - Reader should not need to open the paper to understand the message
  - TIFF or PNG output, 300+ dpi

Compositional layout (landscape, 3-zone, left-to-right):
  +----------------------------------------------------------+
  |  INPUT          ->     METHOD     ->     OUTPUT          |
  |  Sparse PIV            PINN              Full NS state   |
  |  (|V| or vector)       (NS residual      (u, v, p, WSS)  |
  |                         + data loss)                     |
  +----------------------------------------------------------+
  |  Two-benchmark validation row:                           |
  |  [Nozzle field PIV/PINN]  |  [Pump scatter 5 conditions] |
  +----------------------------------------------------------+
  |  Headline numbers:                                       |
  |  rel L^2 < 1.7%  *  E_g = 0.14 (beats best CFD 2.5x)     |
  |  *  11-26 min per case on free Colab T4 GPU              |
  +----------------------------------------------------------+
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "submission_tiffs"
OUT_DIR.mkdir(exist_ok=True)

# Target dimensions: 2400 x 1200 px (well above 1328x531 minimum,
# at 300 dpi this is 8.0 x 4.0 inches landscape).
# Made taller than first draft so the panel titles don't overlap the
# top banner row.
FIG_W_IN = 8.0
FIG_H_IN = 4.0
DPI = 300


def imread_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im)


fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor="white")

# 3-row layout:
#   Row 0 (small): conceptual flow banner (Input -> Method -> Output)
#   Row 1 (large): two-panel validation row (nozzle + pump)
#   Row 2 (small): headline numbers bar
gs = GridSpec(3, 6, figure=fig,
              height_ratios=[0.55, 3.2, 0.45],
              width_ratios=[1, 1, 1, 1, 1, 1],
              hspace=0.45, wspace=0.30,
              left=0.02, right=0.98, top=0.97, bottom=0.05)

# ── Row 0: conceptual flow banner ────────────────────────────────
ax_banner = fig.add_subplot(gs[0, :])
ax_banner.set_xlim(0, 10); ax_banner.set_ylim(0, 1)
ax_banner.axis("off")

# Three rounded boxes + two arrows
box_y, box_h = 0.05, 0.9
def add_box(x0, x1, label, sub, color):
    rect = mpatches.FancyBboxPatch(
        (x0, box_y), x1 - x0, box_h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        edgecolor=color, facecolor=color + "20",
        linewidth=1.5, transform=ax_banner.transData,
    )
    ax_banner.add_patch(rect)
    cx = (x0 + x1) / 2
    ax_banner.text(cx, box_y + box_h * 0.65, label,
                   ha="center", va="center",
                   fontsize=11, fontweight="bold", color=color)
    ax_banner.text(cx, box_y + box_h * 0.28, sub,
                   ha="center", va="center",
                   fontsize=8, color="#333333")

def add_arrow(x0, x1):
    ax_banner.annotate(
        "", xy=(x1, 0.5), xytext=(x0, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#444444"),
    )

add_box(0.1, 2.9, "Sparse PIV input",
        "|V| or (u_x, u_r), 4,000 samples", "#2c7fb8")
add_arrow(2.95, 3.55)
add_box(3.6, 6.4, "PINN",
        "Navier-Stokes residual + data loss", "#7b3294")
add_arrow(6.45, 7.05)
add_box(7.1, 9.9, "Full Navier-Stokes state",
        "u(x), p(x), WSS(x) recovered", "#1b9e77")

# ── Row 1: two validation panels ─────────────────────────────────
# Left: nozzle Re=500 fields (PIV vs PINN)
ax_left = fig.add_subplot(gs[1, 0:3])
im_left = imread_rgb(FIG_DIR / "fields_Re500.png")
ax_left.imshow(im_left, aspect="auto"); ax_left.axis("off")
ax_left.set_title("FDA nozzle, Re = 500   (rel L$^2$ = 7.5%)",
                  fontsize=10, pad=4)

# Right: pump scatter summary (5 conditions)
ax_right = fig.add_subplot(gs[1, 3:6])
im_right = imread_rgb(FIG_DIR / "pump_scatter_summary.png")
ax_right.imshow(im_right, aspect="auto"); ax_right.axis("off")
ax_right.set_title("FDA Blood Pump, 5 conditions   "
                   "(rel L$^2$ < 1.7%)",
                   fontsize=10, pad=4)

# ── Row 2: headline numbers ──────────────────────────────────────
ax_num = fig.add_subplot(gs[2, :])
ax_num.set_xlim(0, 1); ax_num.set_ylim(0, 1); ax_num.axis("off")
ax_num.text(
    0.5, 0.5,
    r"$E_g = 0.14$ at Re=500  (2.5$\times$ smaller than best CFD "
    r"in FDA 28-lab study) "
    r"$\;\bullet\;$  11-26 min per case on free Colab T4 GPU  "
    r"$\;\bullet\;$  5-30$\times$ CFD speed-up",
    ha="center", va="center",
    fontsize=10, color="#222222",
    bbox=dict(boxstyle="round,pad=0.3",
              facecolor="#f0f0f0", edgecolor="#999999", lw=0.8),
)

# Save
out_path = OUT_DIR / "graphical_abstract.tif"
fig.savefig(out_path, format="tiff", dpi=DPI,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight", facecolor="white")
fig.savefig(OUT_DIR / "graphical_abstract.png", dpi=DPI,
            bbox_inches="tight", facecolor="white")

w_px, h_px = int(FIG_W_IN * DPI), int(FIG_H_IN * DPI)
print(f"  graphical_abstract.tif: ~{w_px}x{h_px} px @ {DPI} dpi")
print(f"  graphical_abstract.png: same as preview")
print(f"  Saved to {OUT_DIR}")
