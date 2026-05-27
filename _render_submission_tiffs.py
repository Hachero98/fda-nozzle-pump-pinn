"""
_render_submission_tiffs.py
===========================
Render the 7 paper figures as journal-grade TIFFs for submission to
Nature Communications / Nature Computational Science.

Output specs (Nature-compliant):
  - TIFF format with LZW compression
  - 600 dpi (set in EXIF metadata for the journal pipeline)
  - RGB color mode (no alpha)
  - Single TIFF per figure (multi-panel figures composed via vertical
    stacking with thin separator)

Figure manifest:
  Fig1  fields_Re500 + centreline_Re500    (Re=500 vector + centerline)
  Fig2  wss_Re500 + Re2000 + Re3500        (3-panel WSS)
  Fig3  centreline_Re2000 + Re3500         (2-panel centerline at higher Re)
  Fig4  cfd_vs_pinn_summary                (CFD vs PINN comparison)
  Fig5  summary                            (cross-Re summary)
  Fig6  pump_fields_C5                     (pump diffuser C5)
  Fig7  pump_scatter_summary               (pump 5-condition scatter)
"""

from pathlib import Path
from PIL import Image

ROOT = Path(__file__).parent
SRC  = ROOT / "figures"
OUT  = ROOT / "submission_tiffs"
OUT.mkdir(exist_ok=True)

DPI = 600
SEPARATOR_PX = 20    # white gap between stacked panels


def open_rgb(name):
    """Open PNG, flatten RGBA -> RGB on white background."""
    im = Image.open(SRC / name)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


def stack_vertical(images):
    """Stack images vertically, centering narrower ones, white separator."""
    max_w = max(im.width for im in images)
    total_h = sum(im.height for im in images) + SEPARATOR_PX * (len(images) - 1)
    canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for im in images:
        x = (max_w - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height + SEPARATOR_PX
    return canvas


def save_tiff(im, out_path):
    im.save(
        out_path,
        format="TIFF",
        compression="tiff_lzw",
        dpi=(DPI, DPI),
    )
    kb = out_path.stat().st_size / 1024
    print(f"  {out_path.name}: {im.size[0]}x{im.size[1]} px  "
          f"{kb:.0f} KB  @ {DPI} dpi")


# ─────────────────────────────────────────────────────────────────────
# Fig 1: Re=500 fields (vector) + centerline (line plot), stacked
# ─────────────────────────────────────────────────────────────────────
print("Rendering Fig1 (Re=500 fields + centerline)...")
fig1 = stack_vertical([
    open_rgb("fields_Re500.png"),
    open_rgb("centreline_Re500.png"),
])
save_tiff(fig1, OUT / "Fig1.tif")

# ─────────────────────────────────────────────────────────────────────
# Fig 2: WSS at Re=500, 2000, 3500 (3-panel vertical stack)
# ─────────────────────────────────────────────────────────────────────
print("Rendering Fig2 (WSS triple panel)...")
fig2 = stack_vertical([
    open_rgb("wss_Re500.png"),
    open_rgb("wss_Re2000.png"),
    open_rgb("wss_Re3500.png"),
])
save_tiff(fig2, OUT / "Fig2.tif")

# ─────────────────────────────────────────────────────────────────────
# Fig 3: Centerline at Re=2000, 3500 (2-panel)
# ─────────────────────────────────────────────────────────────────────
print("Rendering Fig3 (centerline at Re=2000, 3500)...")
fig3 = stack_vertical([
    open_rgb("centreline_Re2000.png"),
    open_rgb("centreline_Re3500.png"),
])
save_tiff(fig3, OUT / "Fig3.tif")

# ─────────────────────────────────────────────────────────────────────
# Fig 4: CFD vs PINN comparison (single)
# ─────────────────────────────────────────────────────────────────────
print("Rendering Fig4 (CFD vs PINN)...")
save_tiff(open_rgb("cfd_vs_pinn_summary.png"), OUT / "Fig4.tif")

# ─────────────────────────────────────────────────────────────────────
# Fig 5: Cross-Re summary (single)
# ─────────────────────────────────────────────────────────────────────
print("Rendering Fig5 (cross-Re summary)...")
save_tiff(open_rgb("summary.png"), OUT / "Fig5.tif")

# ─────────────────────────────────────────────────────────────────────
# Fig 6: Pump diffuser C5 (PIV/PINN/error map)
# ─────────────────────────────────────────────────────────────────────
print("Rendering Fig6 (pump diffuser C5)...")
save_tiff(open_rgb("pump_fields_C5.png"), OUT / "Fig6.tif")

# ─────────────────────────────────────────────────────────────────────
# Fig 7: Pump 5-condition scatter
# ─────────────────────────────────────────────────────────────────────
print("Rendering Fig7 (pump scatter)...")
save_tiff(open_rgb("pump_scatter_summary.png"), OUT / "Fig7.tif")

print(f"\nAll 7 TIFFs written to: {OUT}")
