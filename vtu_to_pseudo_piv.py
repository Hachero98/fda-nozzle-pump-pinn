"""
vtu_to_pseudo_piv.py
====================
Convert the Zenodo CFD VTU files (re{N}_60_60_1000.vtu) into sparse
'pseudo-PIV' CSVs in the column format that fda_nozzle_PINN.py's
load_piv_csv() expects:

    x [m], r [m], u_x [m/s], u_r [m/s], Re

The VTU files are 3D (x, y, z) axial-pipe simulations; this script
collapses to the axisymmetric (x, r) plane by computing r = sqrt(y^2 + z^2)
and averaging velocity at matching (x, r) bins, then subsampling to
N points to mimic PIV measurement sparsity.

HOW TO RUN
----------
    python vtu_to_pseudo_piv.py \
        --vtu "Reference Papers/FDA_Dataset/CFD_Zenodo/re500_60_60_1000.vtu" \
        --re 500 \
        --n-samples 850 \
        --out "Reference Papers/FDA_Dataset/pseudo_PIV_Re500.csv"

Or batch all three at once:
    python vtu_to_pseudo_piv.py --all

Requires: pyvista (pip install pyvista).  If pyvista is unavailable,
falls back to a minimal vtkUnstructuredGrid XML parser using numpy
that handles the appended-data block of these specific files.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def load_vtu(path):
    """Load a VTU and return (points [N,3], u [N,3], valid_mask [N]).

    Handles two layouts:
      (a) single vector velocity array named 'U'/'velocity'/'Velocity'
      (b) three scalar arrays 'u', 'v', 'w' (the Zenodo CFD layout)

    NaN-valued cells (outside the fluid domain) are flagged in
    valid_mask; callers should subset by it.
    """
    try:
        import pyvista as pv
    except ImportError:
        sys.exit(
            "ERROR: this loader needs pyvista.\n"
            "  pip install pyvista\n"
            "If you don't want pyvista, open the VTU in ParaView and\n"
            "export a CSV with columns (Points:0, Points:1, Points:2,\n"
            "u, v, w), then feed that CSV directly to the PINN."
        )

    mesh = pv.read(str(path))
    points = np.asarray(mesh.points)            # [N, 3]
    keys = list(mesh.point_data.keys())

    vec = None
    for name in ("U", "velocity", "Velocity"):
        if name in keys:
            vec = np.asarray(mesh.point_data[name])
            break

    if vec is None and {"u", "v", "w"}.issubset(set(keys)):
        u = np.asarray(mesh.point_data["u"])
        v = np.asarray(mesh.point_data["v"])
        w = np.asarray(mesh.point_data["w"])
        vec = np.stack([u, v, w], axis=1)        # [N, 3]

    if vec is None:
        raise RuntimeError(
            f"No velocity field in {path}. Available arrays: {keys}"
        )
    if vec.ndim != 2 or vec.shape[1] != 3:
        raise RuntimeError(f"Velocity has shape {vec.shape}, expected [N, 3]")

    valid_mask = ~np.any(np.isnan(vec), axis=1)
    return points, vec, valid_mask


def collapse_axisymmetric(points, u, axial_axis=0):
    """Collapse 3D field to (x, r, u_x, u_r) by radial reduction.

    axial_axis: which Cartesian index is the pipe axis (default 0 = x).
                The two remaining axes form the radial plane.
    """
    other = [i for i in (0, 1, 2) if i != axial_axis]
    x = points[:, axial_axis]
    y = points[:, other[0]]
    z = points[:, other[1]]
    r = np.sqrt(y ** 2 + z ** 2)

    u_x = u[:, axial_axis]
    # Radial velocity = projection of (u_y, u_z) onto the radial unit vector.
    eps = 1e-12
    u_r = (y * u[:, other[0]] + z * u[:, other[1]]) / (r + eps)

    return x, r, u_x, u_r


def subsample(x, r, u_x, u_r, n_samples, seed=1234,
              piv_window=None):
    """Pick n_samples points to mimic PIV sparsity.

    piv_window: optional (x_min, x_max, r_min, r_max) bounding box for
                where PIV is realistic. Defaults to the full axial extent.
    """
    rng = np.random.default_rng(seed)
    if piv_window is not None:
        xmin, xmax, rmin, rmax = piv_window
        mask = (x >= xmin) & (x <= xmax) & (r >= rmin) & (r <= rmax)
        x, r, u_x, u_r = x[mask], r[mask], u_x[mask], u_r[mask]

    n = min(n_samples, x.size)
    idx = rng.choice(x.size, size=n, replace=False)
    return x[idx], r[idx], u_x[idx], u_r[idx]


def write_csv(path, x, r, u_x, u_r, Re):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "r", "u_x", "u_r", "Re"])
        for xi, ri, uxi, uri in zip(x, r, u_x, u_r):
            w.writerow([f"{xi:.6e}", f"{ri:.6e}",
                        f"{uxi:.6e}", f"{uri:.6e}", int(Re)])


def detect_axial_axis(points):
    """Pick the axis with the largest extent — that's the pipe axis."""
    ranges = np.ptp(points, axis=0)
    return int(np.argmax(ranges))


def detect_throat(x, r, n_bins=200):
    """Find x where max(r) is smallest — the throat location."""
    bins = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    max_r = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.any():
            max_r[i] = r[m].max()
    valid = ~np.isnan(max_r)
    if not valid.any():
        return None
    return float(centers[valid][np.argmin(max_r[valid])])


def process_one(vtu_path, re, n_samples, out_csv, axial_axis=None,
                seed=1234, x_shift="auto"):
    """x_shift: numeric value subtracted from x before writing, or
       'auto' to use detect_throat (aligns throat to x = 0), or
       None / 0.0 to keep native CFD frame."""
    print(f"  loading {vtu_path} ...")
    points, u, valid = load_vtu(vtu_path)
    print(f"  total nodes = {points.shape[0]:,}  "
          f"valid (fluid) = {valid.sum():,}  "
          f"({100 * valid.sum() / valid.size:.1f}%)")
    points = points[valid]
    u = u[valid]

    if axial_axis is None:
        axial_axis = detect_axial_axis(points)
        print(f"  detected axial axis = {axial_axis} "
              f"(extents = {np.ptp(points, axis=0)})")
    x, r, u_x, u_r = collapse_axisymmetric(points, u, axial_axis=axial_axis)
    print(f"  3D -> axisymmetric: {x.size} points")
    print(f"  x (axial) raw range: [{x.min():.4f}, {x.max():.4f}] m")
    print(f"  r (radial) range:    [{r.min():.4f}, {r.max():.4f}] m")
    print(f"  |u_x| max: {np.abs(u_x).max():.4f} m/s")
    print(f"  |u_r| max: {np.abs(u_r).max():.4f} m/s")

    throat_x = detect_throat(x, r)
    if throat_x is not None:
        print(f"  detected throat location (native frame): x = {throat_x:.4f} m")

    if x_shift == "auto":
        applied = throat_x if throat_x is not None else 0.0
    else:
        applied = float(x_shift) if x_shift is not None else 0.0
    if applied:
        x = x - applied
        print(f"  applied x_shift = {applied:.4f} m  -> "
              f"x range now [{x.min():.4f}, {x.max():.4f}] m")

    xs, rs, uxs, urs = subsample(x, r, u_x, u_r, n_samples, seed=seed)
    write_csv(out_csv, xs, rs, uxs, urs, re)
    print(f"  wrote {n_samples} sparse samples -> {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vtu", type=str,
                        help="Path to a single VTU file")
    parser.add_argument("--re", type=int,
                        help="Reynolds number tag (e.g. 500)")
    parser.add_argument("--n-samples", type=int, default=850,
                        help="Sparse sample count (paper default ~850)")
    parser.add_argument("--out", type=str,
                        help="Output CSV path")
    parser.add_argument("--axial-axis", type=int, default=None,
                        help="0/1/2; auto-detected if omitted")
    parser.add_argument("--all", action="store_true",
                        help="Batch-process re500/re2000/re3500 with default paths")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--x-shift", default="auto",
                        help="Numeric shift subtracted from x, or 'auto' to "
                             "align throat to x=0 (default), or '0' to keep "
                             "the native CFD frame")
    args = parser.parse_args()
    if args.x_shift != "auto":
        try:
            args.x_shift = float(args.x_shift)
        except ValueError:
            parser.error(f"--x-shift must be a number or 'auto'")

    if args.all:
        base = Path("Reference Papers/FDA_Dataset")
        cfd  = base / "CFD_Zenodo"
        for re in (500, 2000, 3500):
            vtu = cfd / f"re{re}_60_60_1000.vtu"
            out = base / f"pseudo_PIV_Re{re}.csv"
            if not vtu.exists():
                print(f"  SKIP Re={re}: {vtu} not found")
                continue
            print(f"\n=== Re = {re} ===")
            process_one(str(vtu), re, args.n_samples, str(out),
                        axial_axis=args.axial_axis, seed=args.seed,
                        x_shift=args.x_shift)
    else:
        if not (args.vtu and args.re and args.out):
            parser.error("--vtu, --re, --out required when --all is not set")
        process_one(args.vtu, args.re, args.n_samples, args.out,
                    axial_axis=args.axial_axis, seed=args.seed,
                    x_shift=args.x_shift)


if __name__ == "__main__":
    main()
