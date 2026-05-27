"""
parse_fda_piv.py
================
Parse the official FDA OSEL Hariharan 2011 PIV files into the CSV
format that fda_nozzle_PINN.py expects.

Source: github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks
        doi:10.17917/C78G69
        Authors: Hariharan & Malinauskas, FDA (Round Robin 1, v2.0)
        Public domain (17 U.S.C. § 105)

File format (per Hariharan 2011 conformance-converted .txt):

    dataset-*   metadata
    geometry-*  metadata
    fluid-*     metadata

    plot-wall-distribution-pressure
    <count>
    <z>  <p>
    ...

    plot-wall-distribution-wall-shear-stress
    <count>
    <z>  <tau_w>
    ...

    plot-z-distribution-axial-velocity
    <count>
    <z>  <u_z_centerline>
    ...

    plot-profile-axial-velocity-at-z  <z_location>  0
    <count>
    <r>  <u_x>
    ...

    plot-profile-radial-velocity-at-z  <z_location>  0
    <count>
    <r>  <u_r>
    ...

OUTPUTS — per Reynolds number, two CSVs:

    PIV_Re{N}.csv          full point cloud (x, r, u_x, u_r, Re)
                            — fed to fda_nozzle_PINN.py via --piv-csv
    PIV_wall_Re{N}.csv     wall measurements (x, p_wall, tau_w, Re)
                            — used for validation of PINN WSS predictions

When axial and radial velocity profiles share the same (x, r) location,
they are merged into a single row. Otherwise, missing components are
written as NaN and the PINN loader filters them out.

HOW TO RUN
----------
    pip install numpy
    python parse_fda_piv.py --all
    # or per-Re:
    python parse_fda_piv.py --zip SE_exp_0500.zip --re 500 \
        --out "Reference Papers/FDA_Dataset/PIV_Re500.csv"
"""

import argparse
import csv
import os
import re as _re
import sys
import zipfile
from pathlib import Path

import numpy as np


# x-shift to align PIV frame to scaffold frame.
# PIV files set geometry-sudden-z = 0.0 (sudden expansion at z=0); the scaffold
# puts the sudden expansion at X_THROAT_END = 0.040. So scaffold_x = PIV_x +
# 0.040 m, i.e., DEFAULT_X_SHIFT = -0.040 (we *subtract* this from PIV x).
DEFAULT_X_SHIFT = -0.040

# Scaffold geometry: piecewise wall radius (mirrors fda_nozzle_PINN.py).
# Used to drop PIV rows that fall outside the lumen (|r| > R(x) + eps).
R_THROAT = 0.002
R_INLET  = 0.006
X_INLET_START    = -0.088
X_CONTRACT_START = -0.022675
X_THROAT_START   = 0.0
X_THROAT_END     = 0.040
X_OUTLET_END     = 0.143
R_OUT_TOL        = 5e-4   # 0.5 mm leniency on the lumen boundary

def _wall_radius(x):
    """Same piecewise R(x) as the scaffold (scalar version, numpy-safe)."""
    if x < X_CONTRACT_START:
        return R_INLET
    if x < X_THROAT_START:
        frac = (x - X_CONTRACT_START) / (X_THROAT_START - X_CONTRACT_START)
        return R_INLET + (R_THROAT - R_INLET) * np.clip(frac, 0.0, 1.0)
    if x < X_THROAT_END:
        return R_THROAT
    return R_INLET


def _read_count(it):
    """Read the next non-empty line as an integer count."""
    for line in it:
        s = line.strip()
        if not s:
            continue
        try:
            return int(s)
        except ValueError:
            raise RuntimeError(f"expected integer count, got: {s!r}")
    raise RuntimeError("unexpected EOF while reading count")


def _read_n_xy(it, n):
    """Read n lines of 'pos  value' (whitespace-separated) into two arrays."""
    pos = []
    val = []
    while len(pos) < n:
        try:
            line = next(it)
        except StopIteration:
            raise RuntimeError(f"EOF after {len(pos)} of {n} rows")
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 2:
            raise RuntimeError(f"bad data line: {s!r}")
        try:
            pos.append(float(parts[0]))
            val.append(float(parts[1]))
        except ValueError:
            raise RuntimeError(f"non-numeric data line: {s!r}")
    return np.array(pos), np.array(val)


def parse_piv_file(path):
    """Parse one Hariharan .txt and return a dict of sections.

    Returns:
        meta       dict of dataset/geometry/fluid metadata
        wall       dict with keys 'pressure', 'shear_stress' -> (z, value)
        centerline dict with keys 'pressure', 'axial_velocity',
                                  'reynolds_stress' -> (z, value)
        profiles   dict { (component, z): (r, value) } where component is
                   one of 'u_x', 'u_r', 'shear_stress', 'reynolds_stress'
    """
    meta = {}
    wall = {}
    centerline = {}
    profiles = {}

    component_map = {
        "profile-axial-velocity-at-z":   "u_x",
        "profile-radial-velocity-at-z":  "u_r",
        "profile-shear-stress-at-z":     "shear_stress",
        "profile-reynolds-stress-at-z":  "reynolds_stress",
    }
    centerline_map = {
        "z-distribution-pressure":        "pressure",
        "z-distribution-axial-velocity":  "axial_velocity",
        "z-distribution-reynolds-stress": "reynolds_stress",
    }
    wall_map = {
        "wall-distribution-pressure":         "pressure",
        "wall-distribution-wall-shear-stress": "shear_stress",
    }

    with open(path) as fh:
        lines = iter(fh.readlines())

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith(("dataset-", "geometry-", "fluid-")):
            tag, _, rest = s.partition(" ")
            meta[tag] = rest.strip().strip('"')
            continue
        if not s.startswith("plot-"):
            continue

        body = s[5:]  # strip 'plot-'

        # Wall distributions
        wall_key = next((k for k in wall_map if body == k), None)
        if wall_key:
            n = _read_count(lines)
            z, v = _read_n_xy(lines, n)
            wall[wall_map[wall_key]] = (z, v)
            continue

        # Centerline z-distributions
        cl_key = next((k for k in centerline_map if body == k), None)
        if cl_key:
            n = _read_count(lines)
            z, v = _read_n_xy(lines, n)
            centerline[centerline_map[cl_key]] = (z, v)
            continue

        # Radial profiles at a fixed z
        prof_key = next((k for k in component_map if body.startswith(k)), None)
        if prof_key:
            tail = body[len(prof_key):].strip()
            # tail looks like "-0.08800 0" — first number is z_location
            parts = tail.split()
            if not parts:
                raise RuntimeError(f"profile line missing z: {s!r}")
            z_loc = float(parts[0])
            n = _read_count(lines)
            r, v = _read_n_xy(lines, n)
            comp = component_map[prof_key]
            profiles[(comp, z_loc)] = (r, v)
            continue

        # plot-jet-width-0 and other one-offs: skip body
        # (read count and discard if a count exists)
        try:
            n = _read_count(lines)
            _, _ = _read_n_xy(lines, n)
        except RuntimeError:
            pass

    return meta, wall, centerline, profiles


def build_point_cloud(profiles, x_shift=0.0,
                      fold_to_positive_r=True, drop_outside_lumen=True):
    """Merge axial + radial velocity profiles into (x, r, u_x, u_r) rows.

    Rows where both u_x and u_r are present at the same (x, r) are merged.
    Rows with only one component are written with NaN for the other.

    fold_to_positive_r:
        PIV measured across the full diameter (signed r). Axisymmetry says
        u_x(x, -r) = u_x(x, r) and u_r(x, -r) = -u_r(x, r). We take |r| and
        flip the sign of u_r on the negative-r side, doubling the effective
        sample count.

    drop_outside_lumen:
        Far-upstream PIV rows extend past the wall (|r| > R_INLET). They
        report u = 0 (outside the imaging region); they contain no fluid
        information. Drop them so they do not anchor the network to zero.
    """
    bucket = {}   # (z, r_signed_rounded) -> {'u_x': v, 'u_r': v}
    R_TOL = 1e-7
    def keyf(z, r):
        return (float(z), round(float(r) / R_TOL) * R_TOL)

    for (comp, z), (rs, vs) in profiles.items():
        if comp not in ("u_x", "u_r"):
            continue
        for r, v in zip(rs, vs):
            k = keyf(z, r)
            d = bucket.setdefault(k, {"u_x": np.nan, "u_r": np.nan})
            d[comp] = float(v)

    rows = []
    n_dropped_outside = 0
    for (z, r), d in sorted(bucket.items()):
        x = z - x_shift
        ux = d["u_x"]
        ur = d["u_r"]
        if fold_to_positive_r:
            if r < 0:
                # Axisymmetry: u_x unchanged, u_r flips sign
                ur = -ur if not np.isnan(ur) else ur
            r = abs(r)
        if drop_outside_lumen and r > _wall_radius(x) + R_OUT_TOL:
            n_dropped_outside += 1
            continue
        rows.append((x, r, ux, ur))
    return rows, n_dropped_outside


def write_point_cloud_csv(rows, Re, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "r", "u_x", "u_r", "Re"])
        for x, r, ux, ur in rows:
            w.writerow([
                f"{x:.6e}", f"{r:.6e}",
                "" if np.isnan(ux) else f"{ux:.6e}",
                "" if np.isnan(ur) else f"{ur:.6e}",
                int(Re),
            ])


def write_wall_csv(wall, Re, out_path, x_shift=0.0):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    p_z, p_v = wall.get("pressure", (np.array([]), np.array([])))
    tw_z, tw_v = wall.get("shear_stress", (np.array([]), np.array([])))
    # Merge by z
    bucket = {}
    R_TOL = 1e-6
    def keyf(z):
        return round(float(z) / R_TOL) * R_TOL
    for z, p in zip(p_z, p_v):
        bucket.setdefault(keyf(z), {})["p"] = p
    for z, tw in zip(tw_z, tw_v):
        bucket.setdefault(keyf(z), {})["tau_w"] = tw
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "p_wall_Pa", "tau_w_Pa", "Re"])
        for z, d in sorted(bucket.items()):
            w.writerow([
                f"{(z - x_shift):.6e}",
                "" if "p"     not in d else f"{d['p']:.6e}",
                "" if "tau_w" not in d else f"{d['tau_w']:.6e}",
                int(Re),
            ])


def process_zip(zip_path, Re, out_dir, x_shift=0.0):
    """Extract a ZIP, parse all .txt files inside, merge across experiments."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract to a temp folder
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        txt_files = sorted(Path(tmp).rglob("*.txt"))
        if not txt_files:
            print(f"  WARNING: no .txt files in {zip_path}")
            return
        print(f"  found {len(txt_files)} experiment file(s) in {zip_path.name}")

        all_rows = []
        wall_pressure = []
        wall_tau = []
        meta_summary = None
        total_dropped = 0

        for tx in txt_files:
            meta, wall, cl, prof = parse_piv_file(tx)
            if meta_summary is None:
                meta_summary = meta
            rows, n_drop = build_point_cloud(prof, x_shift=x_shift)
            all_rows.extend(rows)
            total_dropped += n_drop
            if "pressure" in wall:
                wall_pressure.append(wall["pressure"])
            if "shear_stress" in wall:
                wall_tau.append(wall["shear_stress"])
            print(f"    {tx.name}: {len(rows)} kept "
                  f"({n_drop} dropped as outside lumen), "
                  f"code={meta.get('dataset-code', '?')}")

        # Write merged point cloud (rows from all experiments)
        cloud_csv = out_dir / f"PIV_Re{Re}.csv"
        write_point_cloud_csv(all_rows, Re, cloud_csv)
        print(f"  wrote {len(all_rows):,} rows -> {cloud_csv}")

        # Average wall distributions across experiments
        if wall_pressure or wall_tau:
            avg_wall = {}
            if wall_pressure:
                zs = wall_pressure[0][0]
                vs = np.mean(np.stack([w[1] for w in wall_pressure
                                       if w[1].size == zs.size], axis=0),
                             axis=0)
                avg_wall["pressure"] = (zs, vs)
            if wall_tau:
                zs = wall_tau[0][0]
                vs = np.mean(np.stack([w[1] for w in wall_tau
                                       if w[1].size == zs.size], axis=0),
                             axis=0)
                avg_wall["shear_stress"] = (zs, vs)
            wall_csv = out_dir / f"PIV_wall_Re{Re}.csv"
            write_wall_csv(avg_wall, Re, wall_csv, x_shift=x_shift)
            print(f"  wrote wall validation CSV   -> {wall_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str,
                        help="Single ZIP file (e.g., SE_exp_0500.zip)")
    parser.add_argument("--re", type=int,
                        help="Reynolds number (e.g., 500)")
    parser.add_argument("--out-dir", type=str,
                        default="Reference Papers/FDA_Dataset",
                        help="Where to write CSVs")
    parser.add_argument("--x-shift", type=float, default=DEFAULT_X_SHIFT,
                        help="Subtract from x (default 0; SE files are already throat-at-0)")
    parser.add_argument("--all", action="store_true",
                        help="Batch-process all 5 SE_exp_*.zip files")
    parser.add_argument("--source-dir", type=str,
                        default="Reference Papers/FDA_Dataset/FDA_OSEL_Benchmarks/Nozzle/Data",
                        help="Where the ZIP files live")
    args = parser.parse_args()

    if args.all:
        src = Path(args.source_dir)
        for re_n in (500, 2000, 3500, 5000, 6500):
            zname = f"SE_exp_{re_n:04d}.zip"
            zpath = src / zname
            if not zpath.exists():
                print(f"\n  SKIP Re={re_n}: {zpath} not found")
                continue
            print(f"\n=== Re = {re_n} ({zname}) ===")
            process_zip(zpath, re_n, args.out_dir, x_shift=args.x_shift)
    else:
        if not (args.zip and args.re):
            parser.error("--zip and --re required when --all is not set")
        process_zip(Path(args.zip), args.re, args.out_dir,
                    x_shift=args.x_shift)


if __name__ == "__main__":
    main()
