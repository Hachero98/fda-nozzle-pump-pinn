"""Generate B4_FDA_Nozzle_PINN_Colab.ipynb from clean cell definitions."""
import json
from pathlib import Path

CELLS = []

def md(text):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })

def code(text):
    CELLS.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


# =====================================================================
# CELL 1 — Title / overview
# =====================================================================
md("""# B4 — FDA Nozzle PINN: Colab Training (Re = 500, 2000, 3500)

Physics-Informed Neural Network reconstruction of full-field hemodynamics
on the FDA Critical Path nozzle benchmark, using the official Hariharan
et al. 2011 PIV dataset.

**Author:** Emmanuel Hackman, USM PhD Computational Science
**Dataset:** doi:10.17917/C78G69 (FDA OSEL, public domain)
**Architecture:** 6 hidden layers × 64 units, tanh, with hard wall+axis BC enforcement
**Training:** 30,000 Adam iters + 5,000 L-BFGS iters per Reynolds number
**Runtime:** ~30–45 min total on a T4 / L4 / A100 GPU; ~3-4 hours on CPU

### Outputs (saved to `/content/B4_outputs/`)
- `PIV_Re{500,2000,3500}.csv` — parsed real PIV data (4,200+ samples each)
- `fields_Re{500,2000,3500}.png` — full-field reconstructions of `u_x`, `u_r`, `p`
- `centreline_Re{500,2000,3500}.png` — axial velocity along the centreline
- `wss_Re{500,2000,3500}.png` — wall shear stress along the wall
- `metrics_Re{500,2000,3500}.json` — convergence + reconstruction numbers
- `summary.png` — combined plot across all three Re
- `B4_results.zip` — everything packaged for one-click download

### How to use
1. Runtime → Change runtime type → GPU (T4 is fine, L4/A100 faster)
2. Runtime → Run all
3. When the last cell finishes, it triggers a download of `B4_results.zip`
""")


# =====================================================================
# CELL 2 — Install dependencies + verify GPU
# =====================================================================
code("""# Clean install of numpy/scipy + JAX stack.
#
# Why uninstall first: Colab's pre-installed numpy can end up in a
# partial-upgrade state where its own _core/strings.py can't import
# _center from _core/umath (those are added in numpy 2.1+). Plain
# --upgrade doesn't always fix this; uninstall + clean install does.
#
# Why numpy>=2.1, scipy>=1.14: that's where the modern numpy.strings
# ufuncs (incl. _center) live. Anything older will trip the
# ImportError on `from jaxopt import LBFGS`.
#
# Why --no-cache-dir: pip's wheel cache may have served the broken
# build; this forces a fresh download.
#
# Why single-line commands (no backslash continuation): line breaks
# inside !pip can fragment in Colab and cause "missing argument"
# errors. Keep each install on one physical line.
import sys, subprocess

print(">> Uninstalling numpy / scipy ...")
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                       "numpy", "scipy"])

print(">> Installing JAX + Flax + NumPy/SciPy as a matched set ...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-cache-dir",
                       "numpy>=2.1", "scipy>=1.14",
                       "jax[cuda12]", "flax>=0.10",
                       "optax>=0.2.3", "jaxopt>=0.8"])

print()
print("=" * 60)
print("  IMPORTANT: now Runtime -> Restart runtime, then run all cells")
print("  again (so the fresh numpy/scipy/jax load into the kernel).")
print("=" * 60)

# These two lines will only succeed AFTER you restart the runtime:
import jax
print("\\nJAX version    :", jax.__version__)
print("Devices        :", jax.devices())
print("Default backend:", jax.default_backend())

if jax.default_backend() == "cpu":
    raise RuntimeError(
        "\\n\\n*** GPU NOT DETECTED ***\\n"
        "Training on CPU would take 3-4 hours per Reynolds number.\\n"
        "Fix:\\n"
        "  1) Runtime -> Change runtime type -> Hardware accelerator: T4 GPU -> Save\\n"
        "  2) Runtime -> Restart runtime\\n"
        "  3) Runtime -> Run all\\n\\n"
        "If T4 GPU is greyed out, your free-tier quota is exhausted.\\n"
        "Wait ~12h for reset, switch Google accounts, or get Colab Pro.\\n"
    )

print("\\nGPU detected. Ready to train.")
""")


# =====================================================================
# CELL 3 — Clone the FDA OSEL benchmark repo
# =====================================================================
code("""# Clone the official FDA Office of Science & Engineering Labs benchmark repo
import os
if not os.path.exists("CFD-and-Blood-Damage-Benchmarks"):
    !git clone --depth 1 https://github.com/OSEL-DAM/CFD-and-Blood-Damage-Benchmarks.git

print("\\nBenchmark contents:")
!ls -la "CFD-and-Blood-Damage-Benchmarks/Nozzle/Data/" | head -20
""")


# =====================================================================
# CELL 4 — Constants, geometry, PIV parser
# =====================================================================
code("""# ============================================================
# Configuration, geometry, and PIV parsing
# ============================================================
import os
import csv
import json
import time
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from jaxopt import LBFGS

jax.config.update("jax_enable_x64", True)
SEED = 1234
np.random.seed(SEED)

OUT_DIR = Path("B4_outputs")
OUT_DIR.mkdir(exist_ok=True)

# === Physical constants (Hariharan 2011, blood analogue: 40% glycerol-water) ===
RHO = 1056.0           # kg / m^3
MU  = 3.5e-3           # Pa s
NU  = MU / RHO         # m^2 / s
D_THROAT = 0.004; R_THROAT = D_THROAT / 2.0
D_INLET  = 0.012; R_INLET  = D_INLET / 2.0

# === Geometry (sudden-expansion orientation; throat starts at x=0) ===
X_INLET_START    = -0.088
X_CONTRACT_START = -0.022675
X_THROAT_START   = 0.0
X_THROAT_END     = 0.040
X_OUTLET_END     = 0.143
R_EPS = 1e-8

# Input normalisation extents
X_MIN_NORM = X_INLET_START
X_MAX_NORM = X_OUTLET_END
R_MIN_NORM = 0.0
R_MAX_NORM = R_INLET

# Output scaling (set per-Re in the main loop). USE_HARD_BC stays True
# so wall no-slip and axis u_r=0 are mathematically guaranteed.
U_SCALE = 1.0
P_SCALE = 1.0
USE_HARD_BC = True

# === Geometry functions ===
def wall_radius(x):
    contract_frac = (x - X_CONTRACT_START) / (X_THROAT_START - X_CONTRACT_START)
    contract_frac = jnp.clip(contract_frac, 0.0, 1.0)
    R_contract = R_INLET + (R_THROAT - R_INLET) * contract_frac
    return jnp.where(x < X_CONTRACT_START, R_INLET,
           jnp.where(x < X_THROAT_START,   R_contract,
           jnp.where(x < X_THROAT_END,     R_THROAT,
                                            R_INLET)))

def _scalar_wall_radius(x):
    # NumPy-friendly version for the parser
    if x < X_CONTRACT_START:
        return R_INLET
    if x < X_THROAT_START:
        frac = (x - X_CONTRACT_START) / (X_THROAT_START - X_CONTRACT_START)
        return R_INLET + (R_THROAT - R_INLET) * np.clip(frac, 0.0, 1.0)
    if x < X_THROAT_END:
        return R_THROAT
    return R_INLET

def in_domain(x, r):
    return (x >= X_INLET_START) & (x <= X_OUTLET_END) & \\
           (r >= 0.0) & (r <= wall_radius(x))

def _normalize_xr(x, r):
    x_n = 2.0 * (x - X_MIN_NORM) / (X_MAX_NORM - X_MIN_NORM) - 1.0
    r_n = 2.0 * (r - R_MIN_NORM) / (R_MAX_NORM - R_MIN_NORM) - 1.0
    return x_n, r_n

def mean_throat_velocity(Re):
    return Re * MU / (RHO * D_THROAT)

def hagen_poiseuille_inlet(r, U_mean):
    U_inlet_mean = U_mean * (R_THROAT / R_INLET) ** 2
    U_max = 2.0 * U_inlet_mean
    return U_max * (1.0 - (r / R_INLET) ** 2)

# === PIV parser for the FDA OSEL .txt format ===
COMPONENT_MAP = {
    "profile-axial-velocity-at-z":  "u_x",
    "profile-radial-velocity-at-z": "u_r",
}

def _read_count(it):
    for line in it:
        s = line.strip()
        if not s: continue
        try: return int(s)
        except ValueError: raise RuntimeError(f"bad count: {s!r}")
    raise RuntimeError("EOF reading count")

def _read_n_xy(it, n):
    pos, val = [], []
    while len(pos) < n:
        try: line = next(it)
        except StopIteration: raise RuntimeError(f"EOF after {len(pos)}/{n}")
        s = line.strip()
        if not s: continue
        parts = s.split()
        if len(parts) < 2: raise RuntimeError(f"bad data line: {s!r}")
        pos.append(float(parts[0])); val.append(float(parts[1]))
    return np.array(pos), np.array(val)

def parse_piv_file(path):
    profiles = {}
    with open(path) as fh:
        lines = iter(fh.readlines())
    for line in lines:
        s = line.strip()
        if not s or not s.startswith("plot-"):
            continue
        body = s[5:]
        prof_key = next((k for k in COMPONENT_MAP if body.startswith(k)), None)
        if prof_key is None:
            # Wall/centerline/etc — skip the section body
            try:
                n = _read_count(lines); _, _ = _read_n_xy(lines, n)
            except RuntimeError:
                pass
            continue
        tail = body[len(prof_key):].strip()
        z_loc = float(tail.split()[0])
        n = _read_count(lines)
        r, v = _read_n_xy(lines, n)
        profiles[(COMPONENT_MAP[prof_key], z_loc)] = (r, v)
    return profiles

def parse_zip_to_csv(zip_path, Re, out_csv, x_shift=-0.040, r_tol=5e-4):
    bucket = {}
    R_KEY_TOL = 1e-7
    def keyf(z, r): return (float(z), round(float(r) / R_KEY_TOL) * R_KEY_TOL)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        txt_files = sorted(Path(tmp).rglob("*.txt"))
        for tx in txt_files:
            if "LDV" in tx.name:
                continue
            profiles = parse_piv_file(tx)
            for (comp, z), (rs, vs) in profiles.items():
                for r, v in zip(rs, vs):
                    k = keyf(z, r)
                    d = bucket.setdefault(k, {"u_x": np.nan, "u_r": np.nan})
                    d[comp] = float(v)
    rows = []
    for (z, r), d in sorted(bucket.items()):
        x = z - x_shift
        ux, ur = d["u_x"], d["u_r"]
        if r < 0:
            ur = -ur if not np.isnan(ur) else ur
            r = abs(r)
        if r > _scalar_wall_radius(x) + r_tol:
            continue
        rows.append((x, r, ux, ur))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "r", "u_x", "u_r", "Re"])
        for x, r, ux, ur in rows:
            w.writerow([
                f"{x:.6e}", f"{r:.6e}",
                "" if np.isnan(ux) else f"{ux:.6e}",
                "" if np.isnan(ur) else f"{ur:.6e}",
                int(Re),
            ])
    return len(rows)

print("Constants and parser loaded.")
print(f"  Throat U_mean: Re=500 -> {mean_throat_velocity(500):.4f} m/s,  "
      f"Re=2000 -> {mean_throat_velocity(2000):.4f},  "
      f"Re=3500 -> {mean_throat_velocity(3500):.4f}")
""")


# =====================================================================
# CELL 5 — Parse all three Re ZIPs into CSV
# =====================================================================
code("""# Parse the three Re ZIPs into PINN-ready CSVs
BENCH_DIR = Path("CFD-and-Blood-Damage-Benchmarks/Nozzle/Data")

for re in (500, 2000, 3500):
    zname = f"SE_exp_{re:04d}.zip"
    out_csv = OUT_DIR / f"PIV_Re{re}.csv"
    n = parse_zip_to_csv(BENCH_DIR / zname, re, out_csv)
    print(f"Re={re:<5d}  ->  {out_csv.name}  ({n:,} rows)")
""")


# =====================================================================
# CELL 6 — PINN model + losses
# =====================================================================
code("""# ============================================================
# PINN: MLP, hard-BC output transform, NS residuals, loss components
# ============================================================
LAYERS_DEFAULT = [2, 64, 64, 64, 64, 64, 64, 3]   # (x, r) -> (u_x, u_r, p)


class MLP(nnx.Module):
    \"\"\"Tanh MLP with optional Fourier-feature input encoding.\"\"\"
    def __init__(self, layers, *, rngs, n_fourier=0, fourier_scale=5.0):
        self.has_ff = n_fourier > 0
        if self.has_ff:
            key = rngs.params()
            self.B = fourier_scale * jax.random.normal(
                key, (layers[0], n_fourier), dtype=jnp.float64)
            eff = [2 * n_fourier] + list(layers[1:])
        else:
            eff = list(layers)
        self.n_layers = len(eff) - 1
        for i in range(self.n_layers):
            setattr(self, f"lin_{i}",
                    nnx.Linear(eff[i], eff[i+1],
                               kernel_init=nnx.initializers.glorot_normal(),
                               bias_init=nnx.initializers.zeros_init(),
                               param_dtype=jnp.float64, rngs=rngs))
    def __call__(self, xr):
        if self.has_ff:
            proj = 2.0 * jnp.pi * (xr @ self.B)
            h = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
        else:
            h = xr
        for i in range(self.n_layers - 1):
            h = jnp.tanh(getattr(self, f"lin_{i}")(h))
        return getattr(self, f"lin_{self.n_layers - 1}")(h)


def uvp_at(model, x, r):
    \"\"\"Predicted (u_x, u_r, p) in physical units, with hard BC.\"\"\"
    x_n, r_n = _normalize_xr(x, r)
    out = model(jnp.stack([x_n, r_n]))
    if USE_HARD_BC:
        R_w = wall_radius(x)
        rho_r = r / R_w
        phi = 1.0 - rho_r ** 2
        return (U_SCALE * phi * out[0],
                U_SCALE * rho_r * phi * out[1],
                P_SCALE * out[2])
    return U_SCALE * out[0], U_SCALE * out[1], P_SCALE * out[2]


def _ux(m, x, r): return uvp_at(m, x, r)[0]
def _ur(m, x, r): return uvp_at(m, x, r)[1]
def _p (m, x, r): return uvp_at(m, x, r)[2]


def _ns_residual_one(model, x, r):
    dux_dx = jax.grad(_ux, argnums=1)(model, x, r)
    dux_dr = jax.grad(_ux, argnums=2)(model, x, r)
    dur_dx = jax.grad(_ur, argnums=1)(model, x, r)
    dur_dr = jax.grad(_ur, argnums=2)(model, x, r)
    dp_dx  = jax.grad(_p,  argnums=1)(model, x, r)
    dp_dr  = jax.grad(_p,  argnums=2)(model, x, r)
    d2ux_dx2 = jax.grad(lambda xx: jax.grad(_ux, argnums=1)(model, xx, r))(x)
    d2ux_dr2 = jax.grad(lambda rr: jax.grad(_ux, argnums=2)(model, x, rr))(r)
    d2ur_dx2 = jax.grad(lambda xx: jax.grad(_ur, argnums=1)(model, xx, r))(x)
    d2ur_dr2 = jax.grad(lambda rr: jax.grad(_ur, argnums=2)(model, x, rr))(r)
    ux, ur, _ = uvp_at(model, x, r)
    r_safe = r + R_EPS
    R_c = dux_dx + dur_dr + ur / r_safe
    R_x = (ux * dux_dx + ur * dux_dr + (1.0/RHO) * dp_dx
           - NU * (d2ux_dx2 + d2ux_dr2 + dux_dr / r_safe))
    R_r = (ux * dur_dx + ur * dur_dr + (1.0/RHO) * dp_dr
           - NU * (d2ur_dx2 + d2ur_dr2 + dur_dr / r_safe - ur / r_safe ** 2))
    return R_c, R_x, R_r


def _residual_losses(model, pts):
    R_c, R_x, R_r = jax.vmap(_ns_residual_one, in_axes=(None, 0, 0))(
        model, pts["x_in"], pts["r_in"])
    return jnp.mean(R_c ** 2), jnp.mean(R_x ** 2), jnp.mean(R_r ** 2)

def _wall_loss(model, pts):
    ux = jax.vmap(_ux, in_axes=(None, 0, 0))(model, pts["x_w"], pts["r_w"])
    ur = jax.vmap(_ur, in_axes=(None, 0, 0))(model, pts["x_w"], pts["r_w"])
    return jnp.mean(ux ** 2 + ur ** 2)

def _inlet_loss(model, pts, Re):
    U_mean = mean_throat_velocity(Re)
    target = jax.vmap(lambda rv: hagen_poiseuille_inlet(rv, U_mean))(pts["r_inl"])
    ux = jax.vmap(_ux, in_axes=(None, 0, 0))(model, pts["x_inl"], pts["r_inl"])
    ur = jax.vmap(_ur, in_axes=(None, 0, 0))(model, pts["x_inl"], pts["r_inl"])
    return jnp.mean((ux - target) ** 2) + jnp.mean(ur ** 2)

def _outlet_loss(model, pts):
    p_out = jax.vmap(_p, in_axes=(None, 0, 0))(model, pts["x_out"], pts["r_out"])
    return jnp.mean(p_out ** 2)

def _axis_loss(model, pts):
    ur = jax.vmap(_ur, in_axes=(None, 0, 0))(model, pts["x_ax"], pts["r_ax"])
    dux_dr = jax.vmap(
        lambda xv, rv: jax.grad(_ux, argnums=2)(model, xv, rv)
    )(pts["x_ax"], pts["r_ax"])
    return jnp.mean(ur ** 2 + dux_dr ** 2)

def _data_loss(model, obs):
    \"\"\"Per-component relative L2: each velocity component is normalised by
    its own RMS magnitude in the data so u_x and u_r contribute equally.

    Without this, u_r (~15x smaller than u_x in pipe flow) is dominated by
    u_x errors and the network ignores it -- as observed in the Re=2000/3500
    runs where u_r relative L2 was ~100% (network predicting ~0).
    \"\"\"
    ux = jax.vmap(_ux, in_axes=(None, 0, 0))(model, obs["x_d"], obs["r_d"])
    ur = jax.vmap(_ur, in_axes=(None, 0, 0))(model, obs["x_d"], obs["r_d"])
    err_x  = jnp.sum(obs["mux"] * (ux - obs["ux_d"]) ** 2)
    err_r  = jnp.sum(obs["mur"] * (ur - obs["ur_d"]) ** 2)
    norm_x = jnp.sum(obs["mux"] * obs["ux_d"] ** 2) + 1e-12
    norm_r = jnp.sum(obs["mur"] * obs["ur_d"] ** 2) + 1e-12
    return 0.5 * (err_x / norm_x + err_r / norm_r)

def total_loss(model, pts, obs, Re, weights):
    L_c, L_mx, L_mr = _residual_losses(model, pts)
    L_w = _wall_loss(model, pts)
    L_i = _inlet_loss(model, pts, Re)
    L_o = _outlet_loss(model, pts)
    L_a = _axis_loss(model, pts)
    L_d = _data_loss(model, obs)
    return (weights["continuity"] * L_c
            + weights["mom_x"]    * L_mx
            + weights["mom_r"]    * L_mr
            + weights["wall"]     * L_w
            + weights["inlet"]    * L_i
            + weights["outlet"]   * L_o
            + weights["axis"]     * L_a
            + weights["data"]     * L_d)


def sample_points(n_interior, n_wall, n_inlet, n_outlet, n_axis, rng):
    xs, rs = [], []
    while len(xs) < n_interior:
        batch = max(2000, n_interior - len(xs))
        cand_x = rng.uniform(X_INLET_START, X_OUTLET_END, batch)
        cand_r = rng.uniform(0.0, R_INLET, batch)
        mask = np.asarray(in_domain(jnp.asarray(cand_x), jnp.asarray(cand_r)))
        xs.extend(cand_x[mask].tolist())
        rs.extend(cand_r[mask].tolist())
    x_in = np.array(xs[:n_interior]); r_in = np.array(rs[:n_interior])
    x_w = rng.uniform(X_INLET_START, X_OUTLET_END, n_wall)
    r_w = np.asarray(wall_radius(jnp.asarray(x_w)))
    x_inl = np.full(n_inlet, X_INLET_START)
    r_inl = rng.uniform(0.0, R_INLET, n_inlet)
    x_out = np.full(n_outlet, X_OUTLET_END)
    r_out = rng.uniform(0.0, R_INLET, n_outlet)
    x_ax = rng.uniform(X_INLET_START, X_OUTLET_END, n_axis)
    r_ax = np.zeros(n_axis)
    return {
        "x_in":  jnp.asarray(x_in),  "r_in":  jnp.asarray(r_in),
        "x_w":   jnp.asarray(x_w),   "r_w":   jnp.asarray(r_w),
        "x_inl": jnp.asarray(x_inl), "r_inl": jnp.asarray(r_inl),
        "x_out": jnp.asarray(x_out), "r_out": jnp.asarray(r_out),
        "x_ax":  jnp.asarray(x_ax),  "r_ax":  jnp.asarray(r_ax),
    }


def load_piv_csv(path, re_filter=None):
    xs, rs, uxs, urs, mux, mur = [], [], [], [], [], []
    with open(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if re_filter is not None and "Re" in row:
                try:
                    if int(float(row["Re"])) != int(re_filter):
                        continue
                except ValueError:
                    pass
            xs.append(float(row["x"]))
            rs.append(float(row["r"]))
            ux_str = (row.get("u_x") or "").strip()
            ur_str = (row.get("u_r") or "").strip()
            if ux_str: uxs.append(float(ux_str)); mux.append(1.0)
            else:      uxs.append(0.0);           mux.append(0.0)
            if ur_str: urs.append(float(ur_str)); mur.append(1.0)
            else:      urs.append(0.0);           mur.append(0.0)
    return {
        "x_d":  jnp.asarray(xs), "r_d":  jnp.asarray(rs),
        "ux_d": jnp.asarray(uxs), "ur_d": jnp.asarray(urs),
        "mux":  jnp.asarray(mux), "mur":  jnp.asarray(mur),
    }

print("PINN definitions loaded.")
""")


# =====================================================================
# CELL 7 — Training + evaluation + plotting helpers
# =====================================================================
code("""# ============================================================
# Training (Adam -> L-BFGS), evaluation, plotting
# ============================================================
def train(model, pts, obs, Re, weights, adam_iters, lbfgs_iters,
          lr=1e-3, print_every=2000, lr_final=5e-4):
    # Gentle cosine decay (only down to 5e-4, half of initial). Aggressive
    # decay to 1e-5 was killing high-Re training: by mid-Adam the LR was
    # too small for the network to climb out of its initial basin into the
    # high-magnitude solution the data calls for. A gentle 2x decay keeps
    # learning ability while still settling the final iters into a better
    # minimum than constant LR.
    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=adam_iters,
        alpha=lr_final / lr,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    @nnx.jit
    def adam_step(model, optimizer, pts, obs):
        loss_val, grads = nnx.value_and_grad(
            lambda m: total_loss(m, pts, obs, Re, weights)
        )(model)
        optimizer.update(model, grads)
        return loss_val

    t0 = time.time()
    for it in range(adam_iters):
        loss_val = adam_step(model, optimizer, pts, obs)
        if it % print_every == 0:
            print(f"  Adam iter {it:>6d}  loss = {float(loss_val):.4e}",
                  flush=True)
    print(f"  Adam phase: {time.time() - t0:.1f}s")

    gdef, state = nnx.split(model)
    def lbfgs_loss(params):
        m = nnx.merge(gdef, params)
        return total_loss(m, pts, obs, Re, weights)
    t0 = time.time()
    solver = LBFGS(fun=lbfgs_loss, maxiter=lbfgs_iters, tol=1e-9)
    result = solver.run(state)
    model = nnx.merge(gdef, result.params)
    print(f"  L-BFGS phase: {time.time() - t0:.1f}s")
    return model


def evaluate(model, Re):
    nx, nr = 401, 81
    xs = np.linspace(X_INLET_START, X_OUTLET_END, nx)
    rs = np.linspace(0.0, R_INLET, nr)
    @jax.jit
    def predict_grid(xs_j, rs_j):
        X, R = jnp.meshgrid(xs_j, rs_j, indexing="xy")
        def one(xv, rv):
            ux, ur, p = uvp_at(model, xv, rv)
            return jnp.stack([ux, ur, p])
        out = jax.vmap(one)(X.ravel(), R.ravel())
        return out.reshape(nr, nx, 3)
    grid = np.asarray(predict_grid(jnp.asarray(xs), jnp.asarray(rs)))
    mask = np.asarray(in_domain(
        jnp.asarray(np.broadcast_to(xs, (nr, nx))),
        jnp.asarray(rs[:, None] * np.ones((1, nx))),
    ))
    ux = np.where(mask, grid[..., 0], np.nan)
    ur = np.where(mask, grid[..., 1], np.nan)
    p  = np.where(mask, grid[..., 2], np.nan)
    return {"xs": xs, "rs": rs, "ux": ux, "ur": ur, "p": p,
            "U_mean": mean_throat_velocity(Re)}


def wall_shear_stress(model, n=400):
    xs = np.linspace(X_INLET_START, X_OUTLET_END, n)
    rs = np.asarray(wall_radius(jnp.asarray(xs)))
    @jax.jit
    def dux_dr_wall(xv, rv):
        return jax.grad(_ux, argnums=2)(model, xv, rv)
    dux = np.asarray(jax.vmap(dux_dr_wall)(jnp.asarray(xs), jnp.asarray(rs)))
    return xs, MU * np.abs(dux)


def plot_fields(ev, savepath, Re):
    xs, rs = ev["xs"], ev["rs"]
    extent = [xs[0] * 1000, xs[-1] * 1000, 0, rs[-1] * 1000]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, (field, label, cmap) in zip(axes, [
        (ev["ux"], r"$u_x$ (m/s)",        "viridis"),
        (ev["ur"], r"$u_r$ (m/s)",        "RdBu_r"),
        (ev["p"],  r"$p$ (Pa, relative)",  "inferno"),
    ]):
        im = ax.imshow(field, origin="lower", extent=extent,
                       aspect="auto", cmap=cmap)
        ax.set_ylabel("r (mm)"); ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    axes[-1].set_xlabel("x (mm)")
    fig.suptitle(f"PINN-reconstructed nozzle fields  (Re = {Re})")
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_centreline(ev, savepath, Re):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(ev["xs"] * 1000, ev["ux"][0, :], "k-", lw=2)
    for xpos in [X_CONTRACT_START, X_THROAT_START, X_THROAT_END]:
        ax.axvline(xpos * 1000, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("x (mm)"); ax.set_ylabel(r"$u_x(x, r=0)$ (m/s)")
    ax.set_title(f"Centreline axial velocity  (Re = {Re})")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_wss(xs, tau_w, savepath, Re):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xs * 1000, tau_w, "r-", lw=2)
    ax.axhline(15.0, color="grey", ls=":", lw=0.8,
               label=r"WSS = 15 Pa (Malek 1999)")
    ax.set_xlabel("x (mm)"); ax.set_ylabel(r"$\\tau_w$ (Pa)")
    ax.set_title(f"Wall shear stress along nozzle  (Re = {Re})")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight"); plt.close(fig)

print("Training, evaluation, and plotting helpers loaded.")
""")


# =====================================================================
# CELL 8 — MAIN training loop (all 3 Re cases)
# =====================================================================
code("""# ============================================================
# MAIN — train Re = 500, 2000, 3500
#
# Per-Re Adam iter schedule: higher Re flow has sharper jet structure
# and slower convergence, so the higher-Re cases get more compute.
# Adjust ADAM_ITERS_BY_RE / N_INTERIOR_BY_RE if you have GPU headroom.
# ============================================================
ADAM_ITERS_BY_RE = {500: 30_000, 2000: 50_000, 3500: 60_000}
N_INTERIOR_BY_RE = {500: 5_000,  2000: 7_000,  3500: 8_000}
# Per-Re data weight: momentum residuals scale as U^2, so at Re=3500 the
# residual term is ~50x larger in magnitude than at Re=500 and dominates
# the loss. Boosting the data weight at high Re restores balance and lets
# the network learn the high-magnitude jet from the data.
DATA_WEIGHT_BY_RE = {500: 100.0, 2000: 200.0, 3500: 400.0}
LBFGS_ITERS = 5_000
N_WALL      = 800
N_INLET     = 300
N_OUTLET    = 300
N_AXIS      = 300
WIDTH       = 64
DEPTH       = 6
WEIGHTS_TEMPLATE = {
    "continuity":  1.0, "mom_x": 1.0, "mom_r": 1.0,
    "wall":        0.0,   # hard-enforced
    "inlet":      10.0, "outlet": 0.1,
    "axis":        0.0,   # hard-enforced (u_r via output transform)
    # "data" filled per-Re from DATA_WEIGHT_BY_RE
}

all_metrics = {}
for re in [500, 2000, 3500]:
    print(f"\\n{'='*64}")
    print(f"  Re = {re}")
    print(f"{'='*64}")

    # Set per-Re output scales BEFORE building the model so JIT bakes them in.
    # Clear caches so new Re re-traces with the new scales.
    jax.clear_caches()
    U_SCALE = mean_throat_velocity(re)
    P_SCALE = RHO * U_SCALE ** 2
    print(f"  U_SCALE = {U_SCALE:.4f} m/s   P_SCALE = {P_SCALE:.2f} Pa")

    adam_iters_re = ADAM_ITERS_BY_RE[re]
    n_interior_re = N_INTERIOR_BY_RE[re]
    weights_re = dict(WEIGHTS_TEMPLATE, data=DATA_WEIGHT_BY_RE[re])
    rng = np.random.default_rng(SEED)
    pts = sample_points(n_interior_re, N_WALL, N_INLET, N_OUTLET, N_AXIS, rng)
    obs = load_piv_csv(OUT_DIR / f"PIV_Re{re}.csv", re_filter=re)
    print(f"  N_PIV = {obs['x_d'].shape[0]}   N_interior = {n_interior_re}")
    print(f"  Hyperparameters: {DEPTH} layers x {WIDTH} units, "
          f"Adam {adam_iters_re}, L-BFGS {LBFGS_ITERS}, cosine LR 1e-3 -> 5e-4")
    print(f"  Loss weights: {weights_re}")

    rngs = nnx.Rngs(SEED)
    layers = [2] + [WIDTH] * DEPTH + [3]
    model = MLP(layers, rngs=rngs, n_fourier=0)
    model = train(model, pts, obs, re, weights_re, adam_iters_re, LBFGS_ITERS)

    # Evaluate + plot
    ev = evaluate(model, re)
    plot_fields(ev,     OUT_DIR / f"fields_Re{re}.png",     re)
    plot_centreline(ev, OUT_DIR / f"centreline_Re{re}.png", re)
    xs_w, tau_w = wall_shear_stress(model)
    plot_wss(xs_w, tau_w, OUT_DIR / f"wss_Re{re}.png", re)

    # Mask-aware metrics
    ux_pred = np.asarray(jax.vmap(_ux, in_axes=(None, 0, 0))(
        model, obs["x_d"], obs["r_d"]))
    ur_pred = np.asarray(jax.vmap(_ur, in_axes=(None, 0, 0))(
        model, obs["x_d"], obs["r_d"]))
    mux_b = np.asarray(obs["mux"]).astype(bool)
    mur_b = np.asarray(obs["mur"]).astype(bool)
    ux_t  = np.asarray(obs["ux_d"]); ur_t = np.asarray(obs["ur_d"])
    rel_ux = 100.0 * np.linalg.norm(ux_pred[mux_b] - ux_t[mux_b]) / \\
             max(np.linalg.norm(ux_t[mux_b]), 1e-12)
    rel_ur = 100.0 * np.linalg.norm(ur_pred[mur_b] - ur_t[mur_b]) / \\
             max(np.linalg.norm(ur_t[mur_b]), 1e-12)
    # Stewart 2012 Eq. (2) global error metric E_g (mean absolute
    # relative error). Same metric the FDA interlaboratory study used.
    # Skip near-zero true velocities (|u_true| < 0.01 * U_throat) to
    # avoid the same blow-up Stewart noted at low-velocity points.
    nonzero = mux_b & (np.abs(ux_t) > 0.01 * float(U_SCALE))
    E_g_ux  = float(np.mean(np.abs(ux_pred[nonzero] - ux_t[nonzero]) /
                            np.abs(ux_t[nonzero])))

    metrics = {
        "reynolds":             re,
        "U_mean_throat_m_per_s": float(U_SCALE),
        "P_scale_Pa":           float(P_SCALE),
        "n_obs":                int(obs["x_d"].shape[0]),
        "data_rel_L2_ux_pct":   float(rel_ux),
        "data_rel_L2_ur_pct":   float(rel_ur),
        "stewart_2012_Eg_ux":   E_g_ux,
        "tau_w_peak_Pa":        float(np.nanmax(tau_w)),
        "ux_peak_predicted":    float(np.nanmax(ev["ux"])),
        "ux_peak_pivdata":      float(np.nanmax(ux_t[mux_b])),
        "p_drop_Pa":            float(np.nanmax(ev["p"]) - np.nanmin(ev["p"])),
        "hyperparameters": {
            "architecture":  f"{DEPTH} hidden x {WIDTH} units, tanh, hard BC",
            "adam_iters":    adam_iters_re,
            "lbfgs_iters":   LBFGS_ITERS,
            "n_interior":    n_interior_re,
            "lr_schedule":   "cosine 1e-3 -> 5e-4",
            "data_loss":     "per-component relative L2",
            "weights":       weights_re,
        },
    }
    all_metrics[f"Re{re}"] = metrics
    with open(OUT_DIR / f"metrics_Re{re}.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\\n  >> Re={re}  data L2(u_x)={rel_ux:5.2f}%   "
          f"Stewart E_g={E_g_ux:.3f}   "
          f"peak WSS={metrics['tau_w_peak_Pa']:6.2f} Pa   "
          f"ux peak (PINN/PIV)={metrics['ux_peak_predicted']:.3f}/"
          f"{metrics['ux_peak_pivdata']:.3f} m/s")

print("\\n=== ALL THREE Re COMPLETE ===")
print(json.dumps(all_metrics, indent=2))
""")


# =====================================================================
# CELL 9 — Summary plot across Re
# =====================================================================
code("""# ============================================================
# Summary plot across all three Reynolds numbers
# ============================================================
res = [500, 2000, 3500]
ux_peaks  = [all_metrics[f"Re{r}"]["ux_peak_predicted"] for r in res]
ux_truths = [all_metrics[f"Re{r}"]["ux_peak_pivdata"]   for r in res]
tau_peaks = [all_metrics[f"Re{r}"]["tau_w_peak_Pa"]     for r in res]
p_drops   = [all_metrics[f"Re{r}"]["p_drop_Pa"]         for r in res]
L2_ux     = [all_metrics[f"Re{r}"]["data_rel_L2_ux_pct"] for r in res]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(res, ux_peaks,  "ko-", label="PINN",       lw=2, ms=8)
axes[0,0].plot(res, ux_truths, "rs--", label="PIV (truth)", lw=2, ms=8)
axes[0,0].set_xlabel("Throat Reynolds number")
axes[0,0].set_ylabel("Peak axial velocity (m/s)")
axes[0,0].set_title("Throat peak $u_x$: PINN vs. PIV")
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(res, tau_peaks, "bo-", lw=2, ms=8)
axes[0,1].axhline(15.0, color="grey", ls=":",
                   label="Malek 1999 platelet activation threshold")
axes[0,1].set_xlabel("Throat Reynolds number")
axes[0,1].set_ylabel("Peak wall shear stress (Pa)")
axes[0,1].set_title("Peak WSS along the nozzle")
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

axes[1,0].plot(res, p_drops, "go-", lw=2, ms=8)
axes[1,0].set_xlabel("Throat Reynolds number")
axes[1,0].set_ylabel("Pressure drop (Pa)")
axes[1,0].set_title("Inlet-to-outlet pressure drop")
axes[1,0].grid(alpha=0.3)

axes[1,1].bar([str(r) for r in res], L2_ux, color="purple", alpha=0.7)
axes[1,1].axhline(20.0, color="grey", ls=":", label="Paper-grade threshold")
axes[1,1].set_xlabel("Throat Reynolds number")
axes[1,1].set_ylabel("Data $L^2$ error (%)")
axes[1,1].set_title("PINN vs. PIV agreement")
axes[1,1].legend(); axes[1,1].grid(alpha=0.3, axis="y")

fig.suptitle("B4 FDA Nozzle PINN — Summary across Reynolds numbers",
             fontsize=14, y=1.00)
fig.tight_layout()
fig.savefig(OUT_DIR / "summary.png", dpi=200, bbox_inches="tight")
plt.show()
""")


# =====================================================================
# CELL 10 — Zip + download
# =====================================================================
code("""# ============================================================
# Package + download
# ============================================================
import shutil
shutil.make_archive("B4_results", "zip", OUT_DIR)
print(f"Archive: B4_results.zip  ({os.path.getsize('B4_results.zip')/1024:.0f} KB)")
print("Contents:")
!unzip -l B4_results.zip | head -20

try:
    from google.colab import files
    files.download("B4_results.zip")
except Exception as e:
    print(f"\\nNot in Colab (or download failed): {e}")
    print("Find the archive at ./B4_results.zip")
""")


# =====================================================================
# Write notebook
# =====================================================================
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        },
        "language_info": {"name": "python"},
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = Path(__file__).parent / "B4_FDA_Nozzle_PINN_Colab.ipynb"
with open(out_path, "w") as fh:
    json.dump(NB, fh, indent=1)
print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB, {len(CELLS)} cells)")
