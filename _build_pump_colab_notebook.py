"""Generate B4_FDA_Pump_Diffuser_PINN_Colab.ipynb (A2 of B4 paper)."""
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
# CELL 1 — Title
# =====================================================================
md("""# B4 — FDA Blood Pump Diffuser PINN: Colab Training (5 conditions)

Physics-Informed Neural Network reconstruction of full-field hemodynamics
in the FDA Blood Pump diffuser region, using the official inter-laboratory
PIV dataset (Hariharan et al. 2018).

**Author:** Emmanuel Hackman, USM PhD Computational Science
**Dataset:** FDA OSEL Blood Pump benchmark, public domain
**Conditions:** C1 (2.5 L/min, 2500 rpm), C2 (2.5, 3500), C4 (6.0, 2500), C5 (6.0, 3500), C6 (7.0, 3500)
**Architecture:** 6 hidden layers × 64 units, tanh, hard input normalization
**Loss:** magnitude-only data loss (|V|) + 2D incompressible Navier-Stokes residuals
**Training:** 60,000 Adam + 5,000 L-BFGS iters per condition
**Runtime:** ~25–35 min total on a T4 / L4 / A100 GPU

### How to use
1. Runtime → Change runtime type → GPU (T4 is fine, L4/A100 faster)
2. Run cell 2 (installs deps), then **Runtime → Restart runtime**
3. Upload `PumpDiffuser_C{1,2,4,5,6}.csv` files when prompted (cell 3)
4. Runtime → Run all
5. When the last cell finishes, `B4_pump_results.zip` downloads automatically
""")


# =====================================================================
# CELL 2 — Install + GPU check
# =====================================================================
code("""# Clean install of numpy/scipy + JAX stack (matches A1 nozzle workflow).
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

print("\\n" + "=" * 60)
print("  IMPORTANT: now Runtime -> Restart runtime, then Run all")
print("=" * 60)

import jax
print("\\nJAX:", jax.__version__, " Devices:", jax.devices(),
      " Backend:", jax.default_backend())

if jax.default_backend() == "cpu":
    raise RuntimeError("*** GPU NOT DETECTED *** "
                       "Runtime -> Change runtime type -> T4 GPU, "
                       "then Restart runtime, then Run all.")

print("\\nGPU detected. Ready to train.")
""")


# =====================================================================
# CELL 3 — Upload PIV CSVs
# =====================================================================
code("""# Upload the 5 parsed PIV CSV files. These are produced locally by
# parse_pump_diffuser_piv.py from the FDA OSEL xlsx files.
from google.colab import files
import os, pathlib

print("Please upload PumpDiffuser_C1.csv, PumpDiffuser_C2.csv,")
print("PumpDiffuser_C4.csv, PumpDiffuser_C5.csv, PumpDiffuser_C6.csv")
print("(you can select all 5 at once)")

uploaded = files.upload()
pathlib.Path("/content/pump_piv").mkdir(exist_ok=True)
for name, data in uploaded.items():
    with open(f"/content/pump_piv/{name}", "wb") as fh:
        fh.write(data)

for c in ("C1", "C2", "C4", "C5", "C6"):
    p = f"/content/pump_piv/PumpDiffuser_{c}.csv"
    if not os.path.exists(p):
        raise RuntimeError(f"Missing: {p} -- please re-run cell 3.")
print("\\nAll 5 PIV CSVs uploaded.")
""")


# =====================================================================
# CELL 4 — Imports + constants
# =====================================================================
code("""import os, json, time, csv
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax, jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jaxopt import LBFGS

jax.config.update("jax_enable_x64", True)
print("JAX backend:", jax.default_backend(), "devices:", jax.devices())

# ---- Fluid + geometry constants (from FDA Blood Pump Instructions) ----
RHO = 1035.0          # kg/m^3
MU  = 0.0035          # N s/m^2 (3.5 cP)
NU  = MU / RHO
D_R = 0.052           # rotor diameter (m)

OP_TABLE = {
    "C1": (2.5, 2500, 261.799, 209338),
    "C2": (2.5, 3500, 366.519, 293073),
    "C4": (6.0, 2500, 261.799, 209338),
    "C5": (6.0, 3500, 366.519, 293073),
    "C6": (7.0, 3500, 366.519, 293073),
}

PIV_DIR = Path("/content/pump_piv")
OUT_DIR = Path("/content/B4_pump_outputs")
OUT_DIR.mkdir(exist_ok=True)
""")


# =====================================================================
# CELL 5 — Model + physics
# =====================================================================
code("""class MLP(nnx.Module):
    def __init__(self, din, dout, width, depth, *, rngs):
        sizes = [din] + [width] * depth + [dout]
        layers = [
            nnx.Linear(sizes[i], sizes[i + 1],
                       kernel_init=nnx.initializers.glorot_normal(),
                       bias_init=nnx.initializers.zeros_init(),
                       rngs=rngs)
            for i in range(len(sizes) - 1)
        ]
        self.layers = nnx.data(layers)

    def __call__(self, z):
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = jnp.tanh(z)
        return z


def normalize(x, y, S):
    x_n = 2.0 * (x - S["x_lo"]) / (S["x_hi"] - S["x_lo"]) - 1.0
    y_n = 2.0 * (y - S["y_lo"]) / (S["y_hi"] - S["y_lo"]) - 1.0
    return jnp.stack([x_n, y_n], axis=-1)


def uvp(model, x, y, S):
    z = normalize(x, y, S)
    raw = model(z)
    return (S["u_scale"] * raw[..., 0],
            S["u_scale"] * raw[..., 1],
            S["p_scale"] * raw[..., 2])


def _u(model, x, y, S): return uvp(model, x, y, S)[0]
def _v(model, x, y, S): return uvp(model, x, y, S)[1]
def _p(model, x, y, S): return uvp(model, x, y, S)[2]


def _grads(fn, model, x, y, S):
    fx  = jax.grad(fn, argnums=1)
    fy  = jax.grad(fn, argnums=2)
    fxx = jax.grad(fx, argnums=1)
    fyy = jax.grad(fy, argnums=2)
    return (fx(model, x, y, S), fy(model, x, y, S),
            fxx(model, x, y, S), fyy(model, x, y, S))


def pde_residuals(model, x, y, S):
    u, v, p = uvp(model, x, y, S)
    u_x, u_y, u_xx, u_yy = _grads(_u, model, x, y, S)
    v_x, v_y, v_xx, v_yy = _grads(_v, model, x, y, S)
    p_x = jax.grad(_p, argnums=1)(model, x, y, S)
    p_y = jax.grad(_p, argnums=2)(model, x, y, S)
    rx = u * u_x + v * u_y + (1.0 / RHO) * p_x - NU * (u_xx + u_yy)
    ry = u * v_x + v * v_y + (1.0 / RHO) * p_y - NU * (v_xx + v_yy)
    rc = u_x + v_y
    return rx, ry, rc


def total_loss(model, batch, S, weights):
    xd, yd, vd = batch["x_d"], batch["y_d"], batch["v_d"]
    u_d = jax.vmap(_u, in_axes=(None, 0, 0, None))(model, xd, yd, S)
    v_d = jax.vmap(_v, in_axes=(None, 0, 0, None))(model, xd, yd, S)
    mag = jnp.sqrt(u_d ** 2 + v_d ** 2 + 1e-12)
    err = jnp.mean((mag - vd) ** 2)
    norm = jnp.mean(vd ** 2) + 1e-12
    L_data = err / norm

    xi, yi = batch["x_i"], batch["y_i"]
    rx, ry, rc = jax.vmap(pde_residuals, in_axes=(None, 0, 0, None))(
        model, xi, yi, S)
    L = max(S["x_hi"] - S["x_lo"], S["y_hi"] - S["y_lo"])
    U = S["u_scale"]
    mom_scale = U ** 2 / L
    cont_scale = U / L
    L_mom_x = jnp.mean((rx / mom_scale) ** 2)
    L_mom_y = jnp.mean((ry / mom_scale) ** 2)
    L_cont  = jnp.mean((rc / cont_scale) ** 2)

    total = (weights["data"]  * L_data +
             weights["mom_x"] * L_mom_x +
             weights["mom_y"] * L_mom_y +
             weights["cont"]  * L_cont)
    return total, dict(data=L_data, mom_x=L_mom_x, mom_y=L_mom_y, cont=L_cont)
""")


# =====================================================================
# CELL 6 — Helpers
# =====================================================================
code("""def load_piv(case):
    xs, ys, vs = [], [], []
    with open(PIV_DIR / f"PumpDiffuser_{case}.csv") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["x_m"]))
            ys.append(float(row["y_m"]))
            vs.append(float(row["v_mag_m_per_s"]))
    return (np.asarray(xs), np.asarray(ys), np.asarray(vs))


def make_scales(xs, ys, vs):
    return dict(
        x_lo=float(xs.min()), x_hi=float(xs.max()),
        y_lo=float(ys.min()), y_hi=float(ys.max()),
        u_scale=float(np.percentile(vs, 95)),
        p_scale=RHO * float(np.percentile(vs, 95)) ** 2,
    )


def sample_interior(rng, n, S):
    rng_x, rng_y = jax.random.split(rng)
    x = jax.random.uniform(rng_x, (n,), minval=S["x_lo"], maxval=S["x_hi"])
    y = jax.random.uniform(rng_y, (n,), minval=S["y_lo"], maxval=S["y_hi"])
    return x, y


def stewart_eg_mag(model, xs, ys, vs, S, eps=1e-3):
    u = jax.vmap(_u, in_axes=(None, 0, 0, None))(model, xs, ys, S)
    v = jax.vmap(_v, in_axes=(None, 0, 0, None))(model, xs, ys, S)
    mag = np.asarray(jnp.sqrt(u ** 2 + v ** 2))
    keep = vs > eps * S["u_scale"]
    return float(np.mean(np.abs(mag[keep] - vs[keep]) / np.abs(vs[keep])))
""")


# =====================================================================
# CELL 7 — Main training loop
# =====================================================================
code("""ADAM_ITERS = 60_000
LBFGS_ITERS = 5_000
N_INTERIOR = 8_000
DATA_WEIGHT = 200.0
WEIGHTS = dict(data=DATA_WEIGHT, mom_x=1.0, mom_y=1.0, cont=1.0)

CASES = ["C1", "C2", "C4", "C5", "C6"]
ALL_METRICS = {}

for case in CASES:
    Q_lpm, rpm, omega, Re = OP_TABLE[case]
    print("\\n" + "=" * 70)
    print(f"  Condition {case}: Q={Q_lpm} L/min  RPM={rpm}  Re_pump={Re:,}")
    print("=" * 70)

    xs_d, ys_d, vs_d = load_piv(case)
    S = make_scales(xs_d, ys_d, vs_d)
    print(f"  PIV: {xs_d.size} points  |V|max={vs_d.max():.3f} m/s")
    print(f"  Scales: U={S['u_scale']:.3f} m/s, P={S['p_scale']:.1f} Pa")

    rngs = nnx.Rngs(0)
    model = MLP(din=2, dout=3, width=64, depth=6, rngs=rngs)
    schedule = optax.cosine_decay_schedule(1e-3, ADAM_ITERS, alpha=0.5)
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    xd = jnp.asarray(xs_d); yd = jnp.asarray(ys_d); vd = jnp.asarray(vs_d)

    @nnx.jit
    def train_step(model, optimizer, key):
        x_i, y_i = sample_interior(key, N_INTERIOR, S)
        batch = dict(x_d=xd, y_d=yd, v_d=vd, x_i=x_i, y_i=y_i)
        def fn(m):
            tot, parts = total_loss(m, batch, S, WEIGHTS)
            return tot, parts
        (tot, parts), grads = nnx.value_and_grad(fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return tot, parts

    t0 = time.time()
    rng = jax.random.PRNGKey(42)
    log_every = max(1, ADAM_ITERS // 10)
    for step in range(ADAM_ITERS):
        rng, subkey = jax.random.split(rng)
        tot, parts = train_step(model, optimizer, subkey)
        if step % log_every == 0 or step == ADAM_ITERS - 1:
            print(f"    Adam {step:6d}  tot={float(tot):.3e}  "
                  f"data={float(parts['data']):.2e}  "
                  f"cont={float(parts['cont']):.2e}")
    t_adam = time.time() - t0

    # L-BFGS finisher
    x_i, y_i = sample_interior(jax.random.PRNGKey(7), N_INTERIOR, S)
    batch_lbfgs = dict(x_d=xd, y_d=yd, v_d=vd, x_i=x_i, y_i=y_i)
    graphdef, state = nnx.split(model)
    def flat_loss(sf):
        m = nnx.merge(graphdef, sf)
        return total_loss(m, batch_lbfgs, S, WEIGHTS)[0]
    t0 = time.time()
    solver = LBFGS(fun=flat_loss, maxiter=LBFGS_ITERS, tol=1e-9)
    result = solver.run(state)
    state = result.params
    model = nnx.merge(graphdef, state)
    t_lbfgs = time.time() - t0

    # Evaluate
    u_pred = np.asarray(jax.vmap(_u, in_axes=(None, 0, 0, None))(model, xd, yd, S))
    v_pred = np.asarray(jax.vmap(_v, in_axes=(None, 0, 0, None))(model, xd, yd, S))
    mag_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    rel_L2 = 100.0 * np.linalg.norm(mag_pred - vs_d) / np.linalg.norm(vs_d)
    Eg = stewart_eg_mag(model, xd, yd, vd, S)

    print(f"\\n  rel L2 |V|       = {rel_L2:.2f} %")
    print(f"  Stewart E_g      = {Eg:.4f}")
    print(f"  |V|_peak pred    = {mag_pred.max():.3f}  PIV: {vs_d.max():.3f}")
    print(f"  Adam: {t_adam:.0f}s  LBFGS: {t_lbfgs:.0f}s")

    m = dict(
        case=case, Q_lpm=Q_lpm, rpm=rpm, Re_pump=Re,
        n_obs=int(vs_d.size),
        U_scale=S["u_scale"], P_scale=S["p_scale"],
        data_rel_L2_mag_pct=float(rel_L2),
        stewart_2012_Eg_mag=float(Eg),
        vmag_peak_predicted=float(mag_pred.max()),
        vmag_peak_pivdata=float(vs_d.max()),
        train_time_seconds=float(t_adam + t_lbfgs),
    )
    ALL_METRICS[case] = m
    with open(OUT_DIR / f"pump_metrics_{case}.json", "w") as fh:
        json.dump(m, fh, indent=2)
    np.savez_compressed(OUT_DIR / f"pump_predictions_{case}.npz",
                        x=xs_d, y=ys_d, v_piv=vs_d,
                        u_pred=u_pred, v_pred=v_pred, mag_pred=mag_pred)
""")


# =====================================================================
# CELL 8 — Figures + summary
# =====================================================================
code("""import matplotlib.pyplot as plt
import matplotlib.tri as mtri

CASES = ["C1", "C2", "C4", "C5", "C6"]

# ---- Per-condition velocity-magnitude fields (PIV vs PINN vs error) ----
for case in CASES:
    d = np.load(OUT_DIR / f"pump_predictions_{case}.npz")
    x, y, vp, mp = d["x"], d["y"], d["v_piv"], d["mag_pred"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    tri = mtri.Triangulation(x * 1000, y * 1000)   # plot in mm
    vmin, vmax = 0, max(vp.max(), mp.max())

    ax = axes[0]
    tc = ax.tripcolor(tri, vp, vmin=vmin, vmax=vmax, cmap="viridis", shading="gouraud")
    plt.colorbar(tc, ax=ax, label="|V| (m/s)"); ax.set_aspect("equal")
    ax.set_title(f"(a) PIV  |V| -- {case}"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    ax = axes[1]
    tc = ax.tripcolor(tri, mp, vmin=vmin, vmax=vmax, cmap="viridis", shading="gouraud")
    plt.colorbar(tc, ax=ax, label="|V| (m/s)"); ax.set_aspect("equal")
    ax.set_title(f"(b) PINN |V| -- {case}"); ax.set_xlabel("x (mm)")

    ax = axes[2]
    err = mp - vp
    a = max(abs(err.min()), abs(err.max()))
    tc = ax.tripcolor(tri, err, vmin=-a, vmax=a, cmap="RdBu_r", shading="gouraud")
    plt.colorbar(tc, ax=ax, label="error (m/s)"); ax.set_aspect("equal")
    ax.set_title(f"(c) PINN - PIV  -- {case}"); ax.set_xlabel("x (mm)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"pump_fields_{case}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote pump_fields_{case}.png")

# ---- Cross-case summary (scatter PINN vs PIV per condition) ----
fig, axes = plt.subplots(1, len(CASES), figsize=(4 * len(CASES), 4), sharey=True)
for i, case in enumerate(CASES):
    d = np.load(OUT_DIR / f"pump_predictions_{case}.npz")
    ax = axes[i]
    ax.scatter(d["v_piv"], d["mag_pred"], s=2, alpha=0.3, color="tab:purple")
    lo = 0; hi = max(d["v_piv"].max(), d["mag_pred"].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("|V|_PIV (m/s)")
    if i == 0: ax.set_ylabel("|V|_PINN (m/s)")
    m = ALL_METRICS[case]
    ax.set_title(f"{case}  rel L2={m['data_rel_L2_mag_pct']:.1f}%  "
                 f"$E_g$={m['stewart_2012_Eg_mag']:.2f}")
fig.suptitle("PINN vs PIV velocity magnitude, FDA Blood Pump diffuser", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "pump_scatter_summary.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  wrote pump_scatter_summary.png")

# ---- Master JSON ----
with open(OUT_DIR / "pump_all_metrics.json", "w") as fh:
    json.dump(ALL_METRICS, fh, indent=2)
print("  wrote pump_all_metrics.json")
print("\\n===== FINAL METRICS =====")
for case in CASES:
    m = ALL_METRICS[case]
    print(f"  {case}: rel L2 = {m['data_rel_L2_mag_pct']:5.2f}%  "
          f"E_g = {m['stewart_2012_Eg_mag']:.4f}  "
          f"({m['train_time_seconds']:.0f}s)")
""")


# =====================================================================
# CELL 9 — Package + download
# =====================================================================
code("""import shutil
from google.colab import files

zip_path = "/content/B4_pump_results"
shutil.make_archive(zip_path, "zip", OUT_DIR)
files.download(zip_path + ".zip")
print("Downloaded B4_pump_results.zip")
""")


# =====================================================================
# Write notebook
# =====================================================================
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "cells": CELLS,
}

out = Path(__file__).parent / "B4_FDA_Pump_Diffuser_PINN_Colab.ipynb"
with open(out, "w") as fh:
    json.dump(nb, fh, indent=2)
size = out.stat().st_size / 1024
print(f"wrote {out}  ({size:.1f} KB, {len(CELLS)} cells)")
