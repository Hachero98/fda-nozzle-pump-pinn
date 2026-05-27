"""
fda_pump_diffuser_PINN.py
=========================
2D PINN for the FDA Blood Pump diffuser benchmark (A2 of the B4 paper).

Geometry:  2D Cartesian slice (x, y) through the diffuser region.
Frame:     Stationary lab frame (no Coriolis / centrifugal source terms).
PDE:       Steady incompressible Navier-Stokes:
             u u_x + v u_y = - (1/rho) p_x + nu (u_xx + u_yy)
             u v_x + v v_y = - (1/rho) p_y + nu (v_xx + v_yy)
             u_x + v_y = 0
Data loss: magnitude-only,  L_data = mean( (sqrt(u^2 + v^2) - |V|_PIV)^2 )
           (no per-component info available; physics drives direction recovery).
BC:        soft Dirichlet on the data points themselves; we do not enforce
           explicit no-slip walls because the diffuser domain bounds are
           defined by the PIV bounding box rather than a known geometry.

Design mirrors fda_nozzle_PINN.py (B4 paper A1) for code parity:
  - Flax NNX MLP, tanh, hard input normalization to [-1, +1]^2
  - Output scaling by U_SCALE / P_SCALE = rho * U_SCALE^2
  - Optax Adam (cosine schedule 1e-3 -> 5e-4) + jaxopt L-BFGS finisher
  - Stewart 2012 E_g metric for one-to-one comparability with A1

Usage:
  python fda_pump_diffuser_PINN.py --case C5
  python fda_pump_diffuser_PINN.py --case C5 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# JAX setup
# ---------------------------------------------------------------------------
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jaxopt import LBFGS

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fluid + operating-condition tables (from PUMP_NOTES.md)
# ---------------------------------------------------------------------------
RHO = 1035.0          # kg/m^3
MU  = 0.0035          # N s/m^2
NU  = MU / RHO        # kinematic viscosity (m^2/s)
D_R = 0.052           # rotor diameter (m)

# (Q [L/min], RPM, omega [rad/s], Re_pump)
OP_TABLE = {
    "C1": (2.5, 2500, 261.799, 209338),
    "C2": (2.5, 3500, 366.519, 293073),
    "C4": (6.0, 2500, 261.799, 209338),
    "C5": (6.0, 3500, 366.519, 293073),
    "C6": (7.0, 3500, 366.519, 293073),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_piv(case: str, root: Path):
    csv_path = root / "Reference Papers/FDA_Dataset/Pump" / f"PumpDiffuser_{case}.csv"
    if not csv_path.exists():
        sys.exit(f"PIV CSV not found: {csv_path}\n"
                 f"Run parse_pump_diffuser_piv.py first.")
    import csv as _csv
    xs, ys, vs = [], [], []
    with open(csv_path) as fh:
        for row in _csv.DictReader(fh):
            xs.append(float(row["x_m"]))
            ys.append(float(row["y_m"]))
            vs.append(float(row["v_mag_m_per_s"]))
    return (np.asarray(xs, dtype=np.float64),
            np.asarray(ys, dtype=np.float64),
            np.asarray(vs, dtype=np.float64))


# ---------------------------------------------------------------------------
# Model: 2D MLP with hard input normalization & output scaling
# ---------------------------------------------------------------------------
class MLP(nnx.Module):
    def __init__(self, din: int, dout: int, width: int, depth: int,
                 *, rngs: nnx.Rngs):
        sizes = [din] + [width] * depth + [dout]
        layers = [
            nnx.Linear(
                sizes[i], sizes[i + 1],
                kernel_init=nnx.initializers.glorot_normal(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
            for i in range(len(sizes) - 1)
        ]
        self.layers = nnx.data(layers)

    def __call__(self, z):
        for i, layer in enumerate(self.layers):
            z = layer(z)
            if i < len(self.layers) - 1:
                z = jnp.tanh(z)
        return z


def make_scales(xs, ys, vs):
    """Compute normalization & physical scales from the data."""
    x_lo, x_hi = float(xs.min()), float(xs.max())
    y_lo, y_hi = float(ys.min()), float(ys.max())
    u_scale = float(np.percentile(vs, 95))          # robust peak speed
    p_scale = RHO * u_scale ** 2                     # dynamic pressure
    return dict(x_lo=x_lo, x_hi=x_hi, y_lo=y_lo, y_hi=y_hi,
                u_scale=u_scale, p_scale=p_scale)


def normalize(x, y, S):
    x_n = 2.0 * (x - S["x_lo"]) / (S["x_hi"] - S["x_lo"]) - 1.0
    y_n = 2.0 * (y - S["y_lo"]) / (S["y_hi"] - S["y_lo"]) - 1.0
    return jnp.stack([x_n, y_n], axis=-1)


def uvp(model, x, y, S):
    """Return (u, v, p) at (x, y), with output scaling baked in."""
    z = normalize(x, y, S)
    raw = model(z)
    u = S["u_scale"] * raw[..., 0]
    v = S["u_scale"] * raw[..., 1]
    p = S["p_scale"] * raw[..., 2]
    return u, v, p


# ---------------------------------------------------------------------------
# Derivatives (autodiff)
# ---------------------------------------------------------------------------
def _u(model, x, y, S):
    return uvp(model, x, y, S)[0]

def _v(model, x, y, S):
    return uvp(model, x, y, S)[1]

def _p(model, x, y, S):
    return uvp(model, x, y, S)[2]


def _grads(fn, model, x, y, S):
    """First & second partial derivatives of scalar fn(model, x, y, S) wrt (x,y)."""
    fx  = jax.grad(fn, argnums=1)
    fy  = jax.grad(fn, argnums=2)
    fxx = jax.grad(fx, argnums=1)
    fyy = jax.grad(fy, argnums=2)
    return fx(model, x, y, S), fy(model, x, y, S), \
           fxx(model, x, y, S), fyy(model, x, y, S)


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


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def total_loss(model, batch, S, weights):
    # ---- Data loss: magnitude-only ----
    xd, yd, vd = batch["x_d"], batch["y_d"], batch["v_d"]
    u_d = jax.vmap(_u, in_axes=(None, 0, 0, None))(model, xd, yd, S)
    v_d = jax.vmap(_v, in_axes=(None, 0, 0, None))(model, xd, yd, S)
    mag = jnp.sqrt(u_d ** 2 + v_d ** 2 + 1e-12)
    err = jnp.mean((mag - vd) ** 2)
    norm = jnp.mean(vd ** 2) + 1e-12
    L_data = err / norm                              # rel. L^2 squared

    # ---- PDE residuals at interior collocation points ----
    xi, yi = batch["x_i"], batch["y_i"]
    rx, ry, rc = jax.vmap(pde_residuals, in_axes=(None, 0, 0, None))(
        model, xi, yi, S
    )
    # Non-dimensionalize residuals by U_SCALE / L_SCALE so they're O(1)
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


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------
def sample_interior(rng, n, S):
    rng_x, rng_y = jax.random.split(rng)
    x = jax.random.uniform(rng_x, (n,), minval=S["x_lo"], maxval=S["x_hi"])
    y = jax.random.uniform(rng_y, (n,), minval=S["y_lo"], maxval=S["y_hi"])
    return x, y


def stewart_eg_magnitude(model, xs, ys, vs, S, eps=1e-3):
    """Stewart 2012 E_g = MARE applied to velocity magnitude."""
    u = jax.vmap(_u, in_axes=(None, 0, 0, None))(model, xs, ys, S)
    v = jax.vmap(_v, in_axes=(None, 0, 0, None))(model, xs, ys, S)
    mag = np.asarray(jnp.sqrt(u ** 2 + v ** 2))
    keep = vs > eps * S["u_scale"]
    if not keep.any():
        return float("nan")
    return float(np.mean(np.abs(mag[keep] - vs[keep]) / np.abs(vs[keep])))


def train_one(case: str, root: Path, out_dir: Path, quick: bool = False):
    print(f"\n{'='*60}\n  FDA Blood Pump Diffuser PINN -- Condition {case}\n{'='*60}")

    Q_lpm, rpm, omega, Re = OP_TABLE[case]
    print(f"  Flow = {Q_lpm} L/min  Speed = {rpm} rpm  Re_pump = {Re:,}")

    xs_d, ys_d, vs_d = load_piv(case, root)
    print(f"  PIV points: {xs_d.size}  |V| range = [{vs_d.min():.3f}, "
          f"{vs_d.max():.3f}] m/s")

    S = make_scales(xs_d, ys_d, vs_d)
    print(f"  Domain: x in [{S['x_lo']:.4f}, {S['x_hi']:.4f}] m, "
          f"y in [{S['y_lo']:.4f}, {S['y_hi']:.4f}] m")
    print(f"  Scales: U = {S['u_scale']:.3f} m/s,  P = {S['p_scale']:.1f} Pa")

    # ---- hyperparameters ----
    if quick:
        adam_iters, lbfgs_iters, n_interior = 1500, 200, 800
    else:
        adam_iters, lbfgs_iters, n_interior = 60_000, 5_000, 8_000
    data_weight = 200.0
    weights = dict(data=data_weight, mom_x=1.0, mom_y=1.0, cont=1.0)
    print(f"  Adam {adam_iters} | L-BFGS {lbfgs_iters} | N_interior {n_interior}")
    print(f"  Weights: {weights}")

    # ---- model & optimizer ----
    rngs = nnx.Rngs(0)
    model = MLP(din=2, dout=3, width=64, depth=6, rngs=rngs)
    schedule = optax.cosine_decay_schedule(1e-3, adam_iters, alpha=0.5)
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    # JIT-able tensors
    xd, yd, vd = jnp.asarray(xs_d), jnp.asarray(ys_d), jnp.asarray(vs_d)

    @nnx.jit
    def train_step(model, optimizer, key):
        x_i, y_i = sample_interior(key, n_interior, S)
        batch = dict(x_d=xd, y_d=yd, v_d=vd, x_i=x_i, y_i=y_i)
        def fn(m):
            tot, parts = total_loss(m, batch, S, weights)
            return tot, parts
        (tot, parts), grads = nnx.value_and_grad(fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return tot, parts

    # ---- Adam phase ----
    t0 = time.time()
    rng = jax.random.PRNGKey(42)
    log_every = max(1, adam_iters // 12)
    for step in range(adam_iters):
        rng, subkey = jax.random.split(rng)
        tot, parts = train_step(model, optimizer, subkey)
        if step % log_every == 0 or step == adam_iters - 1:
            print(f"    Adam {step:6d}  total={float(tot):.4e}  "
                  f"data={float(parts['data']):.3e}  "
                  f"mom={float(parts['mom_x']) + float(parts['mom_y']):.3e}  "
                  f"cont={float(parts['cont']):.3e}")
    t_adam = time.time() - t0
    print(f"  Adam: {t_adam:.1f}s")

    # ---- L-BFGS phase (final polish on data + PDE) ----
    if lbfgs_iters > 0:
        x_i, y_i = sample_interior(jax.random.PRNGKey(7), n_interior, S)
        batch_lbfgs = dict(x_d=xd, y_d=yd, v_d=vd, x_i=x_i, y_i=y_i)
        graphdef, state = nnx.split(model)

        def flat_loss(state_flat):
            m = nnx.merge(graphdef, state_flat)
            tot, _ = total_loss(m, batch_lbfgs, S, weights)
            return tot

        t0 = time.time()
        solver = LBFGS(fun=flat_loss, maxiter=lbfgs_iters, tol=1e-9)
        result = solver.run(state)
        state = result.params
        model = nnx.merge(graphdef, state)
        print(f"  L-BFGS: {time.time() - t0:.1f}s  final={float(result.state.value):.4e}")

    # ---- Evaluate ----
    u_pred = np.asarray(jax.vmap(_u, in_axes=(None, 0, 0, None))(model, xd, yd, S))
    v_pred = np.asarray(jax.vmap(_v, in_axes=(None, 0, 0, None))(model, xd, yd, S))
    mag_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    rel_L2 = 100.0 * np.linalg.norm(mag_pred - vs_d) / max(np.linalg.norm(vs_d), 1e-12)
    Eg = stewart_eg_magnitude(model, xd, yd, vd, S)

    print(f"\n  rel L2 |V|         = {rel_L2:.2f} %")
    print(f"  Stewart-2012 E_g   = {Eg:.4f}")
    print(f"  |V|_peak  PINN     = {mag_pred.max():.3f} m/s")
    print(f"  |V|_peak  PIV      = {vs_d.max():.3f} m/s")

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = dict(
        case=case, Q_lpm=Q_lpm, rpm=rpm, Re_pump=Re,
        n_obs=int(vs_d.size),
        U_scale_m_per_s=S["u_scale"], P_scale_Pa=S["p_scale"],
        data_rel_L2_mag_pct=float(rel_L2),
        stewart_2012_Eg_mag=float(Eg),
        vmag_peak_predicted=float(mag_pred.max()),
        vmag_peak_pivdata=float(vs_d.max()),
        hyperparameters=dict(
            architecture="6 hidden x 64 units, tanh, hard-norm",
            adam_iters=adam_iters, lbfgs_iters=lbfgs_iters,
            n_interior=n_interior,
            lr_schedule="cosine 1e-3 -> 5e-4",
            data_loss="magnitude-only relative L^2",
            weights=weights,
        ),
    )
    metrics_path = out_dir / f"pump_metrics_{case}.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  wrote {metrics_path}")

    np.savez_compressed(
        out_dir / f"pump_predictions_{case}.npz",
        x=xs_d, y=ys_d, v_piv=vs_d,
        u_pred=u_pred, v_pred=v_pred, mag_pred=mag_pred,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", default="C5", choices=list(OP_TABLE))
    p.add_argument("--all", action="store_true",
                   help="train all 5 conditions sequentially")
    p.add_argument("--quick", action="store_true",
                   help="sanity-check run (tiny budget)")
    p.add_argument("--out", default="B4_pump_results",
                   help="output directory for metrics + predictions")
    args = p.parse_args()

    root = Path(__file__).parent
    out_dir = root / args.out

    cases = list(OP_TABLE) if args.all else [args.case]
    for c in cases:
        train_one(c, root, out_dir, quick=args.quick)


if __name__ == "__main__":
    main()
