"""
fda_nozzle_PINN.py
==================
Physics-Informed Neural Network solver for the FDA Critical Path
Nozzle benchmark, corresponding to paper B4:
    "Physics-Informed Neural Networks for Full-Field Hemodynamic
     Reconstruction from Sparse Experimental PIV Data: Validation
     on the FDA Critical Path Nozzle Benchmark"

Implementation mirrors the B3 dopamine PINN stack:
JAX + Flax NNX (network), Optax (Adam), jaxopt (L-BFGS), float64.

GOVERNING PDE — 2D axisymmetric steady incompressible Navier-Stokes
in cylindrical coordinates (x, r), with no swirl (u_theta = 0).

Let bu = (u_x, u_r), p = p(x, r). With kinematic viscosity nu = mu / rho:

  Continuity:
      d u_x / d x  +  (1/r) d (r u_r) / d r  =  0
    = d u_x / d x  +  d u_r / d r  +  u_r / r  =  0

  Axial momentum:
      u_x d u_x / d x  +  u_r d u_x / d r
      = -(1/rho) d p / d x
        + nu [ d^2 u_x / d x^2  +  d^2 u_x / d r^2  +  (1/r) d u_x / d r ]

  Radial momentum:
      u_x d u_r / d x  +  u_r d u_r / d r
      = -(1/rho) d p / d r
        + nu [ d^2 u_r / d x^2  +  d^2 u_r / d r^2  +  (1/r) d u_r / d r
               -  u_r / r^2 ]

The 1/r terms are regularised by replacing r with r + R_EPS for residual
evaluation; axisymmetry is enforced separately as a hard boundary
condition on r = 0 (u_r = 0, d u_x / d r = 0).

GEOMETRY — Hariharan 2011 FDA benchmark, "sudden-expansion" orientation:
    -0.088 m  --  inlet pipe  --  -0.022675 m
    -0.022675 --  conical contraction (20 deg half-angle) --  0.0 m
        0.0   --  throat (radius 0.002 m)              --  0.040 m
        0.040 --  sudden expansion to inlet radius
        0.040 --  outlet pipe                          --  0.143 m
        (outlet length aligns with Zenodo CFD axial extent after the
         throat-centred x_shift = 0.0565 m applied by the converter)
The R(x) function below encodes this profile; tweak if the user is
using the alternative "sudden-contraction" orientation.

FLUID — 40 % glycerol-water blood analogue:
    rho = 1056 kg/m^3
    mu  = 3.5e-3 Pa s   (= 0.0035)
    nu  = mu / rho      (kinematic, used inside the residual)

REYNOLDS NUMBERS — Re based on throat diameter d = 0.004 m:
    Re = 500   (laminar)
    Re = 2000  (transitional inlet, laminar throat)
    Re = 3500  (transitional, separation in expansion)

HOW TO RUN
----------
Local:
    pip install -r ../requirements.txt
    python fda_nozzle_PINN.py --re 500 --quick   # sanity smoke test
    python fda_nozzle_PINN.py --re 500           # paper-quality run

Synthetic-data fallback (works without the nciphub PIV CSVs):
    python fda_nozzle_PINN.py --re 500 --synthetic-piv

With real PIV (drop CSV at Reference Papers/FDA_Dataset/PIV_Re500.csv):
    python fda_nozzle_PINN.py --re 500 \
        --piv-csv "Reference Papers/FDA_Dataset/PIV_Re500.csv"

Outputs:  ./figures/  and  ./figures/metrics.json
"""

# =============================================================
# 0. Imports and reproducibility
# =============================================================
import os
import time
import json
import argparse
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

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


# =============================================================
# 1. Physical constants (Hariharan 2011, Table 1; blood analogue)
# =============================================================
RHO = 1056.0          # kg / m^3       fluid density
MU  = 3.5e-3          # Pa s           dynamic viscosity
NU  = MU / RHO        # m^2 / s        kinematic viscosity

# Throat reference dimensions
D_THROAT = 0.004      # m              throat diameter
R_THROAT = D_THROAT / 2.0
D_INLET  = 0.012      # m              inlet/outlet pipe diameter
R_INLET  = D_INLET / 2.0

# Axial extents (sudden-expansion orientation)
X_INLET_START   = -0.088
X_CONTRACT_START = -0.022675
X_THROAT_START  = 0.0
X_THROAT_END    = 0.040
X_OUTLET_END    = 0.143       # extended to match Zenodo CFD domain

# Numerical regularisation
R_EPS = 1e-8          # added to r when forming 1/r terms in residual

# ---------------------------------------------------------------
# Output scaling (Wang 2021 §4 — fix for gradient pathology).
# Network outputs are O(1); we multiply by physical reference scales
# inside uvp_at so downstream loss terms see physical units.
# Defaults = 1.0 (no scaling, backward compatible). main() sets these
# from args.re before model construction so the JIT cache picks them up.
# ---------------------------------------------------------------
U_SCALE = 1.0          # m/s   axial + radial velocity scale
P_SCALE = 1.0          # Pa    pressure scale = rho * U_SCALE^2

# Input normalisation: map (x, r) -> [-1, +1]^2 before feeding the MLP.
# Without this, tanh activations see |x| < 0.15 and |r| < 0.006 and stay
# in their linear regime, so the network can't represent sharp radial
# profiles. These two constants are derived from the scaffold geometry
# at module load and don't depend on Re.
X_MIN_NORM = X_INLET_START
X_MAX_NORM = X_OUTLET_END
R_MIN_NORM = 0.0
R_MAX_NORM = R_INLET

def _normalize_xr(x, r):
    x_n = 2.0 * (x - X_MIN_NORM) / (X_MAX_NORM - X_MIN_NORM) - 1.0
    r_n = 2.0 * (r - R_MIN_NORM) / (R_MAX_NORM - R_MIN_NORM) - 1.0
    return x_n, r_n


# =============================================================
# 2. Wall geometry R(x) for the FDA benchmark
# =============================================================
def wall_radius(x):
    """Piecewise wall radius R(x) for sudden-expansion FDA nozzle.

    Works for scalar or array inputs. Vectorised with jnp.where so it
    is differentiable inside jit.
    """
    # Contraction: linear from R_INLET to R_THROAT over the cone region
    contract_frac = (x - X_CONTRACT_START) / (X_THROAT_START - X_CONTRACT_START)
    contract_frac = jnp.clip(contract_frac, 0.0, 1.0)
    R_contract = R_INLET + (R_THROAT - R_INLET) * contract_frac

    # Throat: constant R_THROAT
    # Outlet: jumps back to R_INLET at X_THROAT_END (sudden expansion)
    R = jnp.where(x < X_CONTRACT_START, R_INLET,
        jnp.where(x < X_THROAT_START,   R_contract,
        jnp.where(x < X_THROAT_END,     R_THROAT,
                                         R_INLET)))
    return R


def in_domain(x, r):
    """Boolean mask: True where (x, r) lies inside the nozzle lumen."""
    return (x >= X_INLET_START) & (x <= X_OUTLET_END) & \
           (r >= 0.0) & (r <= wall_radius(x))


# =============================================================
# 3. Reference velocity and Reynolds-number bookkeeping
# =============================================================
def mean_throat_velocity(Re):
    """U_mean at the throat for a given throat Reynolds number."""
    return Re * MU / (RHO * D_THROAT)


def hagen_poiseuille_inlet(r, U_mean):
    """Fully developed parabolic profile at the inlet pipe.

    Mean velocity in the inlet pipe = U_mean * (R_THROAT/R_INLET)^2 by
    conservation of mass.
    """
    U_inlet_mean = U_mean * (R_THROAT / R_INLET) ** 2
    U_max = 2.0 * U_inlet_mean
    return U_max * (1.0 - (r / R_INLET) ** 2)


# =============================================================
# 4. PINN: Flax NNX MLP with optional Fourier-feature input encoding
# =============================================================
# Defaults: 8 hidden layers of 128, with 32 Fourier features (-> 64-dim
# encoded input after sin/cos concat). Matches published PINN-for-FDA-
# nozzle architectures. Overrideable via --width / --depth / --n-fourier.
LAYERS = [2, 128, 128, 128, 128, 128, 128, 128, 128, 3]
N_FOURIER = 32          # 32 frequencies -> 64-dim input after sin+cos
FOURIER_SCALE = 5.0     # std of frequency distribution (cycles per unit)


class FourierFeatures(nnx.Module):
    """Random Fourier feature input encoding (Tancik 2020).

    Maps R^d input to R^{2K} via [sin(2 pi B x), cos(2 pi B x)] where
    B in R^{d x K} is sampled from N(0, scale^2) at init time and
    held FIXED. Helps tanh MLPs represent high-frequency spatial
    structure that they otherwise smooth away.
    """
    def __init__(self, in_dim, n_features, scale, *, rngs: nnx.Rngs):
        key = rngs.params()
        # B is a fixed constant array, not a learnable parameter
        self.B = scale * jax.random.normal(
            key, (in_dim, n_features), dtype=jnp.float64
        )

    def __call__(self, x):
        proj = 2.0 * jnp.pi * (x @ self.B)
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)


class MLP(nnx.Module):
    """Tanh MLP with optional Fourier feature input encoding.

    Stores layers as named attributes (lin_0, lin_1, ...) for Flax NNX
    compatibility. If n_fourier > 0, prepends a FourierFeatures encoder
    that replaces the first input dimension count from layers[0] with
    2*n_fourier.
    """
    def __init__(self, layers, *, rngs: nnx.Rngs,
                 n_fourier=0, fourier_scale=1.0):
        if n_fourier > 0:
            self.ff = FourierFeatures(layers[0], n_fourier,
                                      fourier_scale, rngs=rngs)
            eff_layers = [2 * n_fourier] + list(layers[1:])
        else:
            self.ff = None
            eff_layers = list(layers)
        self.n_layers = len(eff_layers) - 1
        for i in range(self.n_layers):
            setattr(self, f"lin_{i}",
                    nnx.Linear(eff_layers[i], eff_layers[i + 1],
                               kernel_init=nnx.initializers.glorot_normal(),
                               bias_init=nnx.initializers.zeros_init(),
                               param_dtype=jnp.float64,
                               rngs=rngs))

    def __call__(self, xr):
        h = self.ff(xr) if self.ff is not None else xr
        for i in range(self.n_layers - 1):
            h = jnp.tanh(getattr(self, f"lin_{i}")(h))
        return getattr(self, f"lin_{self.n_layers - 1}")(h)


USE_HARD_BC = True   # Hard-enforce wall no-slip + axisymmetry. main() may flip.


def uvp_at(model, x, r):
    """Predicted (u_x, u_r, p) at a single point, in physical units.

    Pipeline:  (x, r) -> normalise to [-1, +1]^2 -> tanh MLP ->
                 apply hard BCs (optional) -> multiply by scales ->
                 physical (u_x, u_r, p).

    Hard BCs (USE_HARD_BC=True, default):
        Let phi(x, r) = 1 - (r/R(x))^2  ∈ [0, 1], zero at the wall.

        u_x = U_SCALE * phi(x, r) * Net_0(x, r)
             -> guarantees u_x = 0 at the wall (no-slip).
        u_r = U_SCALE * (r/R(x)) * phi(x, r) * Net_1(x, r)
             -> guarantees u_r = 0 at the wall AND u_r = 0 at the axis
                (axisymmetry).
        p   = P_SCALE * Net_2(x, r)
             -> no hard constraint; outlet/inlet pressure handled in
                soft losses.

    With hard BCs the wall_loss and axis_loss collapse to ~0 by
    construction, removing them from the soft-loss competition that
    previously trapped the optimizer in trivial solutions.
    """
    x_n, r_n = _normalize_xr(x, r)
    out = model(jnp.stack([x_n, r_n]))
    if USE_HARD_BC:
        R_w = wall_radius(x)
        rho_r = r / R_w
        phi = 1.0 - rho_r ** 2
        u_x_phys = U_SCALE * phi * out[0]
        u_r_phys = U_SCALE * rho_r * phi * out[1]
        p_phys   = P_SCALE * out[2]
        return u_x_phys, u_r_phys, p_phys
    return U_SCALE * out[0], U_SCALE * out[1], P_SCALE * out[2]


def _ux(model, x, r):
    return uvp_at(model, x, r)[0]


def _ur(model, x, r):
    return uvp_at(model, x, r)[1]


def _p(model, x, r):
    return uvp_at(model, x, r)[2]


# =============================================================
# 5. PDE residual at one point — steady axisymmetric NS
# =============================================================
def _ns_residual_one(model, x, r):
    """Returns (R_continuity, R_x_momentum, R_r_momentum) at (x, r).

    First and second derivatives are obtained via jax.grad. The 1/r
    singular terms use r + R_EPS for numerical stability; axisymmetry
    is enforced separately as a BC.
    """
    # First derivatives
    dux_dx = jax.grad(_ux, argnums=1)(model, x, r)
    dux_dr = jax.grad(_ux, argnums=2)(model, x, r)
    dur_dx = jax.grad(_ur, argnums=1)(model, x, r)
    dur_dr = jax.grad(_ur, argnums=2)(model, x, r)
    dp_dx  = jax.grad(_p,  argnums=1)(model, x, r)
    dp_dr  = jax.grad(_p,  argnums=2)(model, x, r)

    # Second derivatives
    d2ux_dx2 = jax.grad(lambda xx: jax.grad(_ux, argnums=1)(model, xx, r))(x)
    d2ux_dr2 = jax.grad(lambda rr: jax.grad(_ux, argnums=2)(model, x, rr))(r)
    d2ur_dx2 = jax.grad(lambda xx: jax.grad(_ur, argnums=1)(model, xx, r))(x)
    d2ur_dr2 = jax.grad(lambda rr: jax.grad(_ur, argnums=2)(model, x, rr))(r)

    ux, ur, _ = uvp_at(model, x, r)
    r_safe = r + R_EPS

    # Continuity
    R_c = dux_dx + dur_dr + ur / r_safe

    # Axial momentum
    R_x = (ux * dux_dx + ur * dux_dr
           + (1.0 / RHO) * dp_dx
           - NU * (d2ux_dx2 + d2ux_dr2 + dux_dr / r_safe))

    # Radial momentum
    R_r = (ux * dur_dx + ur * dur_dr
           + (1.0 / RHO) * dp_dr
           - NU * (d2ur_dx2 + d2ur_dr2 + dur_dr / r_safe - ur / r_safe ** 2))

    return R_c, R_x, R_r


# =============================================================
# 6. Point sampling (one-shot, like B3)
# =============================================================
def sample_points(n_interior, n_wall, n_inlet, n_outlet, n_axis, rng):
    """Sample collocation points inside the lumen and on each boundary.

    Interior points are drawn by rejection sampling on the (x, r) box
    [X_INLET_START, X_OUTLET_END] x [0, R_INLET] and kept if in_domain.
    """
    # ---- Interior (rejection sampling) ----
    xs, rs = [], []
    while len(xs) < n_interior:
        batch = max(1000, n_interior - len(xs))
        cand_x = rng.uniform(X_INLET_START, X_OUTLET_END, batch)
        cand_r = rng.uniform(0.0, R_INLET, batch)
        mask = np.asarray(in_domain(jnp.asarray(cand_x), jnp.asarray(cand_r)))
        xs.extend(cand_x[mask].tolist())
        rs.extend(cand_r[mask].tolist())
    x_in = np.array(xs[:n_interior])
    r_in = np.array(rs[:n_interior])

    # ---- Wall (r = R(x)) ----
    x_w = rng.uniform(X_INLET_START, X_OUTLET_END, n_wall)
    r_w = np.asarray(wall_radius(jnp.asarray(x_w)))

    # ---- Inlet (x = X_INLET_START, r in [0, R_INLET]) ----
    x_inl = np.full(n_inlet, X_INLET_START)
    r_inl = rng.uniform(0.0, R_INLET, n_inlet)

    # ---- Outlet (x = X_OUTLET_END, r in [0, R_INLET]) ----
    x_out = np.full(n_outlet, X_OUTLET_END)
    r_out = rng.uniform(0.0, R_INLET, n_outlet)

    # ---- Axis (r = 0, x in [X_INLET_START, X_OUTLET_END]) ----
    x_ax = rng.uniform(X_INLET_START, X_OUTLET_END, n_axis)
    r_ax = np.zeros(n_axis)

    return {
        "x_in":  jnp.asarray(x_in),  "r_in":  jnp.asarray(r_in),
        "x_w":   jnp.asarray(x_w),   "r_w":   jnp.asarray(r_w),
        "x_inl": jnp.asarray(x_inl), "r_inl": jnp.asarray(r_inl),
        "x_out": jnp.asarray(x_out), "r_out": jnp.asarray(r_out),
        "x_ax":  jnp.asarray(x_ax),  "r_ax":  jnp.asarray(r_ax),
    }


# =============================================================
# 7. PIV data — loader + synthetic fallback
# =============================================================
def load_piv_csv(path, re_filter=None):
    """Load PIV CSV produced by parse_fda_piv.py or vtu_to_pseudo_piv.py.

    Columns:
        x [m], r [m], u_x [m/s], u_r [m/s], Re [optional]

    Blank u_x or u_r cells are treated as 'not measured at this point'
    and excluded from the data loss via mask arrays. This is important
    for the real Hariharan files where many rows carry only one
    velocity component (axial-velocity profiles and radial-velocity
    profiles were measured at slightly different grid points, so they
    rarely coincide exactly).

    Returns dict of jnp arrays:
        x_d, r_d        location of each measurement
        ux_d, ur_d      values (zero-filled where blank)
        mux, mur        1.0 where the corresponding value is real,
                        0.0 where the cell was blank
    """
    import csv
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
            if ux_str:
                uxs.append(float(ux_str)); mux.append(1.0)
            else:
                uxs.append(0.0);           mux.append(0.0)
            if ur_str:
                urs.append(float(ur_str)); mur.append(1.0)
            else:
                urs.append(0.0);           mur.append(0.0)
    return {
        "x_d":  jnp.asarray(xs),
        "r_d":  jnp.asarray(rs),
        "ux_d": jnp.asarray(uxs),
        "ur_d": jnp.asarray(urs),
        "mux":  jnp.asarray(mux),
        "mur":  jnp.asarray(mur),
    }


def synthetic_piv(n_obs, Re, rng, noise_pct=1.0):
    """Synthetic 'pseudo-PIV' for smoke-testing the loss before real data.

    Samples points inside the inlet and outlet pipe regions (where the
    flow is approximately Hagen-Poiseuille) and assigns the parabolic
    axial velocity profile. Throat and expansion samples get a coarse
    extrapolation that is physically wrong but loss-compatible.
    """
    U_mean = mean_throat_velocity(Re)

    xs, rs, uxs = [], [], []
    while len(xs) < n_obs:
        batch = max(500, n_obs - len(xs))
        cand_x = rng.uniform(X_INLET_START, X_OUTLET_END, batch)
        cand_r = rng.uniform(0.0, R_INLET, batch)
        mask = np.asarray(in_domain(jnp.asarray(cand_x), jnp.asarray(cand_r)))
        keep_x = cand_x[mask]
        keep_r = cand_r[mask]
        for xv, rv in zip(keep_x, keep_r):
            if xv < X_THROAT_START or xv > X_THROAT_END:
                ux = float(hagen_poiseuille_inlet(jnp.asarray(rv), U_mean))
            else:
                # Crude throat estimate (top-hat-ish, parabolic across throat)
                ux = 2.0 * U_mean * (1.0 - (rv / R_THROAT) ** 2) \
                     if rv <= R_THROAT else 0.0
            xs.append(float(xv))
            rs.append(float(rv))
            uxs.append(ux)
            if len(xs) >= n_obs:
                break

    xs  = np.array(xs[:n_obs])
    rs  = np.array(rs[:n_obs])
    uxs = np.array(uxs[:n_obs])
    sigma = (noise_pct / 100.0) * U_mean
    uxs_noisy = uxs + rng.normal(0.0, sigma, n_obs)
    urs_noisy = rng.normal(0.0, sigma, n_obs)   # small radial component
    # Synthetic data always provides both components -> masks all-ones
    return {
        "x_d":  jnp.asarray(xs),
        "r_d":  jnp.asarray(rs),
        "ux_d": jnp.asarray(uxs_noisy),
        "ur_d": jnp.asarray(urs_noisy),
        "mux":  jnp.ones(n_obs),
        "mur":  jnp.ones(n_obs),
    }


# =============================================================
# 8. Loss assembly
# =============================================================
def _residual_losses(model, pts):
    """Sum-of-squares of the three NS residual components."""
    R_c, R_x, R_r = jax.vmap(
        _ns_residual_one, in_axes=(None, 0, 0)
    )(model, pts["x_in"], pts["r_in"])
    return jnp.mean(R_c ** 2), jnp.mean(R_x ** 2), jnp.mean(R_r ** 2)


def _wall_loss(model, pts):
    """No-slip on the wall: u_x = u_r = 0."""
    ux = jax.vmap(_ux, in_axes=(None, 0, 0))(model, pts["x_w"], pts["r_w"])
    ur = jax.vmap(_ur, in_axes=(None, 0, 0))(model, pts["x_w"], pts["r_w"])
    return jnp.mean(ux ** 2 + ur ** 2)


def _inlet_loss(model, pts, Re):
    """Parabolic inlet profile (u_x given, u_r = 0)."""
    U_mean = mean_throat_velocity(Re)
    ux_target = jax.vmap(
        lambda rv: hagen_poiseuille_inlet(rv, U_mean)
    )(pts["r_inl"])
    ux = jax.vmap(_ux, in_axes=(None, 0, 0))(
        model, pts["x_inl"], pts["r_inl"]
    )
    ur = jax.vmap(_ur, in_axes=(None, 0, 0))(
        model, pts["x_inl"], pts["r_inl"]
    )
    return jnp.mean((ux - ux_target) ** 2) + jnp.mean(ur ** 2)


def _outlet_loss(model, pts):
    """Zero-pressure outlet (p = 0 reference)."""
    p_out = jax.vmap(_p, in_axes=(None, 0, 0))(
        model, pts["x_out"], pts["r_out"]
    )
    return jnp.mean(p_out ** 2)


def _axis_loss(model, pts):
    """Axisymmetry on r = 0: u_r = 0 and d u_x / d r = 0."""
    ur = jax.vmap(_ur, in_axes=(None, 0, 0))(
        model, pts["x_ax"], pts["r_ax"]
    )
    dux_dr = jax.vmap(
        lambda xv, rv: jax.grad(_ux, argnums=2)(model, xv, rv)
    )(pts["x_ax"], pts["r_ax"])
    return jnp.mean(ur ** 2 + dux_dr ** 2)


def _data_loss(model, obs):
    """L2 mismatch between PINN and PIV samples, NaN/mask aware.

    Each row's squared residual is multiplied by the corresponding
    mask (1.0 = real measurement, 0.0 = blank cell). The total is
    normalised by the number of real measurements so the loss
    magnitude is comparable across CSVs with different masking ratios.
    """
    ux = jax.vmap(_ux, in_axes=(None, 0, 0))(model, obs["x_d"], obs["r_d"])
    ur = jax.vmap(_ur, in_axes=(None, 0, 0))(model, obs["x_d"], obs["r_d"])
    sq_x = obs["mux"] * (ux - obs["ux_d"]) ** 2
    sq_r = obs["mur"] * (ur - obs["ur_d"]) ** 2
    n_eff = jnp.sum(obs["mux"]) + jnp.sum(obs["mur"]) + 1e-12
    return (jnp.sum(sq_x) + jnp.sum(sq_r)) / n_eff


COMPONENT_NAMES = (
    "continuity", "mom_x", "mom_r",
    "wall", "inlet", "outlet", "axis",
    "data",
)


def _component_losses(model, pts, obs, Re):
    """Return a dict {name: unweighted loss} for each loss component."""
    L_c, L_mx, L_mr = _residual_losses(model, pts)
    return {
        "continuity": L_c,
        "mom_x":      L_mx,
        "mom_r":      L_mr,
        "wall":       _wall_loss(model, pts),
        "inlet":      _inlet_loss(model, pts, Re),
        "outlet":     _outlet_loss(model, pts),
        "axis":       _axis_loss(model, pts),
        "data":       _data_loss(model, obs),
    }


def total_loss(model, pts, obs, Re, weights):
    """Weighted sum of residual + BC + data losses.

    weights is a dict of (name -> scalar). Values can be Python floats
    or jnp scalars (the latter lets adaptive weighting vary them at
    runtime without JIT recompilation).
    """
    L = _component_losses(model, pts, obs, Re)
    return sum(weights[n] * L[n] for n in COMPONENT_NAMES)


def compute_grad_norms(model, pts, obs, Re):
    """Per-component ||grad_theta L_i||_2 (Wang 2021).

    Splits the model into (graphdef, params) once and computes the
    gradient of each scalar component loss against params. Returns a
    dict {name: float}. One backward pass per component (8 total).
    """
    gdef, state = nnx.split(model)

    def make_loss_fn(name):
        def loss_fn(params):
            m = nnx.merge(gdef, params)
            return _component_losses(m, pts, obs, Re)[name]
        return loss_fn

    norms = {}
    for name in COMPONENT_NAMES:
        grads = jax.grad(make_loss_fn(name))(state)
        leaves = jax.tree_util.tree_leaves(grads)
        sq_sum = sum(jnp.sum(jnp.asarray(g) ** 2) for g in leaves)
        norms[name] = float(jnp.sqrt(sq_sum))
    return norms


def update_weights_ntk(weights, grad_norms, alpha=0.9, eps=1e-12):
    """Wang 2021 NTK rebalancing.

    Target weight for term i: λ_i = max_j(||∇L_j||) / ||∇L_i||
    Exponential moving average with alpha (0.9 = gentle, 0.0 = snap).
    """
    max_norm = max(grad_norms.values())
    new_weights = {}
    for name in weights:
        target = max_norm / (grad_norms[name] + eps)
        new_weights[name] = alpha * weights[name] + (1.0 - alpha) * target
    return new_weights


# =============================================================
# 9. Training (Adam → L-BFGS, mirroring B3)
# =============================================================
def train(model, pts, obs, Re, weights, adam_iters, lbfgs_iters, lr,
          print_every=1000, adaptive=False, rebalance_every=200,
          alpha=0.9):
    """Adam (optionally adaptive) -> L-BFGS.

    When adaptive=True, every `rebalance_every` iters we recompute
    per-component gradient norms and update weights per Wang 2021.
    Weights are converted to jnp scalars so JIT does not recompile
    on each update.
    """
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    # Convert weights to JAX scalars so the jitted step can read new
    # values at runtime without retracing the function.
    weights_j = {k: jnp.asarray(float(v)) for k, v in weights.items()}

    @nnx.jit
    def adam_step(model, optimizer, pts, obs, weights_j):
        loss_val, grads = nnx.value_and_grad(
            lambda m: total_loss(m, pts, obs, Re, weights_j)
        )(model)
        optimizer.update(model, grads)
        return loss_val

    t0 = time.time()
    for it in range(adam_iters):
        loss_val = adam_step(model, optimizer, pts, obs, weights_j)
        if it % print_every == 0:
            print(f"  Adam iter {it:>5d}  loss = {float(loss_val):.4e}",
                  flush=True)
        if adaptive and it > 0 and it % rebalance_every == 0:
            grad_norms = compute_grad_norms(model, pts, obs, Re)
            new_weights = update_weights_ntk(
                {k: float(v) for k, v in weights_j.items()},
                grad_norms, alpha=alpha,
            )
            weights_j = {k: jnp.asarray(v) for k, v in new_weights.items()}
            wstr = "  ".join(f"{k}={v:.2e}" for k, v in new_weights.items())
            print(f"    [rebalance @ it={it}]  weights -> {wstr}",
                  flush=True)
    print(f"  Adam phase: {time.time() - t0:.1f}s")

    # L-BFGS handoff via nnx.split / merge (B3 pattern). Freeze the
    # current weights for L-BFGS — adaptive rebalancing inside L-BFGS
    # would confuse its line search.
    frozen_weights = {k: float(v) for k, v in weights_j.items()}
    gdef, state = nnx.split(model)

    def lbfgs_loss(params):
        m = nnx.merge(gdef, params)
        return total_loss(m, pts, obs, Re, frozen_weights)

    t0 = time.time()
    solver = LBFGS(fun=lbfgs_loss, maxiter=lbfgs_iters, tol=1e-9)
    result = solver.run(state)
    model = nnx.merge(gdef, result.params)
    print(f"  L-BFGS phase: {time.time() - t0:.1f}s")
    print(f"  Final adaptive weights: {frozen_weights}")
    return model


# =============================================================
# 10. Evaluation: full-field prediction and centreline profile
# =============================================================
def evaluate(model, Re):
    """Predict (u_x, u_r, p) in PHYSICAL units on a fine grid.

    Uses uvp_at() so output scaling and hard-BC transforms apply.
    Previously called model() directly, which returned the raw MLP
    outputs (no U_SCALE, no phi wall factor) — that bug caused the
    plotted fields to look inverted and unphysical.
    """
    nx, nr = 401, 81
    xs = np.linspace(X_INLET_START, X_OUTLET_END, nx)
    rs = np.linspace(0.0, R_INLET, nr)

    @jax.jit
    def predict_grid(xs_j, rs_j):
        X, R = jnp.meshgrid(xs_j, rs_j, indexing="xy")
        # Use uvp_at per point so U_SCALE / hard-BC are applied
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

    return {
        "xs": xs, "rs": rs,
        "ux": ux, "ur": ur, "p": p,
        "U_mean": mean_throat_velocity(Re),
    }


def wall_shear_stress(model, n=400):
    """tau_w(x) = mu * |du_x/dn| evaluated on the wall.

    Approximated as mu * |du_x/dr| since the wall is nearly horizontal
    in the throat and outlet; conical-contraction WSS uses the normal
    derivative formula (left for a follow-up refinement).
    """
    xs = np.linspace(X_INLET_START, X_OUTLET_END, n)
    rs = np.asarray(wall_radius(jnp.asarray(xs)))

    @jax.jit
    def dux_dr_wall(xv, rv):
        return jax.grad(_ux, argnums=2)(model, xv, rv)

    dux = np.asarray(jax.vmap(dux_dr_wall)(jnp.asarray(xs), jnp.asarray(rs)))
    tau_w = MU * np.abs(dux)
    return xs, tau_w


# =============================================================
# 11. Plots
# =============================================================
def plot_fields(eval_dict, savepath):
    xs, rs = eval_dict["xs"], eval_dict["rs"]
    extent = [xs[0] * 1000, xs[-1] * 1000, 0, rs[-1] * 1000]  # mm

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fields = [
        (eval_dict["ux"], r"$u_x$ (m/s)",          "viridis"),
        (eval_dict["ur"], r"$u_r$ (m/s)",          "RdBu_r"),
        (eval_dict["p"],  r"$p$ (Pa, relative)",    "inferno"),
    ]
    for ax, (field, label, cmap) in zip(axes, fields):
        im = ax.imshow(field, origin="lower", extent=extent,
                       aspect="auto", cmap=cmap)
        ax.set_ylabel("r (mm)")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    axes[-1].set_xlabel("x (mm)")
    fig.suptitle("PINN-reconstructed nozzle fields")
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_centreline(eval_dict, savepath):
    xs = eval_dict["xs"]
    ux_cl = eval_dict["ux"][0, :]   # r = 0 row
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xs * 1000, ux_cl, "k-")
    ax.axvline(X_CONTRACT_START * 1000, color="grey", ls=":", lw=0.8)
    ax.axvline(X_THROAT_START   * 1000, color="grey", ls=":", lw=0.8)
    ax.axvline(X_THROAT_END     * 1000, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel(r"$u_x(x, r{=}0)$ (m/s)")
    ax.set_title("Centreline axial velocity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_wss(xs, tau_w, savepath):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xs * 1000, tau_w, "r-")
    ax.axhline(15.0, color="grey", ls=":", lw=0.8,
               label=r"WSS = 15 Pa (~150 dyn/cm$^2$, Malek 1999)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel(r"$\tau_w$ (Pa)")
    ax.set_title("Wall shear stress along the nozzle")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================
# 12. Main
# =============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=500.0,
                        help="Throat Reynolds number (500, 2000, 3500)")
    parser.add_argument("--piv-csv", type=str, default=None,
                        help="Path to nciphub PIV CSV; if omitted, "
                             "synthetic-PIV is used as fallback")
    parser.add_argument("--synthetic-piv", action="store_true",
                        help="Force synthetic PIV even if a CSV is provided")
    parser.add_argument("--n-obs", type=int, default=850,
                        help="Number of PIV samples (paper default ~850)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test (reduced iters/points)")
    parser.add_argument("--n-interior", type=int, default=None,
                        help="Override interior collocation count")
    parser.add_argument("--adam-iters", type=int, default=None,
                        help="Override Adam iteration count")
    parser.add_argument("--lbfgs-iters", type=int, default=None,
                        help="Override L-BFGS iteration count")
    parser.add_argument("--print-every", type=int, default=1000,
                        help="Adam iters between log lines (default 1000)")
    parser.add_argument("--width", type=int, default=128,
                        help="MLP hidden width (default 128)")
    parser.add_argument("--depth", type=int, default=8,
                        help="MLP hidden depth (default 8)")
    parser.add_argument("--n-fourier", type=int, default=N_FOURIER,
                        help="Fourier-feature count; 0 disables them "
                             f"(default {N_FOURIER})")
    parser.add_argument("--fourier-scale", type=float, default=FOURIER_SCALE,
                        help=f"Fourier frequency std (default {FOURIER_SCALE})")
    parser.add_argument("--adaptive-weights", action="store_true",
                        help="Enable Wang 2021 NTK adaptive loss weighting "
                             "(rebalances every --rebalance-every iters)")
    parser.add_argument("--rebalance-every", type=int, default=200,
                        help="Iters between adaptive weight rebalances")
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="EMA smoothing for adaptive weights "
                             "(1.0 = no change, 0.0 = snap; default 0.9)")
    parser.add_argument("--no-hard-bc", action="store_true",
                        help="Disable hard wall/axis BC enforcement "
                             "(falls back to pure soft losses)")
    args = parser.parse_args()
    global USE_HARD_BC
    USE_HARD_BC = not args.no_hard_bc

    # Hyperparameters
    n_interior = 10_000
    n_wall     = 500
    n_inlet    = 200
    n_outlet   = 200
    n_axis     = 200
    adam_iters = 20_000
    lbfgs_iters = 5_000
    lr = 1e-3

    if args.quick:
        n_interior  = 1_000
        n_wall      = 80
        n_inlet     = 60
        n_outlet    = 60
        n_axis      = 60
        adam_iters  = 1_000
        lbfgs_iters = 200
        args.n_obs  = min(args.n_obs, 100)
        print("[QUICK MODE] Reduced hyperparameters for fast sanity check.")

    # CLI overrides win over both defaults and --quick
    if args.n_interior  is not None: n_interior  = args.n_interior
    if args.adam_iters  is not None: adam_iters  = args.adam_iters
    if args.lbfgs_iters is not None: lbfgs_iters = args.lbfgs_iters

    # Loss weights — with hard BC enforcement (default), wall and axis
    # are mathematically zero by construction, so their weights are
    # effectively unused. The remaining competition is PDE residual vs.
    # inlet/outlet vs. data; these starting weights are coarse but
    # workable. Without hard BC, the user should override with the
    # heavier weights from prior runs.
    if USE_HARD_BC:
        weights = {
            "continuity":  1.0,
            "mom_x":       1.0,
            "mom_r":       1.0,
            "wall":        0.0,   # hard-enforced
            "inlet":      10.0,
            "outlet":      0.1,
            "axis":        0.0,   # hard-enforced
            "data":      100.0,
        }
    else:
        weights = {
            "continuity":  1.0,
            "mom_x":       1.0,
            "mom_r":       1.0,
            "wall":       10.0,
            "inlet":      10.0,
            "outlet":      0.1,
            "axis":        5.0,
            "data":     1000.0,
        }

    # ------ Set output scales from the Reynolds number ----------
    global U_SCALE, P_SCALE
    U_SCALE = mean_throat_velocity(args.re)          # m/s
    P_SCALE = RHO * U_SCALE ** 2                     # Pa  (rho U^2)
    print(f"  Output scaling: U_SCALE = {U_SCALE:.4f} m/s, "
          f"P_SCALE = {P_SCALE:.2f} Pa")

    rng = np.random.default_rng(SEED)
    pts = sample_points(n_interior, n_wall, n_inlet, n_outlet, n_axis, rng)

    # PIV data: real if given, synthetic otherwise
    if args.piv_csv and not args.synthetic_piv:
        print(f"Loading real PIV from {args.piv_csv} ...")
        obs = load_piv_csv(args.piv_csv, re_filter=args.re)
        piv_source = f"real ({args.piv_csv})"
    else:
        print(f"Using synthetic PIV (n_obs = {args.n_obs}, Re = {args.re}) ...")
        obs = synthetic_piv(args.n_obs, args.re, rng)
        piv_source = "synthetic-fallback"

    # Build the model with the requested architecture
    layers_cli = [2] + [args.width] * args.depth + [3]
    rngs = nnx.Rngs(SEED)
    model = MLP(layers_cli, rngs=rngs,
                n_fourier=args.n_fourier,
                fourier_scale=args.fourier_scale)
    n_params = sum(p.value.size for p in jax.tree_util.tree_leaves(
        nnx.state(model)) if hasattr(p, 'value'))
    print(f"  Architecture   = {args.depth} layers x {args.width} units, "
          f"Fourier features = {args.n_fourier} (scale {args.fourier_scale})")

    print(f"\n{'=' * 60}")
    print(f"  FDA Nozzle PINN — Re = {args.re}")
    print(f"  PIV source     = {piv_source}")
    print(f"  N_PIV          = {obs['x_d'].shape[0]}")
    print(f"  N_interior     = {n_interior}")
    print(f"  Loss weights   = {weights}")
    print(f"{'=' * 60}")
    model = train(model, pts, obs, args.re, weights,
                  adam_iters, lbfgs_iters, lr,
                  print_every=args.print_every,
                  adaptive=args.adaptive_weights,
                  rebalance_every=args.rebalance_every,
                  alpha=args.alpha)

    # Per-component loss breakdown at the final weights — tells us which
    # terms are actually small vs which are stuck (Wang 2021 diagnosis).
    L_c, L_mx, L_mr = _residual_losses(model, pts)
    L_w = _wall_loss(model, pts)
    L_i = _inlet_loss(model, pts, args.re)
    L_o = _outlet_loss(model, pts)
    L_a = _axis_loss(model, pts)
    L_d = _data_loss(model, obs)
    print("\n  Per-component loss (unweighted):")
    print(f"    continuity  = {float(L_c):.4e}")
    print(f"    mom_x       = {float(L_mx):.4e}")
    print(f"    mom_r       = {float(L_mr):.4e}")
    print(f"    wall        = {float(L_w):.4e}")
    print(f"    inlet       = {float(L_i):.4e}")
    print(f"    outlet      = {float(L_o):.4e}")
    print(f"    axis        = {float(L_a):.4e}")
    print(f"    data        = {float(L_d):.4e}")

    # Evaluate and plot
    ev = evaluate(model, args.re)
    plot_fields(ev,      FIG_DIR / f"fields_Re{int(args.re)}.png")
    plot_centreline(ev,  FIG_DIR / f"centreline_Re{int(args.re)}.png")
    xs_w, tau_w = wall_shear_stress(model)
    plot_wss(xs_w, tau_w, FIG_DIR / f"wss_Re{int(args.re)}.png")

    # Diagnostics on the PIV reconstruction (mask-aware: only count rows
    # where u_x was actually measured).
    ux_pred = np.asarray(jax.vmap(_ux, in_axes=(None, 0, 0))(
        model, obs["x_d"], obs["r_d"]))
    ur_pred = np.asarray(jax.vmap(_ur, in_axes=(None, 0, 0))(
        model, obs["x_d"], obs["r_d"]))
    mux = np.asarray(obs["mux"]).astype(bool)
    mur = np.asarray(obs["mur"]).astype(bool)
    ux_true = np.asarray(obs["ux_d"])
    ur_true = np.asarray(obs["ur_d"])
    rel_ux = 100.0 * np.linalg.norm(ux_pred[mux] - ux_true[mux]) \
             / max(np.linalg.norm(ux_true[mux]), 1e-12)
    rel_ur = 100.0 * np.linalg.norm(ur_pred[mur] - ur_true[mur]) \
             / max(np.linalg.norm(ur_true[mur]), 1e-12)
    rel_err = rel_ux   # primary metric (axial is dominant in nozzle flow)

    metrics = {
        "paper":         "B4 FDA Nozzle PINN",
        "implementation": "JAX/Flax NNX + Optax + jaxopt",
        "reynolds":      args.re,
        "piv_source":    piv_source,
        "n_obs":         int(obs["x_d"].shape[0]),
        "U_mean_throat": float(ev["U_mean"]),
        "tau_w_peak_Pa": float(np.nanmax(tau_w)),
        "data_rel_L2_pct":     float(rel_err),
        "data_rel_L2_ux_pct":  float(rel_ux),
        "data_rel_L2_ur_pct":  float(rel_ur),
        "n_ux_measurements":   int(mux.sum()),
        "n_ur_measurements":   int(mur.sum()),
        "hyperparameters": {
            "layers":       LAYERS,
            "adam_iters":   adam_iters,
            "lbfgs_iters":  lbfgs_iters,
            "n_interior":   n_interior,
            "weights":      weights,
        },
    }
    with open(FIG_DIR / f"metrics_Re{int(args.re)}.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n  Data L2 (PINN vs. PIV samples) = {rel_err:.2f}%")
    print(f"  Peak WSS                       = {np.nanmax(tau_w):.2f} Pa")
    print(f"  Throat mean velocity           = {ev['U_mean']:.3f} m/s")
    print(f"  Outputs in {FIG_DIR}/")


if __name__ == "__main__":
    main()
