"""
ablation_pump_C5.py
===================
Ablation study isolating the role of each loss-component in the
pump-diffuser PINN. Re-trains pump Condition 5 (6.0 L/min, 3500 RPM)
under four loss configurations:

  (a) data-only   : only |V| matching, no PDE residuals
  (b) PDE-only    : only Navier-Stokes residuals, no PIV anchoring
  (c) data + continuity (no momentum) : data + divergence-free, no NS
  (d) full        : data + continuity + momentum (canonical baseline)

Headline empirical claim under test:
  The direction recovery from magnitude-only |V| requires the
  **momentum** residuals, not just continuity.

Outputs:
  ablation_results/metrics_ablation.json
  ablation_results/ablation_fields_<cfg>.png   (one per config)
  ablation_results/ablation_streamlines.png    (4-panel comparison)
  ablation_results/ablation_summary.png        (rel-L2 + Eg bar chart)

Usage:
  python ablation_pump_C5.py                # full run (~45 min CPU)
  python ablation_pump_C5.py --quick        # smoke test (~3 min CPU)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jaxopt import LBFGS

jax.config.update("jax_enable_x64", True)

# Reuse the canonical pump-diffuser implementation
from fda_pump_diffuser_PINN import (
    MLP, RHO, NU, OP_TABLE,
    load_piv, make_scales,
    _u, _v, uvp, pde_residuals,
    sample_interior, stewart_eg_magnitude,
)

CASE = "C5"
ROOT = Path(__file__).parent
OUT  = ROOT / "ablation_results"
OUT.mkdir(exist_ok=True)


# ── 4 ablation configurations ────────────────────────────────────
CONFIGS = {
    "a_data_only":     dict(data=200.0, mom_x=0.0, mom_y=0.0, cont=0.0),
    "b_pde_only":      dict(data=0.0,   mom_x=1.0, mom_y=1.0, cont=1.0),
    "c_data_cont":     dict(data=200.0, mom_x=0.0, mom_y=0.0, cont=1.0),
    "d_full":          dict(data=200.0, mom_x=1.0, mom_y=1.0, cont=1.0),
}

CONFIG_LABELS = {
    "a_data_only":  "(a) data only (|V| match, no physics)",
    "b_pde_only":   "(b) PDE only (no PIV anchor)",
    "c_data_cont":  "(c) data + continuity (no momentum)",
    "d_full":       "(d) full (canonical baseline)",
}


# ── Loss with adjustable weights ─────────────────────────────────
def total_loss_weighted(model, batch, S, weights):
    xd, yd, vd = batch["x_d"], batch["y_d"], batch["v_d"]
    u_d = jax.vmap(_u, in_axes=(None, 0, 0, None))(model, xd, yd, S)
    v_d = jax.vmap(_v, in_axes=(None, 0, 0, None))(model, xd, yd, S)
    mag = jnp.sqrt(u_d ** 2 + v_d ** 2 + 1e-12)
    err = jnp.mean((mag - vd) ** 2)
    norm = jnp.mean(vd ** 2) + 1e-12
    L_data = err / norm

    xi, yi = batch["x_i"], batch["y_i"]
    rx, ry, rc = jax.vmap(pde_residuals, in_axes=(None, 0, 0, None))(
        model, xi, yi, S
    )
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


# ── Train one configuration ──────────────────────────────────────
def train_one_config(cfg_name, weights, xs, ys, vs, S, args):
    print(f"\n  --- {CONFIG_LABELS[cfg_name]} ---")
    print(f"      weights: {weights}")
    rngs = nnx.Rngs(0)
    model = MLP(din=2, dout=3, width=64, depth=6, rngs=rngs)
    schedule = optax.cosine_decay_schedule(1e-3, args.adam_iters, alpha=0.5)
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    xd = jnp.asarray(xs); yd = jnp.asarray(ys); vd = jnp.asarray(vs)

    @nnx.jit
    def train_step(model, optimizer, key):
        x_i, y_i = sample_interior(key, args.n_interior, S)
        batch = dict(x_d=xd, y_d=yd, v_d=vd, x_i=x_i, y_i=y_i)
        def fn(m):
            tot, parts = total_loss_weighted(m, batch, S, weights)
            return tot, parts
        (tot, parts), grads = nnx.value_and_grad(fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return tot, parts

    t0 = time.time()
    rng = jax.random.PRNGKey(42)
    log_every = max(1, args.adam_iters // 6)
    for step in range(args.adam_iters):
        rng, subkey = jax.random.split(rng)
        tot, parts = train_step(model, optimizer, subkey)
        if step % log_every == 0 or step == args.adam_iters - 1:
            print(f"        Adam {step:6d}  tot={float(tot):.3e}  "
                  f"data={float(parts['data']):.2e}  "
                  f"mom={float(parts['mom_x']) + float(parts['mom_y']):.2e}  "
                  f"cont={float(parts['cont']):.2e}")
    t_adam = time.time() - t0

    # L-BFGS finisher (skip for ablation if PDE-only — would diverge)
    if args.lbfgs_iters > 0 and cfg_name != "b_pde_only":
        x_i, y_i = sample_interior(jax.random.PRNGKey(7), args.n_interior, S)
        batch_lbfgs = dict(x_d=xd, y_d=yd, v_d=vd, x_i=x_i, y_i=y_i)
        graphdef, state = nnx.split(model)
        def flat_loss(sf):
            m = nnx.merge(graphdef, sf)
            return total_loss_weighted(m, batch_lbfgs, S, weights)[0]
        t0 = time.time()
        solver = LBFGS(fun=flat_loss, maxiter=args.lbfgs_iters, tol=1e-9)
        result = solver.run(state)
        state = result.params
        model = nnx.merge(graphdef, state)
        t_lbfgs = time.time() - t0
    else:
        t_lbfgs = 0.0

    # Evaluate at PIV sample points
    u_pred = np.asarray(jax.vmap(_u, in_axes=(None, 0, 0, None))(model, xd, yd, S))
    v_pred = np.asarray(jax.vmap(_v, in_axes=(None, 0, 0, None))(model, xd, yd, S))
    mag_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
    rel_L2_mag = 100.0 * np.linalg.norm(mag_pred - vs) / max(np.linalg.norm(vs), 1e-12)
    Eg = stewart_eg_magnitude(model, xd, yd, vd, S)

    # Direction quality: how non-degenerate is the vector field?
    # Measure: mean magnitude of (u,v) -- a chaotic field will have a similar
    # mean magnitude regardless of direction, but a structured flow will show
    # clear spatial patterns. We report this as a sanity diagnostic.
    mean_u_abs = float(np.mean(np.abs(u_pred)))
    mean_v_abs = float(np.mean(np.abs(v_pred)))

    print(f"        rel L2 |V|       = {rel_L2_mag:6.2f}%")
    print(f"        Stewart E_g      = {Eg:.4f}")
    print(f"        |V|_peak  pred   = {mag_pred.max():.3f}   PIV: {vs.max():.3f}")
    print(f"        Adam {t_adam:.0f}s  LBFGS {t_lbfgs:.0f}s")

    # Save the trained model's predictions on the full PIV grid
    np.savez_compressed(
        OUT / f"ablation_predictions_{cfg_name}.npz",
        x=xs, y=ys, v_piv=vs,
        u_pred=u_pred, v_pred=v_pred, mag_pred=mag_pred,
    )

    return dict(
        config=cfg_name, label=CONFIG_LABELS[cfg_name],
        weights=weights,
        rel_L2_mag_pct=float(rel_L2_mag),
        stewart_Eg_mag=float(Eg),
        vmag_peak_predicted=float(mag_pred.max()),
        vmag_peak_pivdata=float(vs.max()),
        mean_u_abs=mean_u_abs, mean_v_abs=mean_v_abs,
        train_time_s=float(t_adam + t_lbfgs),
    )


# ── Comparison figures ───────────────────────────────────────────
def make_streamline_figure(metrics_dict):
    """4-panel grid of vector-field streamlines, one per config."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (cfg_name, label) in zip(axes, CONFIG_LABELS.items()):
        d = np.load(OUT / f"ablation_predictions_{cfg_name}.npz")
        x, y, u, v, mag = (d["x"] * 1000, d["y"] * 1000,
                           d["u_pred"], d["v_pred"], d["mag_pred"])
        tri = mtri.Triangulation(x, y)
        tc = ax.tripcolor(tri, mag, cmap="viridis", shading="gouraud",
                          vmin=0, vmax=8)
        # Subsample for quiver
        idx = np.arange(0, len(x), 20)
        ax.quiver(x[idx], y[idx], u[idx], v[idx],
                  color="white", scale=80, width=0.003, alpha=0.85)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x (mm)")
        if ax is axes[0]:
            ax.set_ylabel("y (mm)")

        m = metrics_dict[cfg_name]
        ax.text(0.02, 0.97,
                f"rel L²={m['rel_L2_mag_pct']:.1f}%\n$E_g$={m['stewart_Eg_mag']:.3f}",
                transform=ax.transAxes, fontsize=9, va="top",
                color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.55))

    fig.suptitle("Ablation: pump diffuser C5 vector-field recovery "
                 "under four loss configurations  (PIV |V|max = 7.43 m/s)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_streamlines.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'ablation_streamlines.png'}")


def make_summary_bar(metrics_dict):
    import matplotlib.pyplot as plt

    cfgs = list(CONFIG_LABELS)
    labels = [CONFIG_LABELS[c].split(" ", 1)[1] for c in cfgs]
    rel_L2 = [metrics_dict[c]["rel_L2_mag_pct"] for c in cfgs]
    Eg     = [metrics_dict[c]["stewart_Eg_mag"] for c in cfgs]
    colors = ["#d95f02", "#7570b3", "#1f78b4", "#1b9e77"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.5))

    axL.bar(range(len(cfgs)), rel_L2, color=colors)
    axL.set_xticks(range(len(cfgs)))
    axL.set_xticklabels([f"({c.split('_', 1)[0]})" for c in cfgs])
    axL.set_ylabel("rel $L^2$ |V| error (%)")
    axL.set_title("(a) Magnitude reconstruction error")
    axL.grid(alpha=0.3, axis="y")
    for i, v in enumerate(rel_L2):
        axL.text(i, v + max(rel_L2) * 0.02, f"{v:.1f}%",
                 ha="center", fontsize=9)

    axR.bar(range(len(cfgs)), Eg, color=colors)
    axR.set_xticks(range(len(cfgs)))
    axR.set_xticklabels([f"({c.split('_', 1)[0]})" for c in cfgs])
    axR.set_ylabel("Stewart $E_g$ (|V|)")
    axR.set_title("(b) Mean absolute relative error")
    axR.grid(alpha=0.3, axis="y")
    for i, v in enumerate(Eg):
        axR.text(i, v + max(Eg) * 0.02, f"{v:.3f}",
                 ha="center", fontsize=9)

    # Compact legend mapping (a)/(b)/(c)/(d) to descriptions
    legend_text = "\n".join(
        f"  {l}" for l in
        ["(a) data only         (no physics)",
         "(b) PDE only          (no PIV anchor)",
         "(c) data + continuity (no momentum)",
         "(d) full              (canonical baseline)"]
    )
    fig.text(0.5, -0.02, legend_text, ha="center", fontsize=9,
             family="monospace", color="#333")

    fig.suptitle("Ablation summary: contribution of each loss term to "
                 "pump C5 reconstruction", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {OUT / 'ablation_summary.png'}")


# ── Main ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 5k Adam + 500 L-BFGS, 1500 interior pts")
    args = p.parse_args()

    if args.quick:
        args.adam_iters, args.lbfgs_iters, args.n_interior = 5_000, 500, 1500
    else:
        args.adam_iters, args.lbfgs_iters, args.n_interior = 60_000, 5_000, 8_000

    print("=" * 70)
    print(f"  Ablation study -- pump diffuser Condition {CASE}")
    print(f"  Adam={args.adam_iters}, LBFGS={args.lbfgs_iters}, "
          f"N_int={args.n_interior}")
    print("=" * 70)

    xs, ys, vs = load_piv(CASE, ROOT)
    S = make_scales(xs, ys, vs)
    print(f"  PIV: {xs.size} points  |V|max={vs.max():.3f} m/s")
    print(f"  Scales: U={S['u_scale']:.3f} m/s, P={S['p_scale']:.1f} Pa")

    all_metrics = {}
    for cfg_name, weights in CONFIGS.items():
        all_metrics[cfg_name] = train_one_config(
            cfg_name, weights, xs, ys, vs, S, args
        )

    # ─── Persist + visualise ────────────────────────────────────
    with open(OUT / "metrics_ablation.json", "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\n  wrote {OUT / 'metrics_ablation.json'}")

    make_streamline_figure(all_metrics)
    make_summary_bar(all_metrics)

    # ─── Tabular summary ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY  --  pump C5  --  PIV |V|max = "
          f"{vs.max():.3f} m/s")
    print("=" * 70)
    print(f"  {'config':<22} {'rel L2 |V|':>12} {'E_g':>8} "
          f"{'|V|max':>8} {'time':>8}")
    print(f"  {'-'*22:<22} {'-'*12:>12} {'-'*8:>8} "
          f"{'-'*8:>8} {'-'*8:>8}")
    for c, m in all_metrics.items():
        print(f"  {c:<22} {m['rel_L2_mag_pct']:>10.2f}%  "
              f"{m['stewart_Eg_mag']:>8.4f} {m['vmag_peak_predicted']:>8.3f}"
              f" {m['train_time_s']:>7.0f}s")


if __name__ == "__main__":
    main()
