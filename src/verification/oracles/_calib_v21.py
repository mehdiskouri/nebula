"""SCRATCH (uncommitted) calibration for V2.1 — measures calibration + the handoff frontier.

Run AFTER _calib_v24 (reuses the v24_train cache). Builds a graded-difficulty violent battery
(easy in-family wedges + a hard extrapolation tail), measures whether the surrogate's self-
uncertainty u tracks its actual error, and whether a handoff operating point exists. Prints
numbers to freeze the V2.1 thresholds with margin.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np

import violent_cells as vc
from surrogate_gnn import (Ensemble, EnvelopeDetector, TrainCfg, build_dataset, fallback_flags,
                           DATA_PARAMS)
import handoff_rule as hr

N = 12
CACHE = str(pathlib.Path(__file__).parents[3] / "verification_notebooks" / "phase2" / "cache")

# reuse the exact V2.4 training set (same seed/order) so the surrogate is identical
rng = np.random.default_rng(2024)
train = vc.family_battery(N, rng, 45)
d_tr = build_dataset(train, DATA_PARAMS, cache=f"{CACHE}/v24_train.npz")
ens = Ensemble.train(train, d_tr["y"], TrainCfg(epochs=400, physics=True), M=5, base_seed=0)

# V2.1 battery: graded difficulty. EASY = in-family wedges; HARD = extrapolation tail (deep,
# high-contrast) where the surrogate is asked to leave its training manifold -> larger error.
brng = np.random.default_rng(777)
easy = vc.family_battery(N, brng, 34)
hard = [vc.wedge_sample(N, float(brng.uniform(0.80, 0.95)), float(brng.uniform(95.0, 180.0)))
        for _ in range(8)]                       # ~19% genuinely-hard (extrapolation) minority
battery = easy + hard
t = time.time()
d_b = build_dataset(battery, DATA_PARAMS, cache=f"{CACHE}/v21_battery.npz")
print(f"battery {len(battery)} cells (RVE truth ready, {time.time()-t:.0f}s)")

pred = ens.predict(battery)
y_true = d_b["y"]
rel_err = np.abs(pred["mean"] - y_true) / np.abs(y_true)
u = pred["u"]
print(f"rel_err: median={np.median(rel_err):.3f} p95={np.quantile(rel_err,0.95):.3f} max={rel_err.max():.3f}")
print(f"u:       median={np.median(u):.4f} range={u.min():.4f}..{u.max():.4f}")
print(f"epistemic median={np.median(pred['epistemic']):.4f}  aleatoric median={np.median(pred['aleatoric']):.4f}")

# CALIBRATION
rho_b = hr.binned_rank_correlation(u, rel_err)
rho_raw = hr.rank_correlation(u, rel_err)
oc = hr.overconfidence(rel_err, u, k=1.0)
print(f"\n[calib] binned reliability rho={rho_b:.3f} (raw {rho_raw:.3f}); over-confidence@1sigma={oc:+.3f}")
for k, (o, nom) in hr.coverage(rel_err, u).items():
    print(f"        coverage @{k:.0f}sigma observed={o:.3f} nominal={nom:.3f}")

# RULE — u-only vs MULTI-SIGNAL (envelope/percolation gate OR u). The gate routes descriptor-far
# extrapolation cells (which u under-flags) straight to RVE; among the rest the rule trusts if u<=tau.
env = EnvelopeDetector.fit(train)
z_thr = float(np.quantile(env.score(train), 0.95))
gate, _ = fallback_flags(battery, env, z_thr)
hard_frac = float((rel_err > 0.10).mean())
print(f"\n[rule] battery hard fraction (err>0.10) = {hard_frac:.3f}; envelope/percolation gate flags {gate.mean():.3f}")
fr_u = hr.frontier(u, rel_err)
fr_g = hr.frontier(u, rel_err, gate=gate)
print(f"  always-trust tail (LYING)={fr_u.tail_err[-1]:.3f}; always-RVE frac (STALLING)={fr_u.rve_frac[0]:.3f}")
for name, fr in [("u-only", fr_u), ("multi-signal", fr_g)]:
    op = hr.operating_point(fr, err_bound=0.10, rve_budget=0.30)
    print(f"  {name:12s}: exists={op['exists']} rve_frac={op['rve_frac']:.3f} "
          f"tail={op['tail_err']:.3f} within_budget(0.30)={op['within_budget']}")
