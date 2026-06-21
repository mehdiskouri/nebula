"""Scratch calibrator for V2.5 — measures the margins behind the FROZEN notebook thresholds.

Mirrors the V2.5 notebook metrics (autodiff / convergence / verified fidelity / incoherent-flagging /
physiology inverse / physiology impossibility) WITHOUT markdown/figures, so the thresholds in
`_build_v25_nb.py` are set *below* what is measured here (repo practice, protocol §1). Uncommitted/aux.
Run: .venv/bin/python src/verification/oracles/_calib_v25.py
"""
import time
import numpy as np
import torch
from scipy.stats import spearmanr

import violent_cells as vc
import regulator as reg
import inverse_design as idz
from surrogate_gnn import Ensemble, TrainCfg, build_dataset, set_determinism, DEVICE, DATA_PARAMS

t0 = time.time()
set_determinism(0)
CACHE = "/tmp/v25_calib_cache"

# ---- train the surrogate once (production config; beta=0 since trust uses FeatureDensity distance) ----
rng = np.random.default_rng(2025)
train = vc.family_battery(idz.N, rng, 45)
y_tr = build_dataset(train, DATA_PARAMS, cache="/tmp/v25_calib_train.npz")["y"]
ens = Ensemble.train(train, y_tr, TrainCfg(epochs=400, beta=0.0), M=5, base_seed=0)
print(f"[{time.time()-t0:.0f}s] surrogate trained (45 cells, M=5, 400ep)")

# ---- (M1) autodiff vs finite difference over several theta ----
errs = []
for d0, c0 in [(0.4, 30.0), (0.55, 45.0), (0.7, 65.0)]:
    depth = torch.tensor(d0, dtype=torch.float64, device=DEVICE, requires_grad=True)
    contrast = torch.tensor(c0, dtype=torch.float64, device=DEVICE, requires_grad=True)
    s = idz.differentiable_strength(ens, depth, contrast); s.backward()
    g_ad = np.array([float(depth.grad), float(contrast.grad)])
    h = 1e-4
    sv = lambda d, c: float(idz.differentiable_strength(
        ens, torch.tensor(d, dtype=torch.float64, device=DEVICE),
        torch.tensor(c, dtype=torch.float64, device=DEVICE)).detach())
    g_fd = np.array([(sv(d0 + h, c0) - sv(d0 - h, c0)) / (2 * h),
                     (sv(d0, c0 + h) - sv(d0, c0 - h)) / (2 * h)])
    errs.append(np.linalg.norm(g_ad - g_fd) / (np.linalg.norm(g_fd) + 1e-12))
M1 = float(max(errs))
print(f"[{time.time()-t0:.0f}s] (M1) autodiff-vs-FD max rel err = {M1:.2e}")

# ---- achievable ranges: family (in-distribution) and the WIDER design box (reachable) ----
flo, fhi = idz.achievable_range(ens, *idz.FAMILY_BOX)
wlo, whi = idz.achievable_range(ens, *idz.DEPTH_BOX, *idz.CONTRAST_BOX)
print(f"     family strength range [{flo:.3f}, {fhi:.3f}]; wide-box reachable [{wlo:.3f}, {whi:.3f}]")

# coherent (interior, should verify) + extrapolation band (reachable only OOD, gate test)
coherent = list(np.linspace(flo + 0.05 * (fhi - flo), fhi - 0.05 * (fhi - flo), 8))
extrap = list(np.linspace(fhi + 0.15 * (whi - fhi), whi - 0.05 * (whi - wlo), 4))
conv, real_rel, trust = [], [], []
verified_rel = []
for tg in coherent + extrap:
    c = idz.inverse_design(float(tg), ens, cache_dir=CACHE, iters=300)
    conv.append(abs(c.surrogate_pred - tg) <= 0.02)
    if c.verified_real is not None:
        real_rel.append(abs(c.verified_real - tg) / abs(tg)); trust.append(c.trust)
        if c.status == "verified":
            verified_rel.append(abs(c.verified_real - tg) / abs(tg))
M2 = float(np.mean(conv))
M3 = float(np.median(verified_rel)) if verified_rel else float("nan")
RHO = float(spearmanr(trust, real_rel).correlation) if len(real_rel) > 2 else float("nan")
print(f"[{time.time()-t0:.0f}s] (M2) convergence frac = {M2:.2f}; (M3) verified median real err = {M3:.3f} "
      f"({len(verified_rel)} verified of {len(coherent)} coherent)")
print(f"     (M4) Spearman(trust, real err) = {RHO:+.3f} over {len(real_rel)} reachable candidates")

# ---- (M4) truly-incoherent targets (beyond the wide-box reach) -> impossible, 0 fabricated ----
inc = [whi + 0.3 * (whi - wlo), whi + 0.6 * (whi - wlo), wlo - 0.3 * (whi - wlo)]
inc_status = [idz.inverse_design(float(t), ens, cache_dir=CACHE, iters=300).status for t in inc]
M4_inc = all(s == "impossible" for s in inc_status)
print(f"     (M4) incoherent statuses = {inc_status} -> all impossible: {M4_inc}")

# ---- (M5/M6) physiology: homeostatic inverse + impossibility ----
base = reg.RegulatorParams()
rc = reg.r_critical(base)
P_base = idz.homeostatic_pressure(base, base.r0)
P_targets = [P_base * f for f in (0.9, 1.0, 1.1)]
phys = [idz.inverse_homeostasis(P, base, base.r0) for P in P_targets]
M5 = all(c.status == "verified" and abs(c.surrogate_pred - P) / P < 0.05
         for c, P in zip(phys, P_targets))
phys_err = max(abs(c.surrogate_pred - P) / P for c, P in zip(phys, P_targets))
dead = idz.inverse_homeostasis(P_base, base, 0.5 * rc)
M6 = dead.status == "impossible"
print(f"[{time.time()-t0:.0f}s] (M5) homeostatic inverse verified={M5} (max rel err {phys_err:.4f}; "
      f"margins {[round(c.trust,2) for c in phys]}, basin_area {[round(c.verified_real,2) for c in phys]})")
print(f"     (M6) world r={0.5*rc:.3f} < r_crit={rc:.3f} -> {dead.status} (impossible: {M6})")

print(f"\n[{time.time()-t0:.0f}s total]  --- suggested FROZEN thresholds (set below these) ---")
print(f"  GRAD_TOL     = 1e-3   (measured {M1:.1e})")
print(f"  CONV_MIN     = 0.90   (measured {M2:.2f})")
print(f"  REAL_TOL     = 0.12   (measured median {M3:.3f})")
print(f"  TRUST_RHO_MIN= 0.60   (measured {RHO:+.3f})")
print(f"  PHYS_TOL     = 0.05   (measured {phys_err:.4f})")
