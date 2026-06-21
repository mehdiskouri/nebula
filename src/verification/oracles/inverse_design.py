"""
Inverse design + candidate verification — V2.5 (Decision #18; ARCHITECTURE §III.5-III.6).

The "set the result, gradient-descend to the parameters that produce it" non-render analysis, in the
two domains the protocol names:

  MATERIAL (depends on V2.4) — gradient descent on cell design params theta=(char depth, contrast)
  through a DIFFERENTIABLE forward (the V2.4 surrogate over an analytic featurizer) to hit a target
  normalized peak strength; the candidate is then RE-SIMULATED with the REAL damage-DNS oracle. The
  surrogate is a fast model that can be EXPLOITED at its weak points, so the result is a *verified
  candidate, not an oracle*: every candidate is re-simulated and gated by its calibrated uncertainty /
  distance-to-manifold (V2.4/V2.1). An unreachable target returns a precise IMPOSSIBILITY.

  PHYSIOLOGY (survival-spectrum) — invert a regulator's knobs (gain K / setpoint P_set) to a target
  resting homeostatic pressure in world X, verified by the independent brute-force basin oracle
  (`regulator.basin_map`) + the viability margin; a world whose reserve is below `r_critical` returns
  IMPOSSIBLE (no healthy fixed point) — "can this creature survive here" derived, not authored.

Differentiability:
  * The 2-phase char-wedge featurization is ANALYTIC in theta — the descriptor (Voigt/Reuss/gap/
    soft_frac/log10-contrast) is fraction-only, and the region-graph node features are smooth in depth
    via the wedge profile (a soft sigmoid indicator). So strength(theta) = surrogate(features(theta)) is
    torch-autodiff. The surrogate forward (`MemberNet.__call__`) is differentiable; only
    `Ensemble.predict` blocks grad with `@torch.no_grad()`, so we add a thin differentiable forward over
    the public `ens.members` + `ens.normalizer` — WITHOUT editing surrogate_gnn (V2.4/V2.1 unchanged).
  * The homeostatic pressure is the smooth root of `regulator._G`; we differentiate it w.r.t. P_set by
    the implicit function theorem (analytic Jacobian) — exact ∂outcome/∂param.

Reuses (unedited): surrogate_gnn (Ensemble/FeatureDensity/TrainCfg/featurize/set_determinism),
dns_damage_3d (run_path/DamageParams), violent_cells (family_battery/descriptor/region_graph/
outcome_target/build_dataset/DATA_PARAMS), cells (char_wedge_cell), regulator (healthy_fp/basin_map/
make_viability_margin/r_critical/pump/a_target/...). Pure numpy+torch (+ the GPU damage-DNS path).
"""
import os
from dataclasses import dataclass, replace

import numpy as np
import torch

import cells
import violent_cells as vc
import dns_damage_3d as dd
import regulator as reg
from surrogate_gnn import (Ensemble, FeatureDensity, TrainCfg, featurize, build_dataset,
                           outcome_target, set_determinism, DEVICE, DATA_PARAMS)

E_WOOD = vc.E_WOOD          # 10.0 — featurizer must match the family's wood modulus
NU = vc.NU                  # 0.3
N = 16                      # cell resolution (R=4 -> clean 4-voxel blocks; damage-DNS ~1-2 s/cell)
R = 4
# design box: modestly wider than the training family [0.2,0.85]x[20,80] so inverse design can push a
# little off-manifold — where the surrogate is exploitable and the trust gate must fire — without
# extrapolating to absurd (negative / >1) strengths.
DEPTH_BOX = (0.10, 0.90)
CONTRAST_BOX = (12.0, 100.0)
FAMILY_BOX = (0.20, 0.85, 20.0, 80.0)    # (depth_lo, depth_hi, contrast_lo, contrast_hi)


# ============================================================================================
# Differentiable analytic featurizer  theta=(depth, contrast) -> (X_nodes, edge_index, X_desc)
# ============================================================================================
def _iso_C(E, nu=NU):
    """6x6 isotropic stiffness as a torch tensor (matches homogenization.isotropic_stiffness)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = torch.zeros((6, 6), dtype=torch.float64, device=E.device)
    C[:3, :3] = lam
    for i in range(3):
        C[i, i] = lam + 2.0 * mu
    for i in range(3, 6):
        C[i, i] = mu
    return C


def _wedge_soft_field(depth, n=N, eps=0.012):
    """Smooth (differentiable) char indicator on the (x,z) plane: ~1 where x < depth*z.

    Matches `cells.char_wedge_cell`'s `mask = xi[:,None] < depth*zi[None,:]` (extruded in y), with a
    sigmoid of width `eps` so it is differentiable in `depth` while staying step-like (sub-voxel)."""
    xi = (torch.arange(n, dtype=torch.float64, device=depth.device) + 0.5) / n
    zi = (torch.arange(n, dtype=torch.float64, device=depth.device) + 0.5) / n
    X, Z = torch.meshgrid(xi, zi, indexing="ij")        # (n,n) over (x,z)
    return torch.sigmoid((depth * Z - X) / eps)         # (n,n), ~1 = char


def features_theta(depth, contrast, n=N, r=R, eps=0.012):
    """Differentiable map theta -> (X_nodes (1,R^3,2), edge_index (2,E), X_desc (1,20)).

    `depth`,`contrast` are scalar torch tensors (requires_grad ok). Reproduces, smoothly, exactly what
    `surrogate_gnn.featurize([char_wedge cell])` computes from the discrete grid."""
    soft = _wedge_soft_field(depth, n=n, eps=eps)        # (n,n) over (x,z), extruded in y
    soft_frac = soft.mean()                              # global char fraction (descriptor ch 18)

    # --- descriptor (fraction-only Voigt/Reuss, identical algebra to violent_cells.descriptor) ---
    E_wood = torch.tensor(E_WOOD, dtype=torch.float64, device=depth.device)
    E_char = E_WOOD / contrast
    Cw, Cc = _iso_C(E_wood), _iso_C(E_char)
    f_char, f_wood = soft_frac, 1.0 - soft_frac
    Cv = f_wood * Cw + f_char * Cc
    Cr = torch.linalg.inv(f_wood * torch.linalg.inv(Cw) + f_char * torch.linalg.inv(Cc))
    dV, dR = torch.diagonal(Cv), torch.diagonal(Cr)
    gap = (dV - dR) / (0.5 * (dV + dR))
    log_contrast = torch.log10(contrast)
    desc = torch.cat([dV, dR, gap, soft_frac.reshape(1), log_contrast.reshape(1)])   # (20,)

    # --- region-graph node features (R^3, 2): [block soft fraction, block mean E / E_WOOD] ---
    # block soft fraction depends on (x-block a, z-block c) only (wedge extruded in y=block b).
    # Vectorized over the R x R blocks (n divisible by R); matches violent_cells.region_graph's
    # (a,b,c) row-major node order and 6-neighbour edge_index.
    assert n % r == 0, "vectorized featurizer needs n divisible by R"
    bs = n // r
    sf_ac = soft.reshape(r, bs, r, bs).mean(dim=(1, 3))          # (R,R) over (x-block, z-block)
    sf = sf_ac[:, None, :].expand(r, r, r)                       # (a,b,c), constant over b (y)
    meanE = 1.0 - sf * (1.0 - 1.0 / contrast)                    # block mean E / E_WOOD
    X_nodes = torch.stack([sf, meanE], dim=-1).reshape(r ** 3, 2)[None]   # (1, R^3, 2)

    edge_index = _edge_index(n, r, depth.device)
    return X_nodes, edge_index, desc[None]              # (1,20)


_EDGE_CACHE = {}


def _edge_index(n, r, device):
    """The fixed R^3 6-neighbour edge_index (same topology violent_cells.region_graph emits)."""
    key = (n, r)
    if key not in _EDGE_CACHE:
        grid = cells.char_wedge_cell(n=n, depth=0.5, contrast=60.0).grid
        _, ei = vc.region_graph(grid, [(E_WOOD, NU), (E_WOOD / 60.0, NU)], R=r)
        _EDGE_CACHE[key] = ei
    return torch.tensor(_EDGE_CACHE[key], dtype=torch.long, device=device)


def features_grid(depth, contrast, n=N, r=R):
    """Reference featurization via the DISCRETE grid (surrogate_gnn.featurize) — for the §A agreement
    self-check that the analytic `features_theta` matches the real pipeline."""
    cell = cells.char_wedge_cell(n=n, depth=float(depth), contrast=float(contrast))
    s = vc.wedge_sample(n, float(depth), float(contrast))
    X_nodes, E, X_desc = featurize([s], R=r)
    return X_nodes, E, X_desc


# ============================================================================================
# Differentiable surrogate forward (no edits to surrogate_gnn; uses its public members/normalizer)
# ============================================================================================
def differentiable_strength(ens: Ensemble, depth, contrast, return_epistemic=False):
    """Mean predicted normalized peak strength, DIFFERENTIABLE in (depth, contrast).

    Replicates `Ensemble.predict`'s mean WITHOUT its `@torch.no_grad()` so gradients flow theta ->
    features -> normalizer -> members -> strength."""
    X_nodes, edge_index, X_desc = features_theta(depth, contrast)
    Xd = ens.normalizer(X_desc)
    mus = []
    for model in ens.members:
        model.eval()
        mu, _ = model(X_nodes, edge_index, Xd)
        mus.append(mu)
    mus = torch.stack(mus)                               # (M,1)
    mean = mus.mean(0).squeeze()
    if return_epistemic:
        return mean, mus.var(0, unbiased=False).squeeze()
    return mean


# ============================================================================================
# Real-operator verification (the independent oracle) + the design envelope
# ============================================================================================
def verify_real(depth, contrast, n=N, params=DATA_PARAMS, cache_dir=None):
    """Re-simulate the candidate cell with the REAL damage-DNS; return its normalized peak strength.
    The same oracle the V2.4 surrogate was trained against. Cached by (depth,contrast)."""
    depth, contrast = float(depth), float(contrast)
    tag = f"d{depth:.4f}_c{contrast:.3f}"
    if cache_dir is not None:
        f = os.path.join(cache_dir, f"v25_{tag}.npz")
        if os.path.exists(f):
            return float(np.load(f)["y"])
    cell = cells.char_wedge_cell(n=n, depth=depth, contrast=contrast)
    res = dd.run_path(cell.grid, cell.materials, params)
    y = outcome_target(res, params.k0)
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(f, y=y)
    return float(y)


def achievable_range(ens, depth_lo, depth_hi, contrast_lo, contrast_hi, m=9):
    """Min/max surrogate strength over a design box — the coherent-target envelope."""
    vals = []
    for d in np.linspace(depth_lo, depth_hi, m):
        for c in np.linspace(contrast_lo, contrast_hi, m):
            with torch.no_grad():
                vals.append(float(differentiable_strength(
                    ens, torch.tensor(d, dtype=torch.float64, device=DEVICE),
                    torch.tensor(c, dtype=torch.float64, device=DEVICE))))
    return float(min(vals)), float(max(vals))


# ============================================================================================
# The inverse-design optimizer + the verified-candidate gate (the adopted artifact)
# ============================================================================================
@dataclass
class Candidate:
    """An inverse-design result: the optimized params, what the surrogate promised, what the REAL
    operator delivered, the trust signal, and the gated status."""
    theta: tuple                  # (depth, contrast) or (P_set,) for physiology
    surrogate_pred: float         # the cheap model's predicted outcome at theta
    verified_real: float          # the real operator's outcome at theta (None if not applicable)
    trust: float                  # distance-to-manifold / uncertainty (lower = more trustworthy)
    status: str                   # "verified" | "untrusted_rve" | "impossible"
    target: float = None
    iters: int = 0


def _optimize_theta(target, ens, iters, lr, grid=17):
    """Global grid search (robust init) + Adam polish, in NORMALIZED coords so depth and contrast share
    a scale (a single lr traverses both). Returns the best (depth, contrast, surrogate_pred). The grid
    guarantees a good init anywhere in the reachable range; the polish refines below grid resolution."""
    dl, dh = DEPTH_BOX
    cl, ch = CONTRAST_BOX
    phys = lambda nd, nc: (dl + nd * (dh - dl), cl + nc * (ch - cl))
    sval = lambda d, c: float(differentiable_strength(
        ens, torch.tensor(d, dtype=torch.float64, device=DEVICE),
        torch.tensor(c, dtype=torch.float64, device=DEVICE)))
    # --- coarse grid search for the init ---
    gs = np.linspace(0.0, 1.0, grid)
    best = None
    with torch.no_grad():
        for nd in gs:
            for nc in gs:
                d, c = phys(nd, nc)
                s = sval(d, c)
                if best is None or abs(s - target) < abs(best[2] - target):
                    best = (nd, nc, s, d, c)
    # --- Adam polish from the grid best (normalized coords, clamped to [0,1]) ---
    nd = torch.tensor(best[0], dtype=torch.float64, device=DEVICE, requires_grad=True)
    nc = torch.tensor(best[1], dtype=torch.float64, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([nd, nc], lr=lr)
    tgt = torch.tensor(float(target), dtype=torch.float64, device=DEVICE)
    for _ in range(iters):
        opt.zero_grad()
        depth = dl + nd * (dh - dl)
        contrast = cl + nc * (ch - cl)
        loss = (differentiable_strength(ens, depth, contrast) - tgt) ** 2
        loss.backward()
        opt.step()
        with torch.no_grad():
            nd.clamp_(0.0, 1.0)
            nc.clamp_(0.0, 1.0)
    with torch.no_grad():
        d, c = phys(float(nd), float(nc))
        s = sval(d, c)
    if abs(s - target) <= abs(best[2] - target):
        return d, c, s
    return best[3], best[4], best[2]      # keep the grid best if polish didn't improve


def inverse_design(target, ens, cache_dir=None, iters=150, lr=0.05,
                   surr_tol=0.02, real_tol=0.12, trust_max=None, seed=0):
    """Gradient-descend theta=(depth,contrast) through the differentiable surrogate to hit `target`
    strength (global grid init + Adam polish), then VERIFY the candidate with the real damage-DNS and
    gate it by trust.

    status:
      "verified"      surrogate converged AND real within real_tol AND trusted
      "untrusted_rve" surrogate converged but real disagrees / off-manifold -> route to RVE (V2.1)
      "impossible"    surrogate cannot reach the target within the design box (out of achievable range)
    """
    torch.manual_seed(seed)
    depth, contrast, s_pred = _optimize_theta(target, ens, iters, lr)
    # trust = distance to the training manifold (FeatureDensity, V2.1/V2.4)
    sample = vc.wedge_sample(N, depth, contrast)
    trust = float(ens.featdensity.score([sample])[0]) if ens.featdensity is not None else 0.0

    converged = abs(s_pred - target) <= surr_tol
    if not converged:
        return Candidate((depth, contrast), s_pred, None, trust, "impossible", target, iters)
    real = verify_real(depth, contrast, cache_dir=cache_dir)
    rel = abs(real - target) / max(abs(target), 1e-9)
    trusted = (trust_max is None) or (trust <= trust_max)
    status = "verified" if (rel <= real_tol and trusted) else "untrusted_rve"
    return Candidate((depth, contrast), s_pred, real, trust, status, target, iters)


# ============================================================================================
# PHYSIOLOGY — the survival-spectrum inverse (regulator)
# ============================================================================================
def homeostatic_pressure(p: reg.RegulatorParams, r):
    """Resting homeostatic pressure = the healthy fixed point's P (None if the creature is dead)."""
    h = reg.healthy_fp(p, r)
    return None if h is None else float(h["P"])


def _dP_dPset(P, r, p):
    """Exact dP*/dP_set at the (unsaturated) healthy fixed point, by implicit differentiation of
    regulator._G(P;r,p)=0 (the architecture's ∂outcome/∂param, in the physiological domain)."""
    a = reg.a_target(P, r, p)
    pmp, dpmp = reg.pump(P, p), reg.pump_deriv(P, p)
    dG_dP = dpmp * (1.0 + p.beta * a) + pmp * p.beta * (-p.K) - p.gamma
    dG_dPset = pmp * p.beta * p.K
    return -dG_dPset / dG_dP if abs(dG_dP) > 1e-12 else 0.0


def inverse_homeostasis(target_P, base: reg.RegulatorParams, r, iters=60, tol=1e-5,
                        P_grid=None, x_grid=None):
    """Newton (implicit-diff) on the setpoint P_set so the resting pressure hits `target_P`, then
    VERIFY with the brute-force basin oracle + viability margin. Returns a Candidate.

    status: "verified" (hit target, healthy FP sits in a non-empty stable basin, margin>0),
            "impossible" (no healthy FP — world below r_critical, or target beyond pump/reserve capacity).
    """
    p = base
    n_it = 0
    for n_it in range(1, iters + 1):
        P = homeostatic_pressure(p, r)
        if P is None:
            return Candidate((float(p.P_set),), None, None, 0.0, "impossible", target_P, n_it)
        if abs(P - target_P) < tol:
            break
        if reg.saturated(P, r, p):          # capacity-limited: P_set cannot move P further
            break
        slope = _dP_dPset(P, r, p)
        if abs(slope) < 1e-9:
            break
        p = replace(p, P_set=float(np.clip(p.P_set - (P - target_P) / slope, 0.1, 12.0)))
    P = homeostatic_pressure(p, r)
    if P is None or abs(P - target_P) > max(50 * tol, 1e-3):
        return Candidate((float(p.P_set),), P, None, 0.0, "impossible", target_P, n_it)
    # verify against the independent oracle: a non-empty basin + positive viability margin
    if P_grid is None:
        P_grid = np.linspace(0.05, 2.6, 13)
    if x_grid is None:
        x_grid = np.linspace(0.0, 1.2, 13)
    area = reg.basin_area(p, r, P_grid, x_grid)
    margin_fn, _ = reg.make_viability_margin(p, r)
    h = reg.healthy_fp(p, r)
    margin = float(margin_fn(h["P"], h["x"])) if margin_fn is not None else float("nan")
    status = "verified" if (area > 0.0 and margin > 0.0) else "untrusted_rve"
    return Candidate((float(p.P_set),), float(P), float(area), float(margin), status, target_P, n_it)


def survival_spectrum(p: reg.RegulatorParams, r_grid, P_grid=None, x_grid=None):
    """Basin area vs reserve — the viability envelope contracting to the impossibility boundary."""
    if P_grid is None:
        P_grid = np.linspace(0.05, 2.6, 13)
    if x_grid is None:
        x_grid = np.linspace(0.0, 1.2, 13)
    return np.array([reg.basin_area(p, r, P_grid, x_grid) for r in r_grid])


# ============================================================================================
# Self-check (regression guard)
# ============================================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    from dns_damage_3d import _HAS_GPU
    print(f"DNS backend: {'GPU' if _HAS_GPU else 'CPU'}; torch device {DEVICE}\n")
    set_determinism(0)

    # train the V2.4 surrogate (RPF for distance-aware trust) on the char-wedge family
    rng = np.random.default_rng(2025)
    train = vc.family_battery(N, rng, 45)
    y_tr = build_dataset(train, DATA_PARAMS, cache="/tmp/v25_calib_train.npz")["y"]
    ens = Ensemble.train(train, y_tr, TrainCfg(epochs=400, beta=0.0), M=5, base_seed=0)
    print("surrogate trained (45 cells, M=5, 400ep).", flush=True)

    # (i) analytic featurizer agrees with the discrete grid pipeline
    d0, c0 = 0.6, 50.0
    Xn_a, _, Xd_a = features_theta(torch.tensor(d0, dtype=torch.float64, device=DEVICE),
                                   torch.tensor(c0, dtype=torch.float64, device=DEVICE))
    Xn_g, _, Xd_g = features_grid(d0, c0)
    desc_err = float(torch.norm(Xd_a - Xd_g) / torch.norm(Xd_g))
    node_err = float(torch.norm(Xn_a - Xn_g) / torch.norm(Xn_g))
    print(f"(i) featurizer agreement: descriptor rel err {desc_err:.4f}, node-feature rel err {node_err:.4f}")
    assert desc_err < 0.02 and node_err < 0.05, "analytic featurizer must match the grid pipeline."

    # (ii) autodiff d(strength)/d(theta) matches finite difference
    depth = torch.tensor(0.55, dtype=torch.float64, device=DEVICE, requires_grad=True)
    contrast = torch.tensor(45.0, dtype=torch.float64, device=DEVICE, requires_grad=True)
    s = differentiable_strength(ens, depth, contrast)
    s.backward()
    g_ad = np.array([float(depth.grad), float(contrast.grad)])
    h = 1e-4
    def sval(d, c):
        with torch.no_grad():
            return float(differentiable_strength(ens, torch.tensor(d, dtype=torch.float64, device=DEVICE),
                                                 torch.tensor(c, dtype=torch.float64, device=DEVICE)))
    g_fd = np.array([(sval(0.55 + h, 45.0) - sval(0.55 - h, 45.0)) / (2 * h),
                     (sval(0.55, 45.0 + h) - sval(0.55, 45.0 - h)) / (2 * h)])
    grad_err = np.linalg.norm(g_ad - g_fd) / (np.linalg.norm(g_fd) + 1e-12)
    print(f"(ii) autodiff vs finite-diff grad rel err {grad_err:.2e}  (ad={g_ad}, fd={g_fd})")
    assert grad_err < 1e-3, "autodiff path is broken."

    f3 = lambda v: f"{v:.3f}" if v is not None else "None"

    # (iii) a coherent INTERIOR target verifies: target = surrogate strength at a deep-interior theta,
    #       so an in-distribution solution provably exists and the real operator agrees within tol.
    flo, fhi = achievable_range(ens, *FAMILY_BOX)
    wlo, whi = achievable_range(ens, *DEPTH_BOX, *CONTRAST_BOX)
    print(f"     family range [{flo:.3f}, {fhi:.3f}]; wide-box reachable [{wlo:.3f}, {whi:.3f}]")
    with torch.no_grad():
        target = float(differentiable_strength(
            ens, torch.tensor(0.5, dtype=torch.float64, device=DEVICE),
            torch.tensor(50.0, dtype=torch.float64, device=DEVICE)))
    cand = inverse_design(target, ens)
    print(f"(iii) coherent interior target {target:.3f}: theta={tuple(round(t,3) for t in cand.theta)} "
          f"surrogate {f3(cand.surrogate_pred)} (gap {abs(cand.surrogate_pred-target):.3f}) "
          f"real {f3(cand.verified_real)} trust {cand.trust:.2f} -> {cand.status}")
    assert cand.status == "verified", "coherent interior target should verify."

    # (iv) a target beyond the WIDE-box reach is reported impossible (not fabricated).
    bad = whi + 0.4 * (whi - wlo)
    cand_bad = inverse_design(bad, ens)
    print(f"(iv) out-of-range target {bad:.3f}: surrogate {f3(cand_bad.surrogate_pred)} -> {cand_bad.status}")
    assert cand_bad.status == "impossible", "out-of-range target must be reported impossible."

    # (v) physiology: a coherent homeostatic target verifies in-basin; r<r_crit returns impossible
    base = reg.RegulatorParams()
    rc = reg.r_critical(base)
    P_base = homeostatic_pressure(base, base.r0)
    ph = inverse_homeostasis(P_base * 1.05, base, base.r0)
    print(f"(v) homeostatic target {P_base*1.05:.3f}: achieved {f3(ph.surrogate_pred)} "
          f"basin_area {f3(ph.verified_real)} margin {f3(ph.trust)} -> {ph.status}")
    assert ph.status == "verified", "coherent homeostatic target should verify in-basin."
    dead = inverse_homeostasis(P_base, base, 0.5 * rc)
    print(f"    world below r_critical (r={0.5*rc:.3f} < {rc:.3f}): -> {dead.status}")
    assert dead.status == "impossible", "a world below r_critical must report impossible."

    print("\ninverse_design self-check PASSED.")
