"""
Wolff's-law strain-energy remodeling — the MECHANISM UNDER TEST for V1.7 (protocol §V1.7;
Decision #22; ARCHITECTURE §III.7).

The architecture's claim: a skeleton is not authored, it **precipitates** — bone condenses
under a stress field exactly as a tree lays reaction wood (Wolff's law as a generative rule).
This module implements that cheap, biologically-motivated, *local feedback* rule and lets
V1.7 judge it against the global optimizer in `topology_opt.py` (SIMP). The whole question is
whether a dumb local rule self-organizes into a structure competitive with the optimum; if it
instead makes disconnected / non-load-bearing junk, the mechanism fails → REDESIGN.

The rule (Huiskes / Mullender homeostatic bone remodeling, fully-stressed form)
------------------------------------------------------------------------------
Carry a continuous density ρ∈[ρ_min,1], seeded FULL (ρ=1) and resorbing — the
resorption-dominated start keeps the load path connected as idle material is removed
(BESO-like), avoiding the traveling fragmentation instability that a grow-from-seed rule
suffers. Each step:
  1. FE-solve the current layout (shared `topology_opt.fe_solve` — identical physics to the
     oracle, so the comparison is fair).
  2. Sense the local mechanical stimulus `S = (spatial filter)·Ue0`, where `Ue0 = uₑᵀKe0uₑ`
     is the *unit-modulus* strain energy at a site — "the energy this site would carry if it
     were solid, given the current strain field." The spatial filter is the biological
     **sensor influence function** (Mullender): it regularizes the response (the mesh-
     independence role the SIMP density filter plays) and lets a site sense load-path demand
     from its neighbourhood rather than only its own (near-void) energy.
  3. Remodel toward a homeostatic setpoint `k` with the **fully-stressed-design multiplicative
     update** `ρ ← ρ·(S/k)^η` (move-limited, clipped to [ρ_min,1]). This is the load-bearing
     fix: the stimulus is steeply density-dependent (`S∝ρ⁻⁴` for a loaded strut), so a *linear*
     `ρ += rate·(S−k)` update is a stiff feedback that oscillates and fragments; the
     multiplicative update is matched to that exponent and converges (η≈0.3 — calibrated).
Equilibrium drives every load-bearing site to `S≈k` — a fully-stressed design, which is
precisely why Wolff ≈ the compliance optimum. `k` is the single dial; it is held FIXED across
a gravity sweep, so a heavier load (higher `S∝g²`) recruits more material (larger skeleton)
and a cancelled load (seraph support-field, `S→0`) resorbs to near-nothing — the morphology-
sensitivity and seraph criteria of V1.7.

Deterministic (fixed-order numpy + the CG-tolerance solve). Imports the FE solver, domains,
filter, and connectivity helpers from `topology_opt.py`; adds no new physics.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve

import topology_opt as to
from homogenization import isotropic_stiffness
from dns_elasticity_3d import element_stiffness


@dataclass(frozen=True)
class WolffParams:
    k: float = 0.12        # homeostatic SED setpoint (THE dial; sets equilibrium volume)
    eta: float = 0.3       # fully-stressed-design damping exponent (η≈0.3 converges; 0.5 rings)
    move: float = 0.1      # per-step density move limit (additive cap on |Δρ|)
    rho_min: float = 1e-3  # resorption floor (kept >0 so the FE system stays non-singular)
    rho_init: float = 1.0  # seed FULL and resorb (BESO-like; preserves connectivity)
    sensor_radius: float = 3.0   # Mullender sensor influence radius (in elements)
    maxit: int = 300
    dtol: float = 5e-3     # convergence: max |Δρ| (continuous field) below this
    window: int = 15


def run_wolff(dims, domain, wp=WolffParams(), fp=to.FEParams(), use_gpu=True,
              verbose=False):
    """Run Wolff (fully-stressed-design) remodeling to a fixed point. Returns a dict."""
    Ke0 = element_stiffness(isotropic_stiffness(fp.E0, fp.nu)) / fp.E0
    edof = to.edof_map(dims)
    ker = to.make_filter(wp.sensor_radius)
    Hs = convolve(np.ones(dims), ker, mode="constant")

    rho = np.full(dims, wp.rho_init)
    hist_vol, hist_c, hist_dr = [], [], []
    converged = False
    for it in range(wp.maxit):
        u, c = to.fe_solve(dims, rho, domain, fp, Ke0=Ke0, edof=edof, use_gpu=use_gpu)
        Ue0, _ = to.element_energy(dims, u, edof, rho, Ke0, fp)
        S = convolve(Ue0, ker, mode="constant") / Hs        # sensed stimulus (mesh-indep.)
        target = rho * np.power(np.maximum(S, 1e-30) / wp.k, wp.eta)   # fully-stressed update
        rho_new = np.clip(np.clip(target, rho - wp.move, rho + wp.move), wp.rho_min, 1.0)
        max_dr = float(np.abs(rho_new - rho).max())
        rho = rho_new
        hist_vol.append(float((rho > 0.5).mean()))
        hist_c.append(c)
        hist_dr.append(max_dr)
        if verbose:
            print(f"  it {it:3d}  c={c:.4e}  solid_frac={hist_vol[-1]:.3f}  max|dρ|={max_dr:.4f}")
        if it >= wp.window and max_dr < wp.dtol:
            converged = True
            break

    solid_frac = float((rho > 0.5).mean())
    return {
        "rho": rho,
        "solid_frac": solid_frac,
        "compliance": hist_c[-1],
        "n_iter": len(hist_c),
        "hist_vol": np.array(hist_vol),
        "hist_c": np.array(hist_c),
        "hist_dr": np.array(hist_dr),
        "converged": converged,
    }


if __name__ == "__main__":
    import time
    np.set_printoptions(precision=4, suppress=True)
    print(f"backend: {'GPU (cupy CG)' if to._HAS_GPU else 'CPU (sparse LU)'}\n")
    fp = to.FEParams()

    # 1) cantilever: Wolff converges to a fixed point and self-selects a volume.
    dims = (32, 12, 6)
    dom, supp = to.build_cantilever(dims, load=1.0)
    wp = WolffParams()
    t = time.time()
    res = run_wolff(dims, dom, wp=wp, fp=fp)
    dt = time.time() - t
    print("1) cantilever Wolff (dims=%s, %d its, %.1fs)" % (dims, res["n_iter"], dt))
    print("   converged=%s  solid_frac=%.3f  compliance=%.4e"
          % (res["converged"], res["solid_frac"], res["compliance"]))
    assert res["converged"], "Wolff did not reach a fixed point"
    assert 0.05 < res["solid_frac"] < 0.95, "degenerate settled volume"

    # 2) settled structure is connected and load-bearing.
    rb = to.binarize(res["rho"], res["solid_frac"])
    con = to.connectivity(rb, supp)
    print("2) Wolff binary design: %s" % con)
    assert con["connected"], "Wolff structure not connected/load-bearing"

    # 3) competitive with SIMP at matched (continuous) volume — the V1.7 comparison, previewed.
    #    Continuous compliance at matched mean-ρ: the fair, standard metric (SIMP is the
    #    minimizer ⇒ ratio ≥ 1); binarizing first would inject a thresholding-disconnection.
    Vc = float(res["rho"].mean())
    rho_s, _ = to.simp_optimize(dims, dom, volfrac=Vc, params=fp, n_iter=50)
    c_w = to.compliance_of(dims, res["rho"], dom, fp)
    c_s = to.compliance_of(dims, rho_s, dom, fp)
    print("3) matched-volume compliance: Wolff=%.4e  SIMP=%.4e  ratio=%.3f (Vcont=%.3f)"
          % (c_w, c_s, c_w / c_s, Vc))
    assert c_w / c_s < 2.0, "Wolff far from optimal"

    # 4) determinism: repeat run identical to CG tolerance.
    res2 = run_wolff(dims, dom, wp=wp, fp=fp)
    rel = abs(res["compliance"] - res2["compliance"]) / abs(res["compliance"])
    print("4) determinism: compliance rel diff on repeat = %.2e" % rel)
    assert rel < 1e-6, "non-deterministic Wolff result"
    print("\nall wolff self-checks passed.")
