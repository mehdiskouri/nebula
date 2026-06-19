"""
Jensen sub-cell-variance correction for nonlinear (Arrhenius) rates — the mechanism
under test for V1.3 (protocol §V1.3; Decision #16; ARCHITECTURE §III.4 "the nonlinear trap").

A homogenized cell carries a single mean temperature T̄. But the reaction rate is the
Arrhenius law g(T) = exp(-Ta/T), which is CONVEX over the physical range (Ta/T ≫ 2), so by
Jensen's inequality the true cell-averaged rate ⟨g(T)⟩ EXCEEDS g(T̄): lumping at the mean
systematically UNDER-estimates the rate. A cell with a 700°C face and a 60°C core has a
mean temperature whose Arrhenius rate badly underestimates the face's pyrolysis — and in a
coupled burn that error can SILENTLY EXTINGUISH the fire (a correctness failure masquerading
as physics).

The fix the architecture names: carry the sub-cell VARIANCE σ²_T of hot fields and apply a
second-order correction
        g_corr(T̄, σ²) = g(T̄) + ½ g''(T̄) σ²_T          (g'' > 0  ⇒  correction > 0).
This is the SECOND tracked homogenization error — the variance term that joins V0.1's
Voigt–Reuss gap (the responses term). The 2nd-order truncation itself fails once higher
moments dominate; that breakdown is wired as a refine trigger via `variance_error_scalar`.

Regression-safe: imports the rate-law CONSTANTS from fire_operators (FireParams: A_py/Ta_py,
A_cb/Ta_cb, …) but leaves fire_operators / bus_runtime / multirate / monolithic_fire /
semi_implicit_fire UNTOUCHED. The fine-scale extinction reference reuses monolithic_fire.
Pure numpy.
"""
import numpy as np

import fire_operators as fo
import bus_runtime as br

EPS = 1e-300


# ---------------- Arrhenius value + T-derivatives ----------------

def g(T, Ta):
    """Arrhenius shape g(T) = exp(-Ta/T)."""
    return np.exp(-Ta / np.maximum(T, 1.0))


def g1(T, Ta):
    """g'(T) = g(T) · Ta/T²  (> 0: rate rises with T)."""
    T = np.maximum(T, 1.0)
    return g(T, Ta) * Ta / T**2


def g2(T, Ta):
    """g''(T) = g(T) · (Ta/T³)(Ta/T − 2)  (> 0 for Ta/T > 2, i.e. always here)."""
    T = np.maximum(T, 1.0)
    return g(T, Ta) * (Ta / T**3) * (Ta / T - 2.0)


# ---------------- three rate estimators (per unit reactant: rate/A factored as A·ĝ) ----------------
# A T-field carries the sub-cell temperature distribution; the reactant marginals (m_s, gas, o2)
# are held UNIFORM so the only Jensen effect is the T-nonlinearity (the architecture's framing).

def mean_only_rate(Tfield, A, Ta):
    """Naive lumped rate: evaluate the Arrhenius law at the cell mean. A · g(T̄)."""
    return A * g(float(np.mean(Tfield)), Ta)


def true_mean_rate(Tfield, A, Ta):
    """THE ORACLE: fine-scale integral of the rate over the resolved sub-cell field. A · ⟨g(T)⟩."""
    return A * float(np.mean(g(np.asarray(Tfield, float), Ta)))


def variance_corrected_rate(Tfield, A, Ta):
    """Second-order (Jensen) correction: A · (g(T̄) + ½ g''(T̄) σ²_T)."""
    T = np.asarray(Tfield, float)
    Tbar = float(T.mean()); var = float(T.var())
    return A * (g(Tbar, Ta) + 0.5 * g2(Tbar, Ta) * var)


# moment-based forms (for the lumped burn, which carries only T̄ and σ²) -----------------
def mean_only_from_moments(Tbar, A, Ta):
    return A * g(Tbar, Ta)


def corrected_from_moments(Tbar, var, A, Ta):
    return A * (g(Tbar, Ta) + 0.5 * g2(Tbar, Ta) * var)


def variance_error_scalar(Tbar, var, Ta):
    """The dimensionless magnitude of the 2nd-order term, ε = ½ σ² |g''(T̄)/g(T̄)|.

    This is the companion of the Voigt–Reuss gap: refine when ε exceeds the documented
    validity edge ε* (where higher moments overwhelm the 2nd-order truncation)."""
    return 0.5 * var * abs(g2(Tbar, Ta) / (g(Tbar, Ta) + EPS))


# ---------------- the 3-profile sub-cell T-field battery ----------------
# Each builds an (n,n,n) temperature field spanning [T_lo (cold core) .. T_hi (hot face)] with a
# different SHAPE; sweeping T_hi raises both the mean and the variance (a hotter face on a cold core).

def ramp_field(n, T_lo, T_hi):
    """Linear hot-face → cold-core gradient (the 700°C/60°C example). Var = ΔT²/12."""
    x = (np.arange(n) + 0.5) / n
    col = T_lo + (T_hi - T_lo) * x
    return np.broadcast_to(col[:, None, None], (n, n, n)).copy()


def boundary_layer_field(n, T_lo, T_hi, delta=0.15):
    """Exponential thermal boundary layer: hot at the face, decays into a cold bulk.

    Variance is concentrated in a thin hot skin (small `delta`) — most voxels near T_lo,
    a few near T_hi: a long hot tail that the symmetric 2nd-order term captures only partly."""
    x = (np.arange(n) + 0.5) / n
    col = T_lo + (T_hi - T_lo) * np.exp(-x / delta)
    return np.broadcast_to(col[:, None, None], (n, n, n)).copy()


def bimodal_field(n, T_lo, T_hi, hot_frac=0.25):
    """A hot sub-volume (fraction `hot_frac`) embedded in a cold matrix — strongly NON-Gaussian.

    Var = f(1−f)ΔT²; the missing higher (even) moments are large, so the 2nd-order correction
    breaks down EARLIEST here → the worst case the refine trigger must catch."""
    field = np.full((n, n, n), float(T_lo))
    k = max(int(round(hot_frac * n)), 1)
    field[:k] = T_hi
    return field


PROFILES = {"ramp": ramp_field, "boundary_layer": boundary_layer_field, "bimodal": bimodal_field}


# ---------------- the spurious-extinction demo (load-bearing payoff) ----------------
# A single coarse cell against a sustained sub-cell gradient (variance held — the persistent
# unresolved reaction front). The self-sustaining loop: hot face → pyrolysis releases volatiles
# (gas) → gas+O2 combust (exothermic) → hotter; boundary loss opposes it. Mean-only lumping
# under-produces gas → loss wins → EXTINCTION; the variance correction captures the hot-face
# pyrolysis → the burn SUSTAINS, matching the fine-scale truth.

def run_lumped(Tbar0, var, p, dt, nsteps, corrected, m_s0=1.0, gas0=0.02, o2_0=0.23):
    """Integrate the lumped scalar cell with either the mean-only or variance-corrected rate.

    Returns (T_trace, fuel_consumed_trace) — both length nsteps+1. `var` is the held sub-cell
    temperature variance (0 ⇒ mean-only and corrected coincide). Mean-only under-estimates the
    Arrhenius rate of a hot-faced cell and the reaction stalls — the spurious extinction."""
    Tbar, m_s, gas, o2 = float(Tbar0), float(m_s0), float(gas0), float(o2_0)
    m0 = m_s; T_tr = [Tbar]; fuel_tr = [0.0]
    for _ in range(nsteps):
        o2 += p.o2_influx * (p.o2_amb - o2) * dt                       # replenished (open cell)
        if corrected:
            kpy = corrected_from_moments(Tbar, var, p.A_py, p.Ta_py)
            kcb = corrected_from_moments(Tbar, var, p.A_cb, p.Ta_cb)
        else:
            kpy = mean_only_from_moments(Tbar, p.A_py, p.Ta_py)
            kcb = mean_only_from_moments(Tbar, p.A_cb, p.Ta_cb)
        r_py = kpy * max(m_s, 0.0)                                     # gas source
        r_cb = kcb * max(gas, 0.0) * max(o2, 0.0)                      # gas/O2 sink, heat source
        Tbar += (p.dH_cb * r_cb - p.dH_py * r_py - p.h_loss * (Tbar - p.T_amb)) / p.C_V * dt
        m_s = max(m_s - r_py * dt, 0.0)
        gas = max(gas + (p.nu_g * r_py - r_cb) * dt, 0.0)
        o2 = max(o2 - r_cb * dt, 0.0)
        T_tr.append(Tbar); fuel_tr.append(m0 - m_s)
    return np.array(T_tr), np.array(fuel_tr)


def run_fine(field0, p, dt, nsteps, m_s0=1.0, gas0=0.02, o2_0=0.23):
    """Fine-scale truth: a small grid carrying the resolved sub-cell gradient, advanced by the
    real coupled fire system at full resolution (every sub-voxel gets its OWN rate — no
    homogenization). Uses the V1.2-validated rate-substepped split (`multirate`, <0.02% vs the
    monolithic oracle) so a vigorous hot-face burn stays affordable. Returns the cell-mean
    T-trace (per global step) and total fuel consumed."""
    import multirate as mr
    n = field0.shape[0]
    st = {"T": field0.astype(float).copy(),
          "m_s": np.full((n, n, n), float(m_s0)), "gas": np.full((n, n, n), float(gas0)),
          "o2": np.full((n, n, n), float(o2_0)), "char": np.zeros((n, n, n)),
          "q": np.zeros((n, n, n))}
    m0 = float(st["m_s"].sum()); ncell = st["m_s"].size
    T_tr = [float(st["T"].mean())]; fuel_tr = [0.0]
    for _ in range(nsteps):
        st, _ = mr.step_split_substep(st, p, dt)
        T_tr.append(float(st["T"].mean()))
        fuel_tr.append((m0 - float(st["m_s"].sum())) / ncell)         # per-cell, comparable to lumped
    return np.array(T_tr), np.array(fuel_tr)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    p = fo.FireParams()
    n, Ta = 24, p.Ta_py

    # 1) convexity + Jensen sign: g'' > 0, and true mean-rate >= mean-only across steepness.
    print("1) Jensen sign (pyrolysis, cold core 350K, hot face swept):")
    for T_hi in (450, 600, 800, 1000):
        f = ramp_field(n, 350.0, T_hi)
        mo = mean_only_rate(f, p.A_py, Ta); tr = true_mean_rate(f, p.A_py, Ta)
        assert g2(f.mean(), Ta) > 0 and tr >= mo - 1e-30
        print(f"   face={T_hi}K  T̄={f.mean():.0f}  mean-only={mo:.3e}  true={tr:.3e}  "
              f"under-est={100*(1-mo/tr):.0f}%")

    # 2) corrected recovers true at mild steepness, then the 2nd-order truncation diverges.
    print("\n2) corrected vs true (ramp), and the variance-error scalar ε:")
    for T_hi in (450, 600, 800, 1000):
        f = ramp_field(n, 350.0, T_hi)
        tr = true_mean_rate(f, p.A_py, Ta); co = variance_corrected_rate(f, p.A_py, Ta)
        eps = variance_error_scalar(f.mean(), f.var(), Ta)
        print(f"   face={T_hi}K  corrected err={100*abs(co-tr)/tr:5.1f}%  ε={eps:7.3f}")
