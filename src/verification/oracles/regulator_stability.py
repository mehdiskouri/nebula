"""
Regulator numerical stability — passivity vs a naïve force controller (V1.6).
Protocol §V1.6; Risk: regulator gain tuning / limit cycles; ARCHITECTURE Part IV guardrail #1
("laws as energies/potentials, not raw forces") and §III.6. Depends on V1.5 (`regulator.py`).

A coupled nonlinear feedback regulator can settle into a spurious **limit cycle** — sustained
oscillation in a steady environment. It looks like the creature *trembling*, but it is the model
ringing, not physiology. The architecture's guardrail is **passivity**: formulate actuation as a
**dissipative draw on a bounded reserve** (energy-shaping + damping injection) so the closed loop
is passive and cannot self-oscillate. V1.6 falsifies the claim that the passive formulation has a
**strictly larger oscillation-free gain region** than a naïve force-style controller.

Model — the V1.5 plant with an INERTIAL actuator
-------------------------------------------------
V1.5's actuator is first-order (`ẋ=(a_target−x)/τ`): a single real pole that cannot oscillate.
To exhibit (and then suppress) the trembling we give the actuator inertia (second order), so the
lagged negative-feedback loop Hopf-bifurcates at high gain. State (P, x, v), v=ẋ, at full reserve:

    Ṗ = pump(P)·(1+β x) − γ P                     # reuse regulator.pump (the V1.5 plant)
    ẋ = v
    v̇ = ω²·(a_target(P) − x) − d_eff·v            # damped 2nd-order tracking of the demanded tone

with a_target(P) = clip(K·(P_set − P), 0, a_max) the regulator demand (loop gain K is the swept
control). Two controllers differ ONLY in the dissipation:
  * naïve (force-style): d_eff = d0           (fixed light intrinsic damping)
  * passivity:           d_eff = d0 + c_d     (extra velocity-proportional dissipation injected by
                                               the controller, a draw on the bounded reserve)

Linear-stability oracle (the independent ground truth)
------------------------------------------------------
Linearizing at the healthy fixed point (a = ∂Ṗ/∂P = pump'(P*)(1+βx*)−γ < 0, b = ∂Ṗ/∂x =
β·pump(P*) > 0, ∂a_target/∂P = −K on the active branch) gives the cubic

    λ³ + (d_eff − a) λ² + (ω² − a d_eff) λ + ω²(bK − a).

The Routh condition for a complex pair crossing the imaginary axis (Hopf) is
(d_eff−a)(ω²−a d_eff) = ω²(bK − a), i.e. the eigenvalues give the **exact analytic Hopf gain**.
Because d_eff is larger under passivity, K_hopf(passive) > K_hopf(naïve): the oscillation-free
region is strictly larger by construction. The eigenvalue boundary is judged against the NONLINEAR
truth — integrate the full ODE and detect a sustained limit cycle (post-transient peak-to-peak of
P above a floor). Pure numpy + scipy. Reuses regulator.py; leaves it untouched.
"""
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp

import regulator as rg


@dataclass(frozen=True)
class StabilityParams:
    omega: float = 2.0                                  # actuator natural frequency (stiffness)
    d0: float = 1.65                                    # intrinsic actuator damping (naïve)
    c_d: float = 1.5                                    # extra dissipation injected by passivity
    r_op: float = 1.0                                   # operating reserve (ample — testing oscillation, not death)
    plant: rg.RegulatorParams = field(default_factory=rg.RegulatorParams)


def d_eff(sp: StabilityParams, controller):
    return sp.d0 if controller == "naive" else sp.d0 + sp.c_d


# ---------------- plant fixed point & linearization (K-dependent) ----------------

def healthy_state(sp: StabilityParams, K):
    """The (P*, x*, 0) operating point for loop gain K, or None if no healthy FP exists."""
    h = rg.healthy_fp(replace(sp.plant, K=K), sp.r_op)
    return None if h is None else (h["P"], h["x"], 0.0)


def _lin_coef(sp: StabilityParams, K):
    """Plant linearization at the K-dependent healthy FP: (a = ∂Ṗ/∂P, b = ∂Ṗ/∂x, saturated?),
    or None if no healthy operating point exists at this gain (regulator too weak to sustain life)."""
    p = replace(sp.plant, K=K)
    h = rg.healthy_fp(p, sp.r_op)
    if h is None:
        return None
    P, x = h["P"], h["x"]
    a = rg.pump_deriv(P, p) * (1.0 + p.beta * x) - p.gamma
    b = p.beta * rg.pump(P, p)
    return a, b, rg.saturated(P, sp.r_op, p)


def jacobian(sp: StabilityParams, K, controller):
    """3x3 closed-loop Jacobian at the healthy FP, or None if no healthy FP exists.
    ∂a_target/∂P = −K on the active branch."""
    lc = _lin_coef(sp, K)
    if lc is None:
        return None
    a, b, sat = lc
    dK = 0.0 if sat else K                              # saturated ⇒ no proportional coupling
    de = d_eff(sp, controller)
    return np.array([[a, b, 0.0],
                     [0.0, 0.0, 1.0],
                     [-sp.omega**2 * dK, -sp.omega**2, -de]])


def max_real_eig(sp: StabilityParams, K, controller):
    """THE ORACLE — the leading eigenvalue real part (>0 ⇒ linearly unstable / Hopf). Returns +inf
    when no healthy operating point exists (not a viable stable regime)."""
    J = jacobian(sp, K, controller)
    if J is None:
        return np.inf
    return float(np.linalg.eigvals(J).real.max())


def hopf_gain(sp: StabilityParams, controller, K_lo=1.0, K_hi=8.0, iters=60):
    """Critical loop gain where max Re(λ) crosses 0 (bisection), or None if stable across [lo,hi].
    K_lo defaults to a gain with a stable healthy FP (life requires a minimum regulation gain)."""
    if max_real_eig(sp, K_lo, controller) > 0:
        return K_lo
    if max_real_eig(sp, K_hi, controller) < 0:
        return None                                    # no Hopf in range ⇒ larger stable region
    for _ in range(iters):
        mid = 0.5 * (K_lo + K_hi)
        if max_real_eig(sp, mid, controller) < 0:
            K_lo = mid
        else:
            K_hi = mid
    return 0.5 * (K_lo + K_hi)


# ---------------- nonlinear truth: limit-cycle detection ----------------

def rhs(t, state3, sp: StabilityParams, K, controller):
    P, x, v = state3
    p = sp.plant
    at = min(max(K * (p.P_set - P), 0.0), rg.a_max(sp.r_op, p))
    dP = rg.pump(P, p) * (1.0 + p.beta * x) - p.gamma * P
    return [dP, v, sp.omega**2 * (at - x) - d_eff(sp, controller) * v]


def limit_cycle_amplitude(sp: StabilityParams, K, controller, kick=0.05, tmax=500.0):
    """Integrate from a small kick off the FP; post-transient peak-to-peak of P (the sustained
    oscillation amplitude). ~0 ⇒ settled; > floor ⇒ a limit cycle (trembling)."""
    h = healthy_state(sp, K)
    if h is None:
        return 0.0
    P0, x0, _ = h
    sol = solve_ivp(rhs, [0.0, tmax], [P0 - kick, x0, 0.0], args=(sp, K, controller),
                    rtol=1e-8, atol=1e-10, method="LSODA", max_step=0.2,
                    t_eval=np.linspace(0.7 * tmax, tmax, 2500))
    return float(sol.y[0].max() - sol.y[0].min())


def nonlinear_onset(sp: StabilityParams, controller, K_grid, floor=0.02):
    """Lowest swept gain at which a sustained limit cycle appears, or None."""
    for K in K_grid:
        if limit_cycle_amplitude(sp, K, controller) > floor:
            return float(K)
    return None


# ---------------- 2-D stable region (gain x intrinsic damping) ----------------

def stable_region(sp: StabilityParams, controller, K_grid, d_grid):
    """Boolean grid[i,j] = linearly stable at (K_grid[j], d0=d_grid[i]) for this controller.
    Rows = intrinsic damping d0, cols = loop gain K."""
    mask = np.zeros((len(d_grid), len(K_grid)), bool)
    for i, d in enumerate(d_grid):
        sp_d = replace(sp, d0=d)
        for j, K in enumerate(K_grid):
            mask[i, j] = max_real_eig(sp_d, K, controller) < 0
    return mask


def region_area(sp: StabilityParams, controller, K_grid, d_grid):
    return float(stable_region(sp, controller, K_grid, d_grid).mean())


# ---------------- supporting: actuator storage (Lyapunov) energy ----------------

def storage_energy(state3, sp: StabilityParams, K):
    """Actuator mechanical storage E = ½v² + ½ω²(x − x*)² about the operating tone — a Lyapunov
    candidate. Monotone decay ⇒ passive/dissipative; growth/ringing ⇒ the naïve loop pumping energy."""
    h = healthy_state(sp, K)
    x_star = h[1] if h else 0.0
    P, x, v = state3
    return 0.5 * v**2 + 0.5 * sp.omega**2 * (x - x_star) ** 2


# ---------------- self-check (regression guard) ----------------

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    sp = StabilityParams()
    K_grid = np.linspace(0.1, 6.0, 60)

    kh_naive = hopf_gain(sp, "naive")
    kh_pass = hopf_gain(sp, "passive")
    on_naive = nonlinear_onset(sp, "naive", K_grid)
    on_pass = nonlinear_onset(sp, "passive", K_grid)
    print(f"1) naïve  : K_hopf(eig)={kh_naive:.3f}  nonlinear onset={on_naive}")
    print(f"   passive: K_hopf(eig)={kh_pass:.3f}  nonlinear onset={on_pass}")

    # A — eigenvalue oracle matches the nonlinear onset (naïve, which actually oscillates)
    err = abs(on_naive - kh_naive) / kh_naive
    print(f"\n2) oracle agreement (naïve): |onset−K_hopf|/K_hopf = {err:.1%}")
    assert err <= 0.10

    # B — passivity's oscillation-free region is strictly larger
    print(f"\n3) passive K_hopf / naïve K_hopf = {kh_pass / kh_naive:.2f}x")
    assert kh_pass >= 2.0 * kh_naive
    Kg = np.linspace(0.5, 5.0, 28); dg = np.linspace(0.5, 2.5, 24)
    a_naive = region_area(sp, "naive", Kg, dg); a_pass = region_area(sp, "passive", Kg, dg)
    print(f"   2-D stable-region area: naïve={a_naive:.3f}  passive={a_pass:.3f}")
    assert a_pass > a_naive

    # C — at the production gain the naïve loop trembles, the passive one does not
    Kp = 2.0
    amp_n = limit_cycle_amplitude(sp, Kp, "naive"); amp_p = limit_cycle_amplitude(sp, Kp, "passive")
    print(f"\n4) production gain K={Kp}: naïve amp={amp_n:.3f} (trembles)  passive amp={amp_p:.3e} (settles)")
    assert amp_n > 0.05 and amp_p < 0.02

    print("\nOK — passivity has a strictly larger oscillation-free gain region; the linear-stability")
    print("     oracle predicts the limit-cycle onset; the naïve force controller trembles where the")
    print("     passive (reserve-dissipating) one stays still.")
