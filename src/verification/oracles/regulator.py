"""
Minimal regulated cardiovascular loop + bounded reserve — the mechanism under test for
V1.5 (protocol §V1.5; Decisions #19, #21; ARCHITECTURE §III.6 "Regulators — living things").

A creature adds exactly ONE new primitive to the passive-asset machinery: the regulator — a
closed-loop controller that senses a variable, compares it to a setpoint, and drives an actuator
to close the error (negative feedback — the thing a tree utterly lacks). The architecture's
load-bearing guardrail: a regulator may only actuate by SPENDING A BOUNDED, CONSERVED RESERVE
against a finite capacity. This is what makes a creature "killable correctly" — bleed it, the
reserve depletes, the controller saturates, correction fails, and it dies. Mortality is then
EMERGENT from a conserved quantity running out, not a hit-point counter; and the VIABILITY MARGIN
(how far from the envelope boundary) plays the exact role the homogenization bound plays for
passive matter (ARCHITECTURE Part IV — the single currency).

The minimal model (2-D fast loop, reserve `r` as the swept parameter)
------------------------------------------------------------------------
State (P, x):  P = mean perfusion pressure (the sensed variable);  x = autonomic / pump tone (the
actuator state). Reserve level r ∈ (0, r0] sets the actuator's finite capacity a_max(r)=a_cap·r/r0.

    dP/dt = pump(P)·(1 + β·x) − γ·P          pump(P) = Pmax·Pⁿ/(Pⁿ + Pcⁿ)   (sigmoidal cardiac curve)
    dx/dt = (1/τ)·[ a_target(P; r) − x ]      a_target = clip( K·(P_set − P), 0, a_max(r) )

`pump(0)=0` ⇒ P=0 is an EXACT absorbing collapse (low P → weak pump → lower P: the hemorrhagic
spiral). γ is chosen so the BARE pump (x=0) has NO upper equilibrium — the healthy state exists
ONLY because the reserve-fed regulator boosts the pump. Structure: a healthy stable node (high P)
and the absorbing collapse, separated by a SADDLE whose stable manifold is the separatrix (the
viability-envelope boundary). At full reserve the controller runs UNSATURATED (active negative
feedback, headroom); as r depletes a_max falls, the controller SATURATES, the healthy node and the
saddle approach and finally annihilate at a SADDLE-NODE bifurcation r_crit — past which only death
exists. The viable basin contracts monotonically to zero as r → r_crit.

Coupled slow reserve (the emergent-mortality demonstration)
-----------------------------------------------------------
    dr/dt = −c·max(x − x_base, 0) + s·(r0 − r)
The reserve is itself a conserved store: working the actuator above baseline draws it down; rest
refills it. A sustained bleed (a step reserve sink, or a forced low-P load) depletes r, dragging
the operating point across the saddle-node into the absorbing cascade — death because the reserve
ran out, exactly as the architecture claims.

Independent oracle (protocol §V1.5)
-----------------------------------
The shippable object is the CHEAP `viability_margin` scalar (signed distance to the separatrix:
>0 alive, <0 dead). The GROUND TRUTH, obtained a different way, is the BRUTE-FORCE BASIN: integrate
the ODE from a dense grid of initial states and label each by its attractor. V1.5 passes iff the
cheap predicate agrees with the true basin, the basin contracts monotonically with reserve, and
collapse is absorbing.

Pure numpy + scipy (solve_ivp, brentq). Self-contained; touches no other oracle.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@dataclass(frozen=True)
class RegulatorParams:
    # cardiac pump (sigmoidal Hill curve) + systemic runoff
    Pmax: float = 1.0          # max pump magnitude
    Pc: float = 0.6            # half-activation pressure of the pump
    n: float = 4.0             # Hill steepness
    gamma: float = 1.4         # systemic runoff rate (> max bare pump/P ⇒ no unregulated healthy FP)
    beta: float = 2.0          # how strongly autonomic tone boosts the pump
    # regulator (baroreflex): drive tone to close the pressure error, capped by the reserve
    P_set: float = 1.4         # pressure setpoint
    K: float = 2.0             # proportional gain
    tau: float = 0.1           # actuator time constant (fast vs the perfusion loop)
    a_cap: float = 1.0         # actuator capacity at full reserve
    r0: float = 1.0            # full reserve level
    # slow conserved reserve (for the coupled death-cascade demonstration)
    c: float = 0.15            # reserve drawn per unit supra-baseline tone
    s: float = 0.05            # reserve refill rate toward r0
    x_base: float = 0.2        # baseline tone that costs nothing to hold


# ---------------- constitutive pieces ----------------

def pump(P, p: RegulatorParams):
    """Sigmoidal cardiac output; pump(0)=0 so collapse P=0 is an exact absorbing state.
    Written as Pmax/(1+(Pc/P)ⁿ) — algebraically identical but overflow-safe for large P."""
    P = np.maximum(P, 1e-12)
    return p.Pmax / (1.0 + (p.Pc / P) ** p.n)


def pump_deriv(P, p: RegulatorParams):
    P = np.maximum(P, 1e-12)
    num = p.Pmax * p.n * P ** (p.n - 1) * p.Pc ** p.n
    return num / (P ** p.n + p.Pc ** p.n) ** 2


def a_max(r, p: RegulatorParams):
    """Actuator capacity set by the bounded reserve (the conserved-reserve guardrail)."""
    return p.a_cap * r / p.r0


def a_target(P, r, p: RegulatorParams):
    """Regulator demand: proportional error-closing, clipped to [0, a_max(r)]."""
    return min(max(p.K * (p.P_set - P), 0.0), a_max(r, p))


def saturated(P, r, p: RegulatorParams):
    """True when the controller is pegged at capacity (the failure mode as reserve depletes)."""
    return p.K * (p.P_set - P) >= a_max(r, p) - 1e-12


# ---------------- dynamics ----------------

def rhs(state, p: RegulatorParams, r):
    """The 2-D fast loop d(P, x)/dt at fixed reserve r."""
    P, x = state
    dP = pump(P, p) * (1.0 + p.beta * x) - p.gamma * P
    dx = (a_target(P, r, p) - x) / p.tau
    return np.array([dP, dx])


def rhs_full(state3, p: RegulatorParams):
    """The 3-D system with the slow conserved reserve d(P, x, r)/dt (coupled demonstration)."""
    P, x, r = state3
    dP = pump(P, p) * (1.0 + p.beta * x) - p.gamma * P
    dx = (a_target(P, r, p) - x) / p.tau
    dr = -p.c * max(x - p.x_base, 0.0) + p.s * (p.r0 - r)
    return np.array([dP, dx, dr])


# ---------------- fixed points, Jacobian, classification ----------------

def _G(P, r, p):
    """Reduced 1-D equilibrium residual: at a fixed point x* = a_target(P), so dP/dt = G(P)."""
    return pump(P, p) * (1.0 + p.beta * a_target(P, r, p)) - p.gamma * P


def positive_pressures(p: RegulatorParams, r, hi=None, N=8000):
    """All P>0 equilibrium pressures (bracket-scan + brentq on the reduced residual)."""
    hi = hi if hi is not None else 3.0 * p.P_set
    Ps = np.linspace(1e-4, hi, N)
    g = np.array([_G(P, r, p) for P in Ps])
    roots = []
    for i in range(N - 1):
        if g[i] == 0.0:
            roots.append(Ps[i])
        elif g[i] * g[i + 1] < 0:
            roots.append(brentq(_G, Ps[i], Ps[i + 1], args=(r, p)))
    return sorted(roots)


def jacobian(state, p: RegulatorParams, r):
    """Analytic 2x2 Jacobian of `rhs` (saturation branch from the local demand)."""
    P, x = state
    da = 0.0 if saturated(P, r, p) else -p.K          # ∂a_target/∂P on the active branch
    J = np.array([[pump_deriv(P, p) * (1.0 + p.beta * x) - p.gamma, p.beta * pump(P, p)],
                  [da / p.tau, -1.0 / p.tau]])
    return J


def classify(state, p: RegulatorParams, r):
    """'stable_node' (all Re<0), 'saddle' (mixed sign), or 'unstable' (all Re>0)."""
    ev = np.linalg.eigvals(jacobian(state, p, r)).real
    if ev.max() < -1e-9:
        return "stable_node"
    if ev.min() < -1e-9 < ev.max():
        return "saddle"
    return "unstable"


def fixed_points(p: RegulatorParams, r):
    """Every equilibrium of the fast loop as dicts {P, x, kind, eig}.

    Always includes the collapse state (P=0, x=a_max) — the absorbing death attractor —
    plus the saddle and the healthy node when they exist."""
    out = []
    for P in [0.0] + positive_pressures(p, r):
        x = a_target(P, r, p)
        out.append(dict(P=P, x=x, kind=classify((P, x), p, r),
                        eig=np.linalg.eigvals(jacobian((P, x), p, r))))
    return out


def healthy_fp(p: RegulatorParams, r):
    """The highest-pressure stable node (the homeostatic operating point), or None if dead."""
    cand = [fp for fp in fixed_points(p, r) if fp["kind"] == "stable_node" and fp["P"] > 1e-6]
    return max(cand, key=lambda fp: fp["P"]) if cand else None


def saddle_fp(p: RegulatorParams, r):
    saddles = [fp for fp in fixed_points(p, r) if fp["kind"] == "saddle"]
    return saddles[0] if saddles else None


def r_critical(p: RegulatorParams, lo=1e-3, hi=None, iters=60):
    """The saddle-node reserve below which no healthy fixed point exists (life impossible)."""
    hi = hi if hi is not None else p.r0
    if healthy_fp(p, hi) is None:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if healthy_fp(p, mid) is not None:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------- the oracle: brute-force basin of attraction ----------------

def converges_healthy(P0, x0, p: RegulatorParams, r, Ph, t_max=200.0):
    """Integrate the fast loop from (P0, x0); True iff it ends at the healthy node."""
    sol = solve_ivp(lambda t, s: rhs(s, p, r), [0.0, t_max], [P0, x0],
                    rtol=1e-7, atol=1e-9, method="LSODA")
    return sol.y[0, -1] > 0.5 * Ph


def basin_map(p: RegulatorParams, r, P_grid, x_grid):
    """THE ORACLE — boolean grid[i,j] = does (P_grid[j], x_grid[i]) converge to the healthy node.
    Rows = x, cols = P. Returns (mask, Ph); mask is all-False (dead) when no healthy FP exists."""
    h = healthy_fp(p, r)
    if h is None:
        return np.zeros((len(x_grid), len(P_grid)), bool), None
    Ph = h["P"]
    mask = np.zeros((len(x_grid), len(P_grid)), bool)
    for i, x0 in enumerate(x_grid):
        for j, P0 in enumerate(P_grid):
            mask[i, j] = converges_healthy(P0, x0, p, r, Ph)
    return mask, Ph


def basin_area(p: RegulatorParams, r, P_grid, x_grid):
    mask, _ = basin_map(p, r, P_grid, x_grid)
    return mask.mean()


def critical_bleed(p: RegulatorParams, r, iters=40):
    """Largest instantaneous pressure drop from the healthy node that still recovers — the basin
    boundary location along the bleed axis (measured by integration: an independent number, not
    the saddle). 0.0 when no healthy node exists."""
    h = healthy_fp(p, r)
    if h is None:
        return 0.0
    Ph, xh = h["P"], h["x"]
    lo, hi = 0.0, Ph                                   # bleed magnitude bracket
    if converges_healthy(Ph - hi, xh, p, r, Ph):       # survives even a total drop (shouldn't happen)
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if converges_healthy(Ph - mid, xh, p, r, Ph):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------- separatrix + the cheap viability margin ----------------

def separatrix(p: RegulatorParams, r, span=60.0, n=400):
    """The saddle's stable manifold (the envelope boundary), traced by integrating the loop
    BACKWARD in time from just off the saddle along its stable eigenvector. Returns (P, x) samples
    sorted by x, or None if no saddle exists."""
    s = saddle_fp(p, r)
    if s is None:
        return None
    J = jacobian((s["P"], s["x"]), p, r)
    w, V = np.linalg.eig(J)
    v = np.real(V[:, int(np.argmin(w.real))])          # stable eigenvector
    v = v / np.linalg.norm(v)
    pts = []
    for sgn in (+1.0, -1.0):
        s0 = np.array([s["P"], s["x"]]) + sgn * 1e-4 * v
        sol = solve_ivp(lambda t, st: rhs(st, p, r), [0.0, -span], s0,
                        rtol=1e-8, atol=1e-10, method="LSODA",
                        t_eval=np.linspace(0.0, -span, n))
        pts.append(sol.y.T)
    P = np.concatenate([pts[1][::-1, 0], pts[0][:, 0]])
    X = np.concatenate([pts[1][::-1, 1], pts[0][:, 1]])
    order = np.argsort(X)
    return np.column_stack([P[order], X[order]])


def make_viability_margin(p: RegulatorParams, r):
    """Build the CHEAP shippable predicate: a function margin(P, x) = signed normalized distance
    from the separatrix (>0 alive / healthy side, <0 dead / collapse side). Costs ONE backward
    integration to build (the separatrix) then a table lookup per query — vs the full basin sweep
    it is judged against. Returns (margin_fn, separatrix) or (None, None) when no saddle exists."""
    sep = separatrix(p, r)
    h = healthy_fp(p, r)
    if sep is None or h is None:
        return None, None
    P_sep, X_sep = sep[:, 0], sep[:, 1]
    scale = max(h["P"] - saddle_fp(p, r)["P"], 1e-6)

    def margin(P, x):
        P_boundary = np.interp(x, X_sep, P_sep)        # separatrix pressure at this tone
        return (P - P_boundary) / scale                # healthy side (higher P) is positive

    return margin, sep


# ---------------- self-check (regression guard) ----------------

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    p = RegulatorParams()

    # 1) bare pump has NO upper FP — the healthy state exists only via the regulator.
    Ps = np.linspace(0.05, 3.0, 4000)
    bare = pump(Ps, p) - p.gamma * Ps
    bare_upper = np.any(bare[:-1] * bare[1:] < 0)
    print(f"1) bare-pump upper FP exists: {bare_upper}  (want False — life requires regulation)")
    assert not bare_upper

    # 2) at full reserve: absorbing collapse + saddle + UNSATURATED stable healthy node.
    print("\n2) full-reserve fixed points:")
    for fp in fixed_points(p, p.r0):
        print(f"   P={fp['P']:.3f}  x={fp['x']:.3f}  {fp['kind']:11s}  eig.re={fp['eig'].real}")
    h = healthy_fp(p, p.r0); s = saddle_fp(p, p.r0); collapse = fixed_points(p, p.r0)[0]
    assert h is not None and s is not None
    assert collapse["P"] == 0.0 and collapse["kind"] == "stable_node"     # absorbing death
    assert not saturated(h["P"], p.r0, p)                                 # active control w/ headroom
    assert (s["eig"].real.max() > 0)                                      # saddle ⇒ positive-feedback cascade
    print(f"   healthy P={h['P']:.3f} (saturated={saturated(h['P'], p.r0, p)}), "
          f"saddle P={s['P']:.3f}, collapse absorbing ✓")

    # 3) reserve sweep: basin (critical bleed) contracts monotonically → 0 at a finite r_crit.
    rc = r_critical(p)
    print(f"\n3) saddle-node r_crit = {rc:.3f}")
    rs = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25]
    cb = [critical_bleed(p, r) for r in rs]
    print("   reserve     :", rs)
    print("   crit-bleed  :", np.round(cb, 3))
    assert all(cb[i] >= cb[i + 1] - 1e-3 for i in range(len(cb) - 1))     # monotone non-increasing
    assert healthy_fp(p, rc * 0.9) is None and healthy_fp(p, 1.0) is not None
    assert 0.05 < rc < 0.5

    # 4) the cheap viability margin agrees with the brute-force basin (the currency check).
    Pg = np.linspace(0.0, 2.4, 36); Xg = np.linspace(0.0, p.a_cap, 24)
    mask, Ph = basin_map(p, 1.0, Pg, Xg)
    margin, sep = make_viability_margin(p, 1.0)
    pred = np.array([[margin(P0, x0) > 0 for P0 in Pg] for x0 in Xg])
    agree = (pred == mask).mean()
    print(f"\n4) viability-margin vs brute-force basin: agreement = {agree:.3f} over {mask.size} states")
    assert agree > 0.97

    print("\nOK — regulator oracle: regulation-dependent homeostasis, reserve-shrinking viable")
    print("     envelope, absorbing positive-feedback death, and a calibrated viability margin.")
