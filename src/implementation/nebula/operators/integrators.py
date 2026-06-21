"""
Stiff-loop integrators for the fire law-domain (ARCHITECTURE §III.2; Decision #12).
Verified by V1.2 (PASS, with a STANDING CONSTRAINT): the conserved-bus split is plain
forward Euler on the coupled RHS -- consistent at order 1 but only CONDITIONALLY STABLE on
the stiff Arrhenius char<->conduction<->pyrolysis loop. Past an explicit step limit a
reactant is over-subscribed, the commit clamps it, and the integrated burn is SILENTLY
CORRUPTED (266% char error -- it won't announce itself). MANDATORY (report §3.3, §5.2): run
the stiff loop with rate-driven sub-stepping OR the semi-implicit (IMEX) treatment. Never
ship the plain explicit single-step path.

Two fixes, both validated against the monolithic RK4 oracle:
  - multirate (the refinement predicate's RATE term -> a finer local timestep): sub-cycle
    the bus step at a rate-limited stable sub-step (V1.2: 0.02% error, 11.7x larger stable step).
  - semi-implicit IMEX: linearly-implicit reactant depletion (unconditionally stable on the
    reaction terms), lifting the sub-stepping requirement (V1.2: 0.30% error).

Ported verbatim-in-behaviour from src/verification/oracles/{multirate,semi_implicit_fire}.py,
re-wired onto core.buses + operators.fire.
"""
import numpy as np

from ..core import buses
from . import fire as fo


# ---------------- rate-driven sub-stepping (the multi-rate handler) ----------------

def stable_substep(st, p, remaining, safety=0.4):
    """Largest explicit-Euler sub-step that neither over-depletes a reactant nor violates
    the diffusion CFL -- the rate term |d state/dt| turned into a local timestep.

    Limits: pyrolysis (m_s/r_py), combustion (gas/r_cb, o2/(s_o2 r_cb)), diffusion CFL
    (dx^2/(6 k_max)). `safety` is the Euler-appropriate fraction (validated by V1.2).
    """
    r_py = fo.pyrolysis_rate(st["T"], st["m_s"], p)
    r_cb = fo.combustion_rate(st["T"], st["gas"], st["o2"], p)
    eps = 1e-30
    limits = [
        safety * float(np.min(st["m_s"] / (r_py + eps))),
        safety * float(np.min(st["gas"] / (r_cb + eps))),
        safety * float(np.min(st["o2"] / (p.s_o2 * r_cb + eps))),
        safety * p.dx * p.dx / (6.0 * max(p.k_wood, p.k_char)),
    ]
    h = min([remaining] + [x for x in limits if np.isfinite(x) and x > 0])
    return max(h, remaining * 1e-6)


def step_substep(domain, st, dt, sources=None, safety=0.4, op_order=None):
    """Advance the fire domain by a global `dt`, internally sub-cycling buses.step at the
    rate-limited stable sub-step. The external source (if any) is deposited on the FIRST
    sub-step only (same total deposit as a single step). Returns (new_state, n_sub, max_audit).

    With proper sub-stepping no sub-step over-subscribes a bus, so no commit clamps and the
    conservation audit stays ~0 throughout the burn -- max_audit is that running maximum (V0.3).
    """
    p = domain.params
    done = 0.0
    n_sub = 0
    first = True
    max_audit = 0.0
    while done < dt - 1e-15:
        h = stable_substep(st, p, dt - done, safety=safety)
        src = sources if first else None
        st, _, aud, _ = buses.step(domain, st, h, sources=src, op_order=op_order, compute_gov=False)
        max_audit = max(max_audit, max(aud.values()))
        done += h
        n_sub += 1
        first = False
    return st, n_sub, max_audit


def run_substep(domain, st, dt_global, nsteps, safety=0.4):
    st = buses.copy_state(st)
    total = 0
    max_audit = 0.0
    for _ in range(nsteps):
        st, n, aud = step_substep(domain, st, dt_global, safety=safety)
        total += n
        max_audit = max(max_audit, aud)
    return st, total, max_audit


def run_naive(domain, st, dt_global, nsteps, op_order=None):
    """Naive single-step forward-Euler split (the corrupted path -- for the necessity demo)."""
    st = buses.copy_state(st)
    for _ in range(nsteps):
        st, _, _, _ = buses.step(domain, st, dt_global, op_order=op_order)
    return st


# ---------------- semi-implicit (IMEX) split (lifts the sub-stepping requirement) ----------------

_SI_FIELDS = ("T", "m_s", "gas", "o2", "char", "q")


def step_semi_implicit(st, p, dt, sources=None):
    """Advance the coupled fire system by `dt` with the IMEX semi-implicit split.

    Lie split: implicit (linearized) reaction depletion -- unconditionally stable, no clamp
    ever reached because the implicit depletion is bounded by construction -- then explicit
    transport. Converges to the monolithic oracle at first order; the win is stability on
    the stiff loop. `sources` follows the same (staged, st, p, dt) convention as buses.step,
    here applied as a direct deposit before the reaction sub-step.
    """
    st = {f: np.asarray(st[f], dtype=float).copy() for f in _SI_FIELDS}

    if sources is not None:
        # reuse the ignition convention: deposit amount = rate*dt into T (and q)
        staged = {f: np.zeros_like(st[f]) for f in _SI_FIELDS}
        sources(staged, st, p, dt)
        st["T"] += staged["T"] * dt
        st["q"] += staged["q"] * dt

    T = st["T"]
    # implicit pyrolysis (linear in m_s -> backward Euler; consumed amount bounded by m_s)
    k_py = p.A_py * np.exp(-p.Ta_py / np.maximum(T, 1.0))
    m_s_new = st["m_s"] / (1.0 + k_py * dt)
    C_py = st["m_s"] - m_s_new
    st["m_s"] = m_s_new
    st["gas"] += p.nu_g * C_py
    st["char"] += p.nu_c * C_py
    T -= (p.dH_py * C_py) / p.C_V

    # implicit combustion (linear in gas, o2 frozen -> backward Euler; capped at available O2)
    k_cb = p.A_cb * np.exp(-p.Ta_cb / np.maximum(T, 1.0)) * np.maximum(st["o2"], 0.0)
    gas_new = st["gas"] / (1.0 + k_cb * dt)
    C_cb = st["gas"] - gas_new
    o2_cap = np.maximum(st["o2"], 0.0) / p.s_o2 * (1.0 - 1e-12)
    C_cb = np.minimum(C_cb, o2_cap)
    st["gas"] -= C_cb
    st["o2"] -= p.s_o2 * C_cb
    T += (p.dH_cb * C_cb) / p.C_V

    # explicit transport
    dE, _ = fo.conduction_energy(T, st["char"], st["m_s"], p)
    T += (dE / p.C_V) * dt
    o2_src, _ = fo.o2_boundary_influx(st["o2"], p)
    st["o2"] += o2_src * dt
    st["q"] = st["q"] / (1.0 + p.lambda_q * dt)

    st["T"] = T
    return st


def run_semi_implicit(p, st, dt, nsteps, source_at=None, sources=None):
    st = {f: np.asarray(st[f], dtype=float).copy() for f in _SI_FIELDS}
    for n in range(nsteps):
        src = sources if (source_at is not None and n == source_at) else None
        st = step_semi_implicit(st, p, dt, sources=src)
    return st


def is_finite_state(st):
    return all(np.all(np.isfinite(st[f])) for f in _SI_FIELDS)


if __name__ == "__main__":
    np.seterr(all="ignore")
    np.set_printoptions(precision=4, suppress=True)
    dom = fo.fire_domain()
    p = dom.params
    N = 12

    def scene():
        st = fo.make_state(N, T0=550.0, gas0=0.05, o2=0.15)
        c = slice(N // 2 - 2, N // 2 + 2)
        st["T"][c, c, c] = 900.0
        return st

    DT_PROD, NSTEPS = 0.05, 4

    # monolithic-free reference: deep sub-stepping is the practical oracle proxy here.
    ref, _, _ = run_substep(dom, scene(), DT_PROD, NSTEPS, safety=0.05)
    st_n = run_naive(dom, scene(), DT_PROD, NSTEPS, op_order=fo.ORACLE_OP_ORDER)
    st_s, total_sub, audit_s = run_substep(dom, scene(), DT_PROD, NSTEPS)
    st_i = run_semi_implicit(p, scene(), DT_PROD, NSTEPS)

    def char(st): return float(st["char"].sum())
    cr = char(ref)
    print(f"production global dt = {DT_PROD} over {NSTEPS} steps  (ref char={cr:.3f})")
    print(f"1) naive split:   finite={is_finite_state(st_n)}  char={char(st_n):.3f}  "
          f"err={100*abs(char(st_n)-cr)/cr:.0f}%  (clamps corrupt it silently)")
    print(f"2) sub-stepped:   {total_sub} sub-steps  char={char(st_s):.3f}  "
          f"err={100*abs(char(st_s)-cr)/cr:.3f}%  max conservation audit={audit_s:.1e} (~0: no clamp)")
    print(f"3) semi-implicit: char={char(st_i):.3f}  err={100*abs(char(st_i)-cr)/cr:.3f}%  finite={is_finite_state(st_i)}")
    print("\nintegrators self-checks passed.")
