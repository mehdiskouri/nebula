"""
Rate-refinement local sub-stepping — the multi-rate handler for the stiff fire loop
(ARCHITECTURE §III.2; Decision #12). The MECHANISM UNDER TEST for V1.2.

bus_runtime.step_split stages all operator contributions then commits once
(new = old + Sum(contribs)*dt), so in-distribution it is FORWARD EULER on
fire_operators.coupled_rhs: first-order accurate but only CONDITIONALLY STABLE on the
stiff Arrhenius combustion/pyrolysis terms and the char<->conduction<->pyrolysis loop.
Past an explicit step limit a naive split overshoots -> clamps -> diverges.

This module supplies the fix the architecture names: the refinement predicate's RATE
term |d state/dt| sets a finer LOCAL timestep. step_split_substep sub-cycles step_split
with a rate-limited sub-step so the production global step stays stable. It is the
explicit-Euler analog of monolithic_fire._stable_h (which does the same for the RK4
oracle). Kept SEPARATE so the V0.3/V1.1-critical bus_runtime/fire_operators paths stay
untouched (regression-safe).

SCOPE: domain-GLOBAL adaptive sub-stepping (one sub-step for the active domain, sized by
the min over cells) -- exactly what the oracle does. True per-cell local timestepping
(LTS) is a Phase-0 engineering extension, not built here; global sub-stepping is
sufficient for the stability claim V1.2 makes.
"""
import numpy as np

import fire_operators as fo
import bus_runtime as br


def stable_substep(st, p, remaining, safety=0.4):
    """Largest explicit-Euler sub-step that neither over-depletes a reactant nor violates
    the diffusion CFL -- the rate term |d state/dt| turned into a local timestep.

    Limits (the stiff timescales of the loop):
      - pyrolysis consuming solid:   m_s / r_py
      - combustion consuming gas:    gas / r_cb
      - combustion consuming O2:     o2  / (s_o2 * r_cb)
      - explicit 3-D diffusion CFL:  dx^2 / (6 * k_max)
    `safety` is the Euler-appropriate fraction (tighter than the RK4 oracle's 0.2); its
    adequacy is validated empirically by the V1.2 stability-boundary result.
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


def step_split_substep(st, p, dt, lightning=None, safety=0.4):
    """Advance the split runtime by a global `dt`, internally sub-cycling step_split at the
    rate-limited stable sub-step. Returns (new_state, n_sub).

    The lightning impulse (if any) is deposited on the FIRST sub-step only (same total
    deposit as the single-step runtime), then the burn integrates stably.
    """
    done = 0.0
    n_sub = 0
    first = True
    while done < dt - 1e-15:
        h = stable_substep(st, p, dt - done, safety=safety)
        lg = lightning if first else None
        st, _, _, _ = br.step_split(st, p, h, lightning=lg)
        done += h
        n_sub += 1
        first = False
    return st, n_sub


def run_substep(st, p, dt_global, nsteps, safety=0.4):
    """Convenience: integrate nsteps of global dt with sub-stepping. Returns (state, total_sub)."""
    st = br.copy_state(st)
    total = 0
    for _ in range(nsteps):
        st, n = step_split_substep(st, p, dt_global, safety=safety)
        total += n
    return st, total


def run_naive(st, p, dt_global, nsteps):
    """Convenience: integrate nsteps of naive single-step forward-Euler split."""
    st = br.copy_state(st)
    for _ in range(nsteps):
        st, _, _, _ = br.step_split(st, p, dt_global)
    return st


def is_finite_state(st):
    return all(np.all(np.isfinite(st[f])) for f in br.FIELDS)


if __name__ == "__main__":
    import monolithic_fire as mf
    np.seterr(all="ignore")
    np.set_printoptions(precision=4, suppress=True)
    p = fo.FireParams()
    N = 12

    # a BOUNDED stiff burn (peak T self-limits as core O2 depletes -> affordable to
    # sub-step / integrate, yet stiff enough that naive Euler over the production step is
    # corrupted by the depletion clamps).
    def scene():
        st = br.make_state(N, T0=550.0, gas0=0.05, o2=0.15)
        c = slice(N // 2 - 2, N // 2 + 2)
        st["T"][c, c, c] = 900.0
        return st

    DT_PROD = 0.05            # a production global step above the explicit stability limit
    NSTEPS = 4
    mass0 = float(scene()["m_s"].sum())

    st_o = scene()            # monolithic oracle
    for _ in range(NSTEPS):
        st_o, _ = mf.step_monolithic(st_o, p, DT_PROD)
    st_n = run_naive(scene(), p, DT_PROD, NSTEPS)                  # naive single-step Euler
    st_s, total_sub = run_substep(scene(), p, DT_PROD, NSTEPS)     # rate-driven sub-stepping

    def char(st): return float(st["char"].sum())
    cr, cn, cs = char(st_o), char(st_n), char(st_s)
    en = 100 * abs(cn - cr) / cr
    es = 100 * abs(cs - cr) / cr
    print(f"production global dt = {DT_PROD} over {NSTEPS} steps  (oracle Tmax={st_o['T'].max():.0f})")
    print(f"1) naive split:  finite={is_finite_state(st_n)}  total char={cn:.3f}  "
          f"err vs oracle = {en:.0f}%  (clamps keep it finite but the char outcome is corrupted)")
    print(f"2) sub-stepped:  {total_sub} sub-steps  total char={cs:.3f}  err vs oracle = {es:.2f}%  "
          f"(oracle char={cr:.3f})")
