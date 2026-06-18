"""
Conserved-bus operator runtime (ARCHITECTURE.md §III.3, Part IV; Decisions #12, #14).

The universal compute pattern, made real: GATHER -> STAGE into buses -> REDUCE ->
COMMIT. Operators never call each other; they only stage additive contributions
(order-independent) and value transitions (cascade-priority). The runtime reduces per
bus, commits with non-negativity clamps (the realistic limiter), applies transitions,
and AUDITS every conserved bus.

Two monitors are produced each step (V0.3 reports both):
  - conservation audit: per-bus |committed Delta(total) - ledgered net transfer|,
    normalized by bus throughput. ~0 in-distribution; spikes when a stiff runaway makes
    operators over-subscribe a shared bus and the commit must clamp (the composite-OOD
    symptom that per-operator envelopes miss).
  - governing-equation residual: ||(U_new - U_old)/dt - F_coupled(U_new)|| (relative) —
    how far the split-committed state is from satisfying the fully-coupled implicit
    equations (the proxy-error reading).

Shared by V0.3, V1.1 (composition order-independence), V1.2 (split stability).
"""
import numpy as np

import fire_operators as fo

MASS_FIELDS = ("m_s", "gas", "char", "o2")     # clamped >= 0 on commit
FIELDS = ("T", "m_s", "gas", "char", "o2", "q")


def make_state(N, T0=320.0, m_s0=1.0, gas0=0.02, o2=0.23, char0=0.0):
    return {
        "T":   np.full((N, N, N), float(T0)),
        "m_s": np.full((N, N, N), float(m_s0)),
        "gas": np.full((N, N, N), float(gas0)),
        "o2":  np.full((N, N, N), float(o2)),
        "char": np.full((N, N, N), float(char0)),
        "q":   np.zeros((N, N, N)),
    }


def copy_state(st):
    return {k: v.copy() for k, v in st.items()}


def _bus_totals(st):
    return {
        "energy": float(st["T"].sum()),                                  # C_V folded (C_V*T); C_V=1
        "mass":   float(st["m_s"].sum() + st["gas"].sum() + st["char"].sum() + st["o2"].sum()),
        "o2":     float(st["o2"].sum()),
        "charge": float(st["q"].sum()),
    }


def step_split(st, p, dt, lightning=None, op_order=None):
    """One GATHER->STAGE->REDUCE->COMMIT step of the split runtime.

    lightning: optional dict {"mask": bool array, "energy": float, "charge": float} —
    a composite heat+charge impulse deposited this step (ledgered as an external source).
    op_order: optional iterable of contribution-op names (to test order-independence).

    Returns (new_state, ledger, audit, gov_residual).
    """
    old = copy_state(st)
    names = list(op_order) if op_order is not None else list(fo.CONTRIBUTION_OPS)

    # GATHER + STAGE: every operator stages additive contributions (rates) into buses.
    staged = {f: np.zeros_like(st[f]) for f in FIELDS}
    ledger_rate = {}
    for nm in names:
        deltas, led = fo.CONTRIBUTION_OPS[nm](st, p)
        for f, d in deltas.items():                 # REDUCE: additive, order-independent
            staged[f] += d
        for k, v in led.items():
            ledger_rate[k] = ledger_rate.get(k, 0.0) + v
    # operator ledger entries are RATES -> integrate over the step to amounts
    ledger = {k: v * dt for k, v in ledger_rate.items()}

    # external composite source (lightning heat + charge), ledgered as amounts
    e_light = q_light = 0.0
    if lightning is not None:
        m = lightning["mask"]
        ce = lightning.get("energy", 0.0); cq = lightning.get("charge", 0.0)
        ncell = max(int(m.sum()), 1)
        staged["T"][m] += (ce / p.C_V) / ncell / dt        # rate so that *dt deposits ce
        staged["q"][m] += cq / ncell / dt
        e_light, q_light = ce, cq
    ledger["lightning_energy"] = e_light
    ledger["lightning_charge"] = q_light

    # intended (pre-clamp) update, then COMMIT with non-negativity clamps
    new = {}
    for f in FIELDS:
        intended = old[f] + staged[f] * dt
        if f in MASS_FIELDS or f == "q":
            new[f] = np.maximum(intended, 0.0)          # clamp (the limiter)
        else:
            new[f] = intended

    # TRANSITIONS (cascade-priority; here a single transition operator)
    new["S"] = fo.op_char_weakening(new, p)["S"]

    audit = _audit(old, new, staged, ledger, p, dt)
    gov = _governing_residual(old, new, p, dt)
    return new, ledger, audit, gov


def _audit(old, new, staged, ledger, p, dt):
    """Per-bus relative conservation residual.

    residual = |committed Delta(total) - ledgered/intended net transfer| / gross flow,
    where gross flow = total absolute staged movement that step (robust: stays finite
    during an active burn, so a near-zero net imbalance reads ~0 rather than 0/0).
    """
    res = {}

    # ENERGY: Delta(sum T*C_V) vs combustion - pyrolysis + boundary loss + lightning
    dE = (new["T"].sum() - old["T"].sum()) * p.C_V
    led_E = (ledger.get("combustion_exo", 0.0) + ledger.get("pyrolysis_endo", 0.0)
             + ledger.get("boundary_heat_loss", 0.0) + ledger.get("lightning_energy", 0.0))
    tp_E = float(np.abs(staged["T"] * dt).sum()) * p.C_V + abs(ledger.get("lightning_energy", 0.0))
    res["energy"] = abs(dE - led_E) / (tp_E + 1e-30)

    # MASS (m_s+gas+char+o2): only boundary transfers change the total (o2 in, exhaust out)
    dM = ((new["m_s"] + new["gas"] + new["char"] + new["o2"]).sum()
          - (old["m_s"] + old["gas"] + old["char"] + old["o2"]).sum())
    led_M = ledger.get("o2_influx", 0.0) - ledger.get("exhaust_vent", 0.0)
    tp_M = float(sum(np.abs(staged[f] * dt).sum() for f in MASS_FIELDS))
    res["mass"] = abs(dM - led_M) / (tp_M + 1e-30)

    # O2: committed Delta vs intended (influx - consumption); clamp surfaces here
    do2 = new["o2"].sum() - old["o2"].sum()
    intended_do2 = (staged["o2"] * dt).sum()
    tp_o2 = float(np.abs(staged["o2"] * dt).sum())
    res["o2"] = abs(do2 - intended_do2) / (tp_o2 + 1e-30)

    # CHARGE: committed Delta vs intended (lightning - dissipation)
    dq = new["q"].sum() - old["q"].sum()
    intended_dq = (staged["q"] * dt).sum()
    tp_q = float(np.abs(staged["q"] * dt).sum()) + abs(ledger.get("lightning_charge", 0.0))
    res["charge"] = abs(dq - intended_dq) / (tp_q + 1e-30)

    return res


def _governing_residual(old, new, p, dt):
    """Relative residual of the committed state vs the fully-coupled implicit equation."""
    F = fo.coupled_rhs(new, p)
    num = den = 0.0
    for f in FIELDS:
        rate = (new[f] - old[f]) / dt
        num += float(np.sum((rate - F[f]) ** 2))
        den += float(np.sum(rate ** 2) + np.sum(F[f] ** 2))
    return float(np.sqrt(num / (den + 1e-30)))


def max_audit(audit):
    return max(audit.values())


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    p = fo.FireParams()
    N = 8
    st = make_state(N)

    # 1) order-independence of additive contributions (commit identical for any op order).
    import itertools
    base, _, _, _ = step_split(st, p, dt=1e-4)
    perms = list(itertools.permutations(list(fo.CONTRIBUTION_OPS)))[:6]
    worst = 0.0
    for order in perms:
        alt, _, _, _ = step_split(st, p, dt=1e-4, op_order=order)
        worst = max(worst, max(float(np.abs(alt[f] - base[f]).max()) for f in FIELDS))
    print(f"1) contribution order-independence: max state divergence over orders = {worst:.2e}")

    # 2) in-distribution single step: all bus residuals ~0 (no clamp at gentle rates).
    _, led, audit, gov = step_split(st, p, dt=1e-4)
    print("2) in-distribution audit:", {k: f"{v:.1e}" for k, v in audit.items()},
          f" gov={gov:.2e}")

    # 3) contention: a hot, fuel-rich, O2-poor cell over one big step -> clamp -> spike.
    st2 = make_state(N, T0=1500.0, gas0=5.0, o2=1e-3)
    _, led2, audit2, gov2 = step_split(st2, p, dt=1e-1)
    print("3) contention audit:", {k: f"{v:.1e}" for k, v in audit2.items()},
          f" gov={gov2:.2e}  (mass/o2 should spike)")
