"""
Transition cascade — the value/category half of the operator write model
(ARCHITECTURE.md §III.3, Part II; Decision #12). The MECHANISM UNDER TEST for V1.1.

The conserved-bus runtime has two write modes (bus_runtime.py):
  - contributions: ADDITIVE staging into conserved buses -> order-independent (verified
    in V1.1 Part A; this is just the REDUCE step, whose determinism V0.5 nailed down);
  - transitions:   a change to a variable's VALUE or CATEGORY -> NOT commutative. When
    several operators want to set the SAME variable on the SAME cell, "last writer wins"
    depends on evaluation order. Determinism is recovered ONLY by a DECLARED CASCADE
    PRIORITY (CSS-specificity-like, Part II): the highest-priority applicable transition
    wins, independent of order.

bus_runtime applies exactly one transition (op_char_weakening) and notes "(cascade-
priority; here a single transition operator)". This module is the GENERALIZATION of
that one line into a real declared-priority cascade, kept SEPARATE so the V0.3-critical
bus_runtime/fire_operators paths stay untouched (regression-safe).

The protocol's contested case is "drying / pyrolysis / smoldering on one variable". Two
faithful variants are built:
  - CATEGORICAL PHASE (core theorem-check): a per-cell phase enum the three transitions
    compete to set;
  - MASS-ROUTING (generalization): the same three compete to set a per-cell `fate`
    selector deciding where the cell's mass goes this step (vapor / gas+char / ash) --
    resolved by the cascade, then applied ONCE (it is a value-transition, never an
    additive contribution, so the contribution/transition line stays clean).

The resolvers are scenario-agnostic: a transition is a (mask, target_value, priority),
and both variants reduce to "resolve competing integer targets on a shared field".
"""
from dataclasses import dataclass, field

import numpy as np

# ---------- categorical phase enum (variant 1) ----------
WET, DRY, PYROLYZING, SMOLDERING, CHAR = 0, 1, 2, 3, 4
PHASE_NAMES = {WET: "WET", DRY: "DRY", PYROLYZING: "PYROLYZING",
               SMOLDERING: "SMOLDERING", CHAR: "CHAR"}

# ---------- mass-fate selector enum (variant 2) ----------
FATE_NONE, FATE_VAPOR, FATE_GASCHAR, FATE_ASH = -1, 0, 1, 2
FATE_NAMES = {FATE_NONE: "none", FATE_VAPOR: "vapor",
              FATE_GASCHAR: "gas+char", FATE_ASH: "ash"}


@dataclass
class TransitionParams:
    # firing thresholds (a contested cell trips several at once)
    moisture_min: float = 0.05    # drying fires where there is moisture to drive off ...
    T_dry: float = 350.0          # ... and it is warm enough
    T_pyro: float = 600.0         # pyrolysis fires where hot ...
    m_s_min: float = 0.05         # ... and solid fuel remains
    char_min: float = 0.05        # smoldering fires where char has formed ...
    o2_min: float = 0.02          # ... and oxygen reaches it ...
    T_smolder: float = 500.0      # ... and it is at least warm
    # DECLARED CASCADE PRIORITIES (the load-bearing declaration; strictly distinct so the
    # winner is unambiguous). Smoldering char outranks pyrolysis outranks drying.
    prio_drying: int = 1
    prio_pyrolysis: int = 2
    prio_smolder: int = 3


# ---------- the three competing transition operators ----------
# Each READS shared state and emits a PROPOSAL: a boolean fire-mask, the integer target
# it would set, and its declared priority. It never mutates state and never looks at the
# other operators -- exactly the operator discipline, one write-mode over.

def _fire_drying(state, p):
    return (state["moisture"] >= p.moisture_min) & (state["T"] >= p.T_dry)


def _fire_pyrolysis(state, p):
    return (state["T"] >= p.T_pyro) & (state["m_s"] >= p.m_s_min)


def _fire_smolder(state, p):
    return ((state["char"] >= p.char_min) & (state["o2"] >= p.o2_min)
            & (state["T"] >= p.T_smolder))


def phase_proposals(state, p):
    """The three transitions competing to set the categorical `phase` field."""
    return [
        {"name": "drying",    "mask": _fire_drying(state, p),
         "target": DRY,        "priority": p.prio_drying},
        {"name": "pyrolysis", "mask": _fire_pyrolysis(state, p),
         "target": PYROLYZING, "priority": p.prio_pyrolysis},
        {"name": "smolder",   "mask": _fire_smolder(state, p),
         "target": SMOLDERING, "priority": p.prio_smolder},
    ]


def fate_proposals(state, p):
    """The same three competing to set the per-cell mass `fate` selector (variant 2)."""
    return [
        {"name": "drying",    "mask": _fire_drying(state, p),
         "target": FATE_VAPOR,   "priority": p.prio_drying},
        {"name": "pyrolysis", "mask": _fire_pyrolysis(state, p),
         "target": FATE_GASCHAR, "priority": p.prio_pyrolysis},
        {"name": "smolder",   "mask": _fire_smolder(state, p),
         "target": FATE_ASH,     "priority": p.prio_smolder},
    ]


# ---------- the two resolvers (scenario-agnostic) ----------

def apply_transitions_ordered(field0, proposals, order):
    """NO cascade: apply transitions in evaluation `order`, LAST WRITER WINS.

    Where two operators' masks overlap, the one applied later overwrites -> the committed
    value depends on `order`. This is the ambiguity the cascade exists to remove.
    """
    out = np.array(field0, copy=True)
    by_name = {pr["name"]: pr for pr in proposals}
    for nm in order:
        pr = by_name[nm]
        out[pr["mask"]] = pr["target"]
    return out


def apply_transitions_cascade(field0, proposals):
    """DECLARED CASCADE: per cell, the applicable proposal of HIGHEST priority wins.

    Order-independent by construction: a strict `priority > best` test with distinct
    declared priorities selects the same winner regardless of the order proposals are
    visited. This is bus_runtime's single transition line, generalized.
    """
    out = np.array(field0, copy=True)
    best = np.full(out.shape, -2**31, dtype=np.int64)   # priority of current winner
    for pr in proposals:
        take = pr["mask"] & (pr["priority"] > best)
        out[take] = pr["target"]
        best[take] = pr["priority"]
    return out


# ---------- mass-routing application (value-transition, applied ONCE) ----------

def apply_routing(fate, mass):
    """Route each cell's `mass` to product pools per its RESOLVED fate (applied once).

    Returns {vapor, gas, char, ash}. This is a value-transition consequence, NOT an
    additive bus contribution -- the cascade decides the single fate first, then mass
    moves deterministically.
    """
    vapor = np.where(fate == FATE_VAPOR, mass, 0.0)
    gas = np.where(fate == FATE_GASCHAR, 0.7 * mass, 0.0)
    char = np.where(fate == FATE_GASCHAR, 0.3 * mass, 0.0)
    ash = np.where(fate == FATE_ASH, mass, 0.0)
    return {"vapor": vapor, "gas": gas, "char": char, "ash": ash}


# ---------- contested scenario ----------

def make_scenario(N=16, seed=0):
    """A field with a deliberately CONTESTED region (cells tripping >=2 firing masks),
    plus singly-firing and inert cells -- so order-dependence is real, not contrived.

    Returns (state, phase0, fate0). `phase0` starts WET everywhere; `fate0` unset.
    """
    rng = np.random.default_rng(seed)
    T = np.full((N, N, N), 300.0)
    moisture = np.zeros((N, N, N))
    m_s = np.ones((N, N, N))
    char = np.zeros((N, N, N))
    o2 = np.full((N, N, N), 0.23)

    half = N // 2
    # CONTESTED core: hot + still moist + already charring + oxygen present
    # -> drying, pyrolysis AND smolder all fire on the same cells.
    c = slice(half - 3, half + 3)
    T[c, c, c] = 700.0
    moisture[c, c, c] = 0.3
    char[c, c, c] = 0.2

    # a drying-only shell (warm, wet, cool enough to not pyrolyze, no char)
    s = slice(2, 5)
    T[s, :, :] = 400.0
    moisture[s, :, :] = 0.4

    # a pyrolysis-only patch (hot, dry, no char, no/low o2 so smolder won't fire)
    q = slice(N - 5, N - 2)
    T[:, q, :] = 650.0
    o2[:, q, :] = 0.0

    # a little deterministic jitter so masks are not perfectly axis-aligned slabs
    T += rng.normal(0.0, 2.0, T.shape)

    state = {"T": T, "moisture": moisture, "m_s": m_s, "char": char, "o2": o2}
    phase0 = np.full((N, N, N), WET, dtype=np.int64)
    fate0 = np.full((N, N, N), FATE_NONE, dtype=np.int64)
    return state, phase0, fate0


def contested_mask(proposals):
    """Cells where >=2 transitions fire simultaneously (the order-sensitive set)."""
    counts = np.zeros(proposals[0]["mask"].shape, dtype=np.int64)
    for pr in proposals:
        counts += pr["mask"].astype(np.int64)
    return counts >= 2


if __name__ == "__main__":
    import itertools

    p = TransitionParams()
    state, phase0, fate0 = make_scenario(N=16, seed=0)
    props = phase_proposals(state, p)

    cm = contested_mask(props)
    print(f"contested cells (>=2 transitions fire): {int(cm.sum())} of {phase0.size}")
    for pr in props:
        print(f"   {pr['name']:10s} fires on {int(pr['mask'].sum()):5d} cells "
              f"(target={PHASE_NAMES[pr['target']]}, priority={pr['priority']})")

    orders = list(itertools.permutations(["drying", "pyrolysis", "smolder"]))

    # the reference winner per cell: target of the highest-priority FIRING proposal
    # (or the initial value where none fire) -- what the cascade must reproduce.
    def reference_winner(field0, proposals):
        out = np.array(field0, copy=True)
        best = np.full(out.shape, -2**31, dtype=np.int64)
        for pr in sorted(proposals, key=lambda x: x["priority"]):
            take = pr["mask"] & (pr["priority"] > best)
            out[take] = pr["target"]; best[take] = pr["priority"]
        return out

    # 1) NO cascade: distinct committed phase-maps across orderings (ambiguity).
    ordered_results = [apply_transitions_ordered(phase0, props, o) for o in orders]
    distinct_ordered = {r.tobytes() for r in ordered_results}
    print(f"\n1) NO cascade: {len(distinct_ordered)} distinct phase-maps over "
          f"{len(orders)} orderings  (>1 => order-dependent)")

    # 2) Cascade: single committed phase-map regardless of ordering (determinism), and
    #    it equals the highest-priority-firing reference everywhere.
    casc = apply_transitions_cascade(phase0, props)
    ref = reference_winner(phase0, props)
    casc_correct = bool(np.array_equal(casc, ref))
    core = (props[0]["mask"] & props[1]["mask"] & props[2]["mask"])   # all three fire
    print(f"2) cascade: equals highest-priority-firing reference everywhere: {casc_correct}; "
          f"cells where all 3 fire resolve to {PHASE_NAMES[SMOLDERING]}: "
          f"{bool(np.all(casc[core] == SMOLDERING))} ({int(core.sum())} cells)")

    # 3) mass-routing variant: same contrast on the `fate` selector.
    fprops = fate_proposals(state, p)
    f_ordered = {apply_transitions_ordered(fate0, fprops, o).tobytes() for o in orders}
    f_casc = apply_transitions_cascade(fate0, fprops)
    f_correct = bool(np.array_equal(f_casc, reference_winner(fate0, fprops)))
    print(f"3) mass-routing: NO cascade -> {len(f_ordered)} distinct fate-maps; "
          f"cascade equals reference: {f_correct}; cells where all 3 fire -> "
          f"{FATE_NAMES[FATE_ASH]}: {bool(np.all(f_casc[core] == FATE_ASH))}")
