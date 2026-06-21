"""
Operator declaration schema (ARCHITECTURE §III.3; Decisions #12, #13).

An operator is a constitutive / transfer law. The load-bearing discipline (verified by
V0.3 / V1.1):

  > Operators never call each other and never mutate state directly. They only read
  > shared state, and they only write by staging contributions into conserved-quantity
  > buses that the runtime reduces and audits.

Two write modes, judged differently:
  - `contribute`  : ADDITIVE rate contributions into conserved buses -> order-independent
                    -> phenomena compose freely (V1.1: 120 orderings bit-identical).
  - `transition`  : a value/category change on a shared variable -> resolved by declared
                    cascade `priority`, NEVER by evaluation order (load-bearing for determinism).

`Field` declares which conserved `bus` a state field belongs to and the realistic commit
limiter (clamp). `Operator` packages the §III.3 declaration. The learned-tier metadata
(envelope/fallback/tiers) is carried but unused by the Phase-0 analytic path.
"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class Field:
    """A per-node state field of a law-domain.

    bus        : the conserved bus this field's changes are audited against (None = a
                 diagnostic/unconserved field, e.g. a derived strength scalar).
    clamp_min/ : the realistic non-negativity / saturation limiter applied on COMMIT
    clamp_max    (e.g. mass >= 0). When a stiff runaway over-subscribes a shared bus the
                 commit clamps, and that clamp is exactly the composite-OOD symptom the
                 conservation audit catches (V0.3).
    """
    name: str
    bus: Optional[str] = None
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None


@dataclass(frozen=True)
class Operator:
    """A declared constitutive/transfer operator (the §III.3 schema).

    name       : unique operator name (also the canonical reduction order key).
    binds      : the hyperedge type tag it operates on (e.g. 'lawdomain', 'constraint').
    reads      : the gather set (state field names) -- documentation of the read footprint.
    contribute : (state, params) -> (deltas {field: rate}, ledger {name: scalar}). ADDITIVE.
    transition : (state, params) -> {field: new_value}. Resolved by `priority` (cascade).
    priority   : higher wins when two transitions target the same field.
    timescale  : (state, params) -> array of characteristic rates, for multi-rate sub-stepping.
    envelope   : (state, params) -> bool array, the per-operator validity box (V0.3 shows it is
                 necessary-but-insufficient; the conservation audit is the primary monitor).
    fallback   : 'refine' | 'drop-tier' | 'both' (learned-tier metadata; unused in Phase 0).
    """
    name: str
    binds: str = "lawdomain"
    reads: tuple = ()
    contribute: Optional[Callable] = None
    transition: Optional[Callable] = None
    priority: int = 0
    timescale: Optional[Callable] = None
    envelope: Optional[Callable] = None
    fallback: str = "refine"

    def __post_init__(self):
        if self.contribute is None and self.transition is None:
            raise ValueError(f"operator {self.name!r} declares neither contribute nor transition")
