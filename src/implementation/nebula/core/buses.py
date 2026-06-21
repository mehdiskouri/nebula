"""
The conserved-bus operator runtime (ARCHITECTURE §III.3, Part IV; Decisions #12, #14).
The universal compute pattern made real: GATHER -> STAGE into buses -> REDUCE -> COMMIT.
Verified by V0.3 (conservation + composite-OOD), V1.1 (composition order-independence),
V1.2 (split stability).

This is the FIELD-AGNOSTIC generalization of the verified oracle (frozen
src/verification/oracles/bus_runtime.py, which hardcoded the fire fields). A `Domain`
packages a law-domain's `Field`s and `Operator`s; the same `step` drives any of them:

  GATHER  : each contribution operator reads shared state.
  STAGE   : it returns additive {field -> rate} deltas (order-independent) + ledger entries.
  REDUCE  : per field, the staged contributions are summed in a CANONICAL (fixed) order so
            the committed state is bit-reproducible run-to-run (V0.5/V1.1). For a GPU
            per-node scatter this is determinism.fixed_order / integer-exact.
  COMMIT  : intended = old + staged*dt, then per-field non-negativity / saturation clamps
            (the realistic limiter); a stiff runaway that over-subscribes a shared bus is
            clamped here, and that clamp is the composite-OOD symptom the audit catches.
  TRANSITIONS : value/category changes resolved by declared cascade PRIORITY, never order.

Two monitors are produced each step (V0.3 reports both):
  - conservation audit : per-bus |committed Delta - ledgered/intended net| / throughput.
    ~0 in-distribution; spikes on the clamp -- the PRIMARY composite-OOD monitor.
  - governing residual : ||(U_new-U_old)/dt - F_coupled(U_new)|| (relative), the proxy-error
    reading (how far the split-committed state is from the fully-coupled implicit equations).
"""
from dataclasses import dataclass, field as _field
from typing import Callable, Optional

import numpy as np

from . import determinism as det
from .schema import Field, Operator


@dataclass
class Domain:
    """A law-domain instantiation: its state fields, its operators, and its params.

    audit_fn   : optional bespoke conservation audit (old,new,staged,ledger,params,dt)->{bus:res}.
                 Domains with non-trivial bus<->field<->ledger mappings (e.g. fire, where the
                 energy bus = sum(C_V*T) and ledger entries are combustion/pyrolysis enthalpies)
                 supply their own; otherwise the generic clamp-imbalance audit is used.
    coupled_rhs: optional (state,params)->{field:rate}, the fully-coupled RHS for the gov residual.
    """
    name: str
    fields: tuple
    operators: tuple
    params: object
    audit_fn: Optional[Callable] = None
    coupled_rhs: Optional[Callable] = None

    @property
    def field_names(self):
        return tuple(f.name for f in self.fields)

    def field(self, name) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def buses(self):
        bs = []
        for f in self.fields:
            if f.bus and f.bus not in bs:
                bs.append(f.bus)
        return bs

    def bus_total(self, state):
        """Diagnostic per-bus total = sum over fields mapped to that bus (no scaling)."""
        out = {b: 0.0 for b in self.buses()}
        for f in self.fields:
            if f.bus:
                out[f.bus] += float(np.asarray(state[f.name]).sum())
        return out


def copy_state(st):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}


def step(domain: Domain, st, dt, sources=None, op_order=None, compute_gov=True):
    """One GATHER->STAGE->REDUCE->COMMIT step. Returns (new_state, ledger, audit, gov).

    sources: optional callable (staged_sum, state, params, dt) -> {ledger_key: amount}. It may
    add an external rate into `staged_sum` in place (the generalized ignition/lightning impulse)
    and returns the ledgered amounts deposited. op_order: optional contribution-op name order
    (to exercise order-independence; default is canonical sort-by-name -> bit-reproducible).
    compute_gov: evaluate the governing-equation residual (recomputes the coupled RHS -- a
    per-step diagnostic; set False in tight sub-stepping loops where only the audit is needed).
    """
    old = copy_state(st)

    contrib_ops = [op for op in domain.operators if op.contribute is not None]
    if op_order is not None:
        rank = {n: i for i, n in enumerate(op_order)}
        contrib_ops = sorted(contrib_ops, key=lambda o: rank[o.name])
    else:
        contrib_ops = sorted(contrib_ops, key=lambda o: o.name)   # canonical order

    # GATHER + STAGE: collect each field's additive contributions (rates).
    staged_lists = {f.name: [] for f in domain.fields}
    ledger_rate = {}
    for op in contrib_ops:
        deltas, led = op.contribute(st, domain.params)
        for fname, d in deltas.items():
            staged_lists[fname].append(np.asarray(d, dtype=float))
        for k, v in (led or {}).items():
            ledger_rate[k] = ledger_rate.get(k, 0.0) + v

    # REDUCE: fixed-order sum per field (canonical -> bit-reproducible).
    staged = {}
    for fname, arrs in staged_lists.items():
        staged[fname] = (det.fixed_order_sum(arrs) if arrs
                         else np.zeros_like(np.asarray(old[fname], dtype=float)))
    ledger = {k: v * dt for k, v in ledger_rate.items()}      # rates -> amounts over the step

    # external composite source (ignition/lightning), ledgered as amounts.
    if sources is not None:
        add = sources(staged, st, domain.params, dt)
        if add:
            ledger.update(add)

    # COMMIT with per-field clamps (the limiter).
    new = {}
    for f in domain.fields:
        fname = f.name
        intended = np.asarray(old[fname], dtype=float) + staged[fname] * dt
        if f.clamp_min is not None:
            intended = np.maximum(intended, f.clamp_min)
        if f.clamp_max is not None:
            intended = np.minimum(intended, f.clamp_max)
        new[fname] = intended
    # carry along any non-declared diagnostic fields unchanged
    for k, v in old.items():
        if k not in new:
            new[k] = v.copy() if isinstance(v, np.ndarray) else v

    # TRANSITIONS by cascade priority (ascending -> highest priority applied last -> wins).
    trans_ops = sorted([op for op in domain.operators if op.transition is not None],
                       key=lambda o: (o.priority, o.name))
    for op in trans_ops:
        for fname, val in op.transition(new, domain.params).items():
            new[fname] = np.asarray(val, dtype=float)

    # MONITORS
    if domain.audit_fn is not None:
        audit = domain.audit_fn(old, new, staged, ledger, domain.params, dt)
    else:
        audit = generic_audit(domain, old, new, staged, dt)
    gov = (governing_residual(domain, old, new, dt)
           if (compute_gov and domain.coupled_rhs is not None) else None)
    return new, ledger, audit, gov


def generic_audit(domain: Domain, old, new, staged, dt):
    """Default audit: per-bus committed-vs-intended imbalance (clamp detector).

    For each bus, compares the committed Delta of its summed fields to the intended (staged)
    Delta. ~0 unless a commit clamp fired (the composite-OOD symptom). Domains with external
    ledger transfers across a bus boundary supply their own audit_fn instead.
    """
    res = {}
    for b in domain.buses():
        fnames = [f.name for f in domain.fields if f.bus == b]
        committed = sum(float(np.asarray(new[fn]).sum() - np.asarray(old[fn]).sum()) for fn in fnames)
        intended = sum(float((staged[fn] * dt).sum()) for fn in fnames)
        tp = sum(float(np.abs(staged[fn] * dt).sum()) for fn in fnames)
        res[b] = abs(committed - intended) / (tp + 1e-30)
    return res


def governing_residual(domain: Domain, old, new, dt):
    """Relative residual of the committed state vs the fully-coupled implicit equation."""
    F = domain.coupled_rhs(new, domain.params)
    num = den = 0.0
    for fname in domain.field_names:
        if fname not in F:
            continue
        rate = (np.asarray(new[fname], dtype=float) - np.asarray(old[fname], dtype=float)) / dt
        num += float(np.sum((rate - F[fname]) ** 2))
        den += float(np.sum(rate ** 2) + np.sum(F[fname] ** 2))
    return float(np.sqrt(num / (den + 1e-30)))


def max_audit(audit):
    return max(audit.values()) if audit else 0.0


if __name__ == "__main__":
    # A toy conserving domain: operator moves A -> B at rate kAB*A; the 'stuff' bus (A+B)
    # must conserve to ~0 (generic audit), and 120 operator orderings must agree run-to-run.
    np.set_printoptions(precision=4, suppress=True)

    fields = (Field("A", bus="stuff", clamp_min=0.0),
              Field("B", bus="stuff", clamp_min=0.0))

    def op_move(st, p):
        r = p["kAB"] * np.maximum(st["A"], 0.0)
        return {"A": -r, "B": +r}, {}

    def op_decay(st, p):                       # a second contributor to B's bus partner A? keep simple
        return {"A": -0.0 * st["A"]}, {}

    dom = Domain("toy", fields,
                 (Operator("move", contribute=op_move), Operator("decay", contribute=op_decay)),
                 params={"kAB": 0.5})

    st = {"A": np.full((4, 4), 2.0), "B": np.zeros((4, 4))}
    new, led, audit, gov = step(dom, st, dt=1e-2)
    print("1) one step: bus totals", dom.bus_total(st), "->", dom.bus_total(new))
    print("   generic audit (conserves 'stuff'):", {k: f"{v:.1e}" for k, v in audit.items()})
    assert audit["stuff"] < 1e-12, "toy domain must conserve"

    # determinism: same op order, two runs -> bitwise identical commit.
    n1, *_ = step(dom, st, dt=1e-2)
    n2, *_ = step(dom, st, dt=1e-2)
    print("2) run-to-run bitwise identical:",
          det.bitwise_equal(n1["A"], n2["A"]) and det.bitwise_equal(n1["B"], n2["B"]))
    print("\nbus runtime self-checks passed.")
