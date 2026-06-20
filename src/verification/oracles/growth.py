"""
Growth front + memoizable growth trace + write-back/heal — the mechanism under test for
V1.8 (protocol §V1.8; Decision #11; ARCHITECTURE §III.1 "Growth — writing the implicit field").

The architecture's growth model: growth is an **active front** (apical meristems) that each
generation senses local fields, runs a small seeded field-biased L-system rule, and **deposits**
nodes plus their layer/ring tags; "layers are the integral of the front's history." Growth is
**mostly offline, deterministic, and memoized into a compact growth trace** that the runtime
**evaluates at any (time, LOD)**. Crucially it is **not only additive**: a wound is recorded by a
**write-back** into the substrate, and later growth reads it and **heals around the cut** rather
than replaying the pre-wound limb. The architecture's load-bearing caveat: *"memoization keys
must include the write-back state."*

What V1.8 falsifies
-------------------
1. (theorem-check / determinism) Evaluating the **memoized trace** at (time, LOD) is
   deterministic and **bit-exactly equals a fresh from-scratch grow** — growth is prefix- and
   truncation-consistent (LOD = branch-order truncation; "truncation IS level-of-detail").
2. (empirical / necessity) A **write-back** (recorded cut) must invalidate stale memoized
   sub-results. With the cut in the cache key → cache miss → recompute → the wound heals
   (callus deposited, severed limb gone). With the cut **omitted** from the key → a stale cache
   hit serves the pre-wound limb: a **stale replay**. The notebook shows both, proving the key
   must include the write-back state.

The load-bearing design point
-----------------------------
Every branch's RNG stream is derived by **hashing its key** (a stable blake2b digest of the
lineage path), NOT the global draw order. So a branch — and the whole subtree under it — is a
**pure deterministic function of (seed, params, key, time, LOD, write-back state)**. That is
exactly what makes per-subtree memoization sound and bit-reproducible (a memoized subtree
reproduces the fresh one to the bit), and it is why a fixed reduction/sub-seed order matters
(Decision #3 / V0.5 — "the program IS the asset"). Python's builtin `hash()` is salted per
process and would silently break this; we use `hashlib`.

Dimension-agnostic: positions live on an integer lattice in d ∈ {2, 3}; the up-axis is the last
coordinate. Pure numpy + hashlib; reuses `determinism.bitwise_equal` for the bit-exact checks.
"""
from dataclasses import dataclass, astuple
import hashlib

import numpy as np

LAYER_WOOD = 0
LAYER_CALLUS = 1


@dataclass(frozen=True)
class GrowthParams:
    dim: int = 2               # 2 or 3 (up-axis = last coordinate)
    max_gen: int = 22          # total generations of growth
    max_order: int = 4         # maximum branch order (the LOD ceiling)
    trunk_len: int = 11        # order-0 branch length; children shorter by len_falloff
    len_falloff: float = 0.62  # child length = round(trunk_len * falloff**order)
    branch_prob: float = 0.55  # base per-step branching probability (× a light factor)
    max_children: int = 2      # up to this many children at a branch point
    heal_delay: int = 1        # generations after a cut before callus begins
    callus_gens: int = 4       # generations the callus front deposits
    callus_radius: int = 1     # lattice radius of the callus ring around the wound


def params_id(gp: GrowthParams):
    """A canonical, hashable identity for a parameter set (used in cache/sub-seed keys)."""
    return astuple(gp)


def _stable_hash(*parts) -> int:
    """Deterministic 63-bit hash of the parts (blake2b over a canonical repr).

    Unlike builtin hash(), this is identical across processes/runs — the requirement for
    bit-reproducible per-branch sub-seeds and for the write-back-state cache key.
    """
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2 ** 63)


def _branch_rng(seed, pid, key):
    return np.random.default_rng(_stable_hash(seed, pid, key))


def _branch_plan(key, lbias, order, birth_gen, gp: GrowthParams, seed, pid):
    """The deterministic plan of one branch from its hashed sub-seed (env-independent).

    Returns (length, drifts[length, L], branch_steps -> list of child lbias). Truncation by
    time/LOD/cut happens later in the walk; the plan itself depends ONLY on the key lineage.
    """
    L = gp.dim - 1
    rng = _branch_rng(seed, pid, key)
    length = max(1, int(round(gp.trunk_len * gp.len_falloff ** order)))
    drifts = np.zeros((length, L), dtype=np.int64)
    for s in range(length):
        for ax in range(L):
            # follow the branch's outward bias, with a deterministic jitter
            drifts[s, ax] = int(rng.integers(0, 2)) * lbias[ax]
            if rng.random() < 0.15:
                drifts[s, ax] = -drifts[s, ax]
    children = {}
    if order < gp.max_order:
        for s in range(1, length):
            light = min(1.0, (birth_gen + s) / gp.max_gen + 0.3)   # field: more light higher up
            if rng.random() < gp.branch_prob * light:
                nc = int(rng.integers(1, gp.max_children + 1))
                kids = []
                for _ in range(nc):
                    ax = int(rng.integers(0, L))
                    sign = 1 if rng.random() < 0.5 else -1
                    cb = [0] * L
                    cb[ax] = sign
                    kids.append(tuple(cb))
                children[s] = kids
    return length, drifts, children


def _expand(ctx, time, lod, cuts, gp, seed, pid, out, cache, include_wb):
    """Recursively expand the subtree rooted at ctx into `out` (a list of node tuples)."""
    key, start_pos, lbias, birth_gen, order = ctx
    if order > lod or birth_gen > time:
        return
    length, drifts, children = _branch_plan(key, lbias, order, birth_gen, gp, seed, pid)
    cut_step = cuts.get(key, None)                 # write-back: local step at which severed
    pos = list(start_pos)
    cut_pos, cut_gen = None, None
    for s in range(length + 1):
        gen = birth_gen + s
        if gen > time:
            break
        if cut_step is not None and s > cut_step:  # severed: distal wood + children removed
            break
        out.append((key, s, gen, order, tuple(pos), LAYER_WOOD))
        if cut_step is not None and s == cut_step:
            cut_pos, cut_gen = tuple(pos), gen
        if s in children:                          # spawn child subtrees from gen+1
            for ci, cb in enumerate(children[s]):
                child_ctx = ((key + ((s, ci),)), tuple(pos), cb, gen + 1, order + 1)
                _grow_into(child_ctx, time, lod, cuts, gp, seed, pid, out, cache, include_wb)
        if s < length:                             # advance one lattice step
            for ax in range(gp.dim - 1):
                pos[ax] += int(drifts[s, ax])
            pos[gp.dim - 1] += 1
    if cut_pos is not None:                         # the wound heals: a callus front reads it
        _grow_callus(key, cut_pos, cut_gen, order, time, lod, gp, seed, pid, out)


def _grow_callus(cut_key, cut_pos, cut_gen, order, time, lod, gp, seed, pid, out):
    """Deposit a healing callus ring over a recorded cut (later growth reading the write-back)."""
    if order > lod:
        return
    ckey = (-1,) + cut_key                          # -1 sentinel marks a callus lineage
    idx = 0
    for g in range(gp.callus_gens):
        gen = cut_gen + gp.heal_delay + g
        if gen > time:
            break
        rr = min(gp.callus_radius + g, gp.callus_radius + 1)
        for ax in range(gp.dim):                     # a small axis-aligned ring around the wound
            for d in (-rr, rr):
                p = list(cut_pos)
                p[ax] += d
                out.append((ckey, idx, gen, order, tuple(p), LAYER_CALLUS))
                idx += 1


def _subtree_key(ctx, time, lod, cuts, include_wb):
    """Cache key for a subtree. include_wb=False OMITS the write-back state (the bug to expose)."""
    key, start_pos, lbias, birth_gen, order = ctx
    base = (key, start_pos, tuple(lbias), birth_gen, order, time, lod)
    if include_wb:
        wb = tuple(sorted(cuts.items()))            # canonical write-back state -> in the key
        return base + ("wb", _stable_hash(wb))
    return base


def _grow_into(ctx, time, lod, cuts, gp, seed, pid, out, cache, include_wb):
    """Memoized wrapper around _expand: serve/​store this subtree's node list in `cache`."""
    if cache is None:
        _expand(ctx, time, lod, cuts, gp, seed, pid, out, cache, include_wb)
        return
    ck = _subtree_key(ctx, time, lod, cuts, include_wb)
    hit = cache.get(ck)
    if hit is not None:
        out.extend(hit)
        return
    local = []
    _expand(ctx, time, lod, cuts, gp, seed, pid, local, cache, include_wb)
    cache[ck] = local
    out.extend(local)


def grow(time, lod, cuts=None, gp=GrowthParams(), seed=0, cache=None, include_wb=True):
    """Grow to (time, LOD) under write-back `cuts` (dict: branch key -> sever local-step).

    Returns the canonical node set (frozenset of node tuples). `cache=None` is the from-scratch
    oracle; pass a dict to memoize subtrees. Set include_wb=False to omit the write-back state
    from cache keys (the buggy scheme the necessity test exposes).
    """
    cuts = dict(cuts or {})
    pid = params_id(gp)
    L = gp.dim - 1
    root_ctx = ((), tuple([0] * gp.dim), tuple([0] * L), 0, 0)
    out = []
    _grow_into(root_ctx, time, lod, cuts, gp, seed, pid, out, cache, include_wb)
    return frozenset(out)


# ---------------- trace memo + evaluation helpers ----------------

def full_trace(gp=GrowthParams(), seed=0, cuts=None):
    """The compact memoized object: grow once to (max_gen, max_order)."""
    return grow(gp.max_gen, gp.max_order, cuts=cuts, gp=gp, seed=seed, cache=None)


def evaluate(trace, time, lod):
    """Evaluate a memoized trace at (time, LOD): a pure filter (gen ≤ time, order ≤ LOD)."""
    return frozenset(n for n in trace if n[2] <= time and n[3] <= lod)


# ---------------- bit-exact digest + diagnostics ----------------

def trace_digest(nodes):
    """A stable integer digest of a node set (canonical-sorted bytes) — the bit-exact key."""
    blob = repr(sorted(nodes)).encode()
    return int.from_bytes(hashlib.blake2b(blob, digest_size=16).digest(), "big")


def wood_positions(nodes):
    """Sorted int array of WOOD node positions — for determinism.bitwise_equal comparisons."""
    pts = sorted(n[4] for n in nodes if n[5] == LAYER_WOOD)
    return np.array(pts, dtype=np.int64)


def stale_replay_count(result, oracle):
    """Nodes the memoized `result` serves that the from-scratch `oracle` does NOT contain.

    After a cut these are exactly the severed-limb nodes that should have been invalidated —
    each is a stale-replay incident. (0 iff the cache honoured the write-back.)"""
    return len(result - oracle)


def callus_count(nodes):
    return sum(1 for n in nodes if n[5] == LAYER_CALLUS)


def severed_subtree_keys(cut_key):
    """Predicate: is node `n` part of the (distal) wood subtree rooted at the cut branch?"""
    def pred(n):
        return n[5] == LAYER_WOOD and len(n[0]) >= len(cut_key) and n[0][:len(cut_key)] == cut_key
    return pred


if __name__ == "__main__":
    import determinism as det

    for dim in (2, 3):
        gp = GrowthParams(dim=dim)
        seed = 7

        # 1) memoized trace evaluation == fresh from-scratch grow, bit-exact, over (time,LOD).
        trace = full_trace(gp, seed)
        mism = 0
        for t in range(0, gp.max_gen + 1, 3):
            for lod in range(gp.max_order + 1):
                memo = evaluate(trace, t, lod)
                fresh = grow(t, lod, gp=gp, seed=seed, cache=None)
                if memo != fresh or not det.bitwise_equal(wood_positions(memo), wood_positions(fresh)):
                    mism += 1
        print(f"[dim={dim}] 1) memo==fresh mismatches over (time,LOD) grid: {mism}")
        assert mism == 0, "memoized trace != fresh grow"

        # 2) determinism: a repeated full grow is bit-identical (stable hashing, not salted).
        d1 = trace_digest(full_trace(gp, seed))
        d2 = trace_digest(full_trace(gp, seed))
        print(f"[dim={dim}] 2) repeat-grow digest equal: {d1 == d2}")
        assert d1 == d2

        # pick a real interior branch to cut (depth-1 branch with the most descendants).
        T, LOD = gp.max_gen, gp.max_order
        intact = grow(T, LOD, gp=gp, seed=seed, cache=None)
        cand = {}
        for n in intact:
            if n[5] == LAYER_WOOD and len(n[0]) >= 1:
                root_branch = n[0][:1]
                cand[root_branch] = cand.get(root_branch, 0) + 1
        cut_key = max(cand, key=cand.get)
        cuts = {cut_key: 1}                          # sever that branch near its base

        oracle_cut = grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=None)   # the healed oracle

        # 3) correct cache (write-back in key): warm intact, then query cut -> heals, 0 stale.
        cache = {}
        grow(T, LOD, cuts={}, gp=gp, seed=seed, cache=cache, include_wb=True)         # warm intact
        good = grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=cache, include_wb=True)
        good_stale = stale_replay_count(good, oracle_cut)
        print(f"[dim={dim}] 3) correct-key: healed==oracle={good==oracle_cut} "
              f"stale={good_stale} callus={callus_count(good)}")
        assert good == oracle_cut and good_stale == 0

        # 4) buggy cache (write-back OMITTED): stale replay of the severed limb appears.
        cache_b = {}
        grow(T, LOD, cuts={}, gp=gp, seed=seed, cache=cache_b, include_wb=False)       # warm intact
        bad = grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=cache_b, include_wb=False)
        bad_stale = stale_replay_count(bad, oracle_cut)
        print(f"[dim={dim}] 4) buggy-key:  stale-replay incidents={bad_stale} (must be > 0)")
        assert bad_stale > 0, "buggy key failed to produce a stale replay (test is vacuous)"

        # 5) healing: the cut limb is pruned (vs intact) and a callus grows over the wound.
        sev = severed_subtree_keys(cut_key)
        cutline_intact = {n for n in intact if sev(n)}
        cutline_healed = {n for n in oracle_cut if sev(n)}
        removed = cutline_intact - cutline_healed                 # the severed limb
        print(f"[dim={dim}] 5) healing: callus={callus_count(oracle_cut)} "
              f"cut-limb wood {len(cutline_intact)} -> {len(cutline_healed)} (pruned {len(removed)})")
        assert callus_count(oracle_cut) > 0, "no callus grew over the wound"
        assert len(removed) > 0 and removed.isdisjoint(oracle_cut), "limb not pruned / replayed"

    print("\nall growth self-checks passed.")
