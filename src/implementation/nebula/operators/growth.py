"""
Growth — writing the implicit field (ARCHITECTURE §III.1; Decision #11).
Verified by V1.8 (PASS): a memoized growth trace evaluates bit-exactly to a fresh grow at
any (time, LOD), and a write-back (recorded cut) heals around the wound rather than
replaying the pre-wound limb -- *iff* the memoization key includes the write-back state.

Growth is an ACTIVE FRONT that each generation senses local fields, runs a seeded
field-biased L-system rule, and DEPOSITS nodes plus their layer/ring tags. "Layers are the
integral of the front's history." Two halves:

  1. The verified trace machinery (ported verbatim-in-behaviour from the frozen oracle
     src/verification/oracles/growth.py): hashed-per-branch-key RNG (bit-reproducible),
     the memoizable trace, (time, LOD)=branch-order-truncation evaluation, and write-back /
     callus heal with the cut IN the cache key. The load-bearing design point: every
     branch's RNG is hashed from its lineage key (determinism.stable_hash), never the global
     draw order or Python's salted hash() -- that is what makes per-subtree memoization sound.

  2. The FIVE growth/process operators that turn the trace into a TreeModel (the implicit
     substrate the SDF, restriction, and physics cloud read):
       (1) apical extension   -- the meristem front advances the skeleton (the L-system walk)
       (2) branching          -- L-system + light-biased child spawning
       (3) cambium rings      -- secondary growth: per-season radius -> taper + ring layers
       (4) reaction wood      -- stress-driven asymmetric thickening (Wolff analog; the same
                                 rule reaction wood and bone share -- multiplicative, not linear)
       (5) heartwood transition-- xylem old enough chemically transitions sapwood -> heartwood

Dimension-agnostic integer lattice; up-axis = last coordinate. Pure numpy + hashlib.
"""
from dataclasses import dataclass, astuple, field as _dc_field

import numpy as np

from ..core.determinism import stable_hash, bitwise_equal

# layer tags
LAYER_WOOD = 0
LAYER_CALLUS = 1
# material layer ids for the TreeModel (bark outermost -> heartwood innermost)
MAT_BARK, MAT_SAPWOOD, MAT_HEARTWOOD = 0, 1, 2


@dataclass(frozen=True)
class GrowthParams:
    # --- L-system front (ported; the V1.8 parameters) ---
    dim: int = 3               # 2 or 3 (up-axis = last coordinate)
    max_gen: int = 22          # total generations of growth
    max_order: int = 4         # maximum branch order (the LOD ceiling)
    trunk_len: int = 14        # order-0 branch length; children shorter by len_falloff
    len_falloff: float = 0.7   # child length = round(trunk_len * falloff**order)
    branch_prob: float = 0.6   # base per-step branching probability (x a light factor)
    max_children: int = 3      # up to this many children at a branch point
    heal_delay: int = 1        # generations after a cut before callus begins
    callus_gens: int = 4       # generations the callus front deposits
    callus_radius: int = 1     # lattice radius of the callus ring around the wound
    # --- secondary growth / material structure (the new process operators) ---
    # radii are kept small RELATIVE to the internode spacing so the capsule union reads as a
    # slender trunk + branches (not a blob): trunk ~ spacing, twigs ~ 0.1x spacing.
    spacing: float = 0.16      # lattice -> world units (internode length)
    base_radius: float = 0.022  # pith radius of a freshly-deposited segment
    growth_per_season: float = 0.005   # cambium radial increment per season of secondary growth
    order_taper: float = 0.6   # radius falloff per branch order (tips thinner)
    min_radius: float = 0.02   # absolute floor so twigs stay visible (not sub-voxel)
    bark_thickness: float = 0.018      # outer bark shell thickness
    heartwood_age: int = 8     # seasons before sapwood chemically transitions to heartwood
    reaction_gain: float = 0.4         # extra thickening per unit bending stress proxy (Wolff)
    reaction_eta: float = 0.3          # Wolff fully-stressed exponent (V1.7: multiplicative)


def params_id(gp: GrowthParams):
    return astuple(gp)


# ============================================================================================
# Part 1 -- the verified L-system trace machinery (ported from the frozen oracle)
# ============================================================================================

def _branch_rng(seed, pid, key):
    return np.random.default_rng(stable_hash(seed, pid, key))


def _branch_plan(key, lbias, order, birth_gen, gp: GrowthParams, seed, pid):
    """Deterministic plan of one branch from its hashed sub-seed (env-independent)."""
    L = gp.dim - 1
    rng = _branch_rng(seed, pid, key)
    length = max(1, int(round(gp.trunk_len * gp.len_falloff ** order)))
    drifts = np.zeros((length, L), dtype=np.int64)
    for s in range(length):
        for ax in range(L):
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
    key, start_pos, lbias, birth_gen, order = ctx
    if order > lod or birth_gen > time:
        return
    length, drifts, children = _branch_plan(key, lbias, order, birth_gen, gp, seed, pid)
    cut_step = cuts.get(key, None)
    pos = list(start_pos)
    cut_pos, cut_gen = None, None
    for s in range(length + 1):
        gen = birth_gen + s
        if gen > time:
            break
        if cut_step is not None and s > cut_step:
            break
        out.append((key, s, gen, order, tuple(pos), LAYER_WOOD))
        if cut_step is not None and s == cut_step:
            cut_pos, cut_gen = tuple(pos), gen
        if s in children:
            for ci, cb in enumerate(children[s]):
                child_ctx = ((key + ((s, ci),)), tuple(pos), cb, gen + 1, order + 1)
                _grow_into(child_ctx, time, lod, cuts, gp, seed, pid, out, cache, include_wb)
        if s < length:
            for ax in range(gp.dim - 1):
                pos[ax] += int(drifts[s, ax])
            pos[gp.dim - 1] += 1
    if cut_pos is not None:
        _grow_callus(key, cut_pos, cut_gen, order, time, lod, gp, seed, pid, out)


def _grow_callus(cut_key, cut_pos, cut_gen, order, time, lod, gp, seed, pid, out):
    if order > lod:
        return
    ckey = (-1,) + cut_key
    idx = 0
    for g in range(gp.callus_gens):
        gen = cut_gen + gp.heal_delay + g
        if gen > time:
            break
        rr = min(gp.callus_radius + g, gp.callus_radius + 1)
        for ax in range(gp.dim):
            for d in (-rr, rr):
                p = list(cut_pos)
                p[ax] += d
                out.append((ckey, idx, gen, order, tuple(p), LAYER_CALLUS))
                idx += 1


def _subtree_key(ctx, time, lod, cuts, include_wb):
    key, start_pos, lbias, birth_gen, order = ctx
    base = (key, start_pos, tuple(lbias), birth_gen, order, time, lod)
    if include_wb:
        wb = tuple(sorted(cuts.items()))
        return base + ("wb", stable_hash(wb))
    return base


def _grow_into(ctx, time, lod, cuts, gp, seed, pid, out, cache, include_wb):
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
    """Grow to (time, LOD) under write-back `cuts` (branch key -> sever local-step).

    cache=None is the from-scratch oracle; pass a dict to memoize subtrees. include_wb=False
    omits the write-back state from cache keys (the buggy scheme the V1.8 necessity test exposes).
    Returns the canonical node set (frozenset of (key, step, gen, order, pos, layer) tuples).
    """
    cuts = dict(cuts or {})
    pid = params_id(gp)
    L = gp.dim - 1
    root_ctx = ((), tuple([0] * gp.dim), tuple([0] * L), 0, 0)
    out = []
    _grow_into(root_ctx, time, lod, cuts, gp, seed, pid, out, cache, include_wb)
    return frozenset(out)


def full_trace(gp=GrowthParams(), seed=0, cuts=None):
    """The compact memoized object: grow once to (max_gen, max_order)."""
    return grow(gp.max_gen, gp.max_order, cuts=cuts, gp=gp, seed=seed, cache=None)


def evaluate(trace, time, lod):
    """Evaluate a memoized trace at (time, LOD): a pure filter (gen <= time, order <= LOD)."""
    return frozenset(n for n in trace if n[2] <= time and n[3] <= lod)


def trace_digest(nodes):
    """A stable integer digest of a node set (canonical-sorted) -- the bit-exact key."""
    import hashlib
    blob = repr(sorted(nodes)).encode()
    return int.from_bytes(hashlib.blake2b(blob, digest_size=16).digest(), "big")


def wood_positions(nodes):
    pts = sorted(n[4] for n in nodes if n[5] == LAYER_WOOD)
    return np.array(pts, dtype=np.int64)


def stale_replay_count(result, oracle):
    return len(result - oracle)


def callus_count(nodes):
    return sum(1 for n in nodes if n[5] == LAYER_CALLUS)


# ============================================================================================
# Part 2 -- the five growth/process operators -> a TreeModel (the implicit substrate)
# ============================================================================================

@dataclass
class TreeModel:
    """The grown tree as an implicit substrate (never explicit fine nodes).

    Per skeleton node: world position, parent index, branch order, birth generation, and the
    derived secondary-growth structure -- cambium radius, heartwood radius r_heart, bark inner
    radius r_bark, and a reaction-wood scalar. (pos[:, last] is the up-axis.)
    """
    seed: int
    age: int
    params: GrowthParams
    pos: np.ndarray          # (M,3) world positions
    parent: np.ndarray       # (M,) parent index, -1 at root
    order: np.ndarray        # (M,)
    gen: np.ndarray          # (M,) birth generation
    radius: np.ndarray       # (M,) cambium outer radius
    r_heart: np.ndarray      # (M,) heartwood radius (<= radius)
    r_bark: np.ndarray       # (M,) sapwood/bark interface radius (<= radius)
    reaction: np.ndarray     # (M,) reaction-wood fraction in [0,1]

    @property
    def n(self):
        return self.pos.shape[0]

    def segments(self):
        """Capsule segments (a, b, ra, rb) for every non-root node (parent -> node)."""
        segs = []
        for i in range(self.n):
            j = int(self.parent[i])
            if j < 0:
                continue
            segs.append((self.pos[j], self.pos[i], float(self.radius[j]), float(self.radius[i])))
        return segs

    def bounds(self, pad=0.2):
        r = float(self.radius.max()) + pad
        lo = self.pos.min(0) - r
        hi = self.pos.max(0) + r
        return lo, hi


def _to3d(p, dim):
    """Lift a lattice tuple to a 3-vector (2D grows in x-z plane, z up)."""
    if dim == 3:
        return (p[0], p[1], p[2])
    return (p[0], 0.0, p[1])


def build_skeleton(trace, gp: GrowthParams):
    """Operators (1) apical extension + (2) branching: the front's deposited wood nodes ->
    a continuous skeleton (positions, parent map, order, birth generation)."""
    wood = sorted(n for n in trace if n[5] == LAYER_WOOD)   # canonical order -> determinism
    index = {(n[0], n[1]): i for i, n in enumerate(wood)}
    M = len(wood)
    pos = np.zeros((M, 3)); parent = np.full(M, -1, np.int64)
    order = np.zeros(M, np.int64); gen = np.zeros(M, np.int64)
    for i, n in enumerate(wood):
        key, step, g, od, p, _ = n
        pos[i] = np.asarray(_to3d(p, gp.dim), float) * gp.spacing
        order[i] = od; gen[i] = g
        if step > 0:
            parent[i] = index.get((key, step - 1), -1)
        elif key:                                   # branch root attaches to its parent branch
            parent[i] = index.get((key[:-1], key[-1][0]), -1)
    return pos, parent, order, gen


def op_cambium_rings(order, gen, age, gp: GrowthParams):
    """Operator (3): radial secondary growth. A node deposited at generation `gen` has had
    (age - gen) seasons of cambium activity -> radius grows with age, tapers with branch order.
    Returns (radius, r_bark)."""
    seasons = np.maximum(age - gen, 0).astype(float)
    taper = gp.order_taper ** order.astype(float)
    radius = (gp.base_radius + gp.growth_per_season * seasons) * taper
    radius = np.maximum(radius, gp.min_radius)         # twigs stay visible (not sub-voxel)
    r_bark = np.maximum(radius - gp.bark_thickness, 0.25 * radius)
    return radius, r_bark


def op_heartwood_transition(gen, age, gp: GrowthParams, taper):
    """Operator (5): xylem old enough chemically transitions sapwood -> heartwood. The
    heartwood radius is the cambium radius as of (age - heartwood_age); younger wood is all
    sapwood (r_heart = 0). A cascade-priority transition on the layer field."""
    heart_seasons = np.maximum(age - gen - gp.heartwood_age, 0).astype(float)
    r_heart = (gp.base_radius + gp.growth_per_season * heart_seasons) * taper
    r_heart = np.where(age - gen > gp.heartwood_age, r_heart, 0.0)
    return r_heart


def op_reaction_wood(pos, parent, gen, gp: GrowthParams):
    """Operator (4): stress-driven asymmetric thickening (Wolff's law -- the same rule bone
    and reaction wood share, ARCHITECTURE §III.7). The bending stress at a node is proxied by
    the horizontal lever arm to the root axis x the supported subtree load (descendant count),
    so a heavy branch reaching far out -- exactly where reaction wood forms -- is most stressed.
    The fully-stressed MULTIPLICATIVE update (V1.7: a naive linear SED rule oscillates and
    fragments) bumps the radius where loaded. Returns (reaction in [0,1], radius_multiplier)."""
    M = len(pos)
    if M == 0:
        return np.zeros(0), np.ones(0)
    # supported subtree load: postorder accumulation (children have strictly greater gen).
    load = np.ones(M)
    for i in np.argsort(-gen):                                      # descending gen -> child before parent
        j = int(parent[i])
        if j >= 0:
            load[j] += load[i]
    root_xy = pos[0, :2]
    lever = np.linalg.norm(pos[:, :2] - root_xy, axis=1)           # horizontal cantilever arm
    stress = lever * load                                          # bending moment proxy
    reaction = np.clip(stress / max(float(stress.max()), 1e-12), 0.0, 1.0)
    mult = (1.0 + gp.reaction_gain * reaction) ** gp.reaction_eta  # fully-stressed multiplicative
    return reaction, mult


def grow_tree(seed=0, age=None, gp: GrowthParams = GrowthParams(), lod=None, cuts=None):
    """Run the five growth/process operators to (age, LOD) -> a TreeModel.

    age defaults to gp.max_gen; lod defaults to gp.max_order. `cuts` applies a write-back
    (recorded sever) so growth heals around the wound (V1.8). Deterministic in (seed, params,
    age, lod, cuts) via hashed sub-seeds.
    """
    age = gp.max_gen if age is None else int(age)
    lod = gp.max_order if lod is None else int(lod)
    trace = grow(age, lod, cuts=cuts, gp=gp, seed=seed, cache=None)
    pos, parent, order, gen = build_skeleton(trace, gp)                  # ops 1+2
    radius, r_bark = op_cambium_rings(order, gen, age, gp)               # op 3
    reaction, mult = op_reaction_wood(pos, parent, gen, gp)              # op 4
    radius = radius * mult
    r_bark = r_bark * mult
    taper = gp.order_taper ** order.astype(float)
    r_heart = op_heartwood_transition(gen, age, gp, taper) * mult        # op 5
    r_heart = np.minimum(r_heart, r_bark)                               # heartwood inside sapwood
    return TreeModel(seed, age, gp, pos, parent, order, gen, radius, r_heart, r_bark, reaction)


if __name__ == "__main__":
    # 1) V1.8 regression guard: memoized trace == fresh grow, and write-back heals.
    gp = GrowthParams(dim=3)
    seed = 7
    trace = full_trace(gp, seed)
    mism = 0
    for t in range(0, gp.max_gen + 1, 4):
        for lod in range(gp.max_order + 1):
            if evaluate(trace, t, lod) != grow(t, lod, gp=gp, seed=seed, cache=None):
                mism += 1
    print(f"1) memo==fresh mismatches over (time,LOD): {mism}")
    assert mism == 0

    T, LOD = gp.max_gen, gp.max_order
    intact = grow(T, LOD, gp=gp, seed=seed, cache=None)
    cand = {}
    for n in intact:
        if n[5] == LAYER_WOOD and len(n[0]) >= 1:
            cand[n[0][:1]] = cand.get(n[0][:1], 0) + 1
    cut_key = max(cand, key=cand.get)
    cuts = {cut_key: 1}
    oracle_cut = grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=None)
    cache = {}
    grow(T, LOD, cuts={}, gp=gp, seed=seed, cache=cache, include_wb=True)
    good = grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=cache, include_wb=True)
    print(f"2) write-back in key: healed==oracle={good == oracle_cut} "
          f"stale={stale_replay_count(good, oracle_cut)} callus={callus_count(good)}")
    assert good == oracle_cut and stale_replay_count(good, oracle_cut) == 0

    cache_b = {}
    grow(T, LOD, cuts={}, gp=gp, seed=seed, cache=cache_b, include_wb=False)
    bad = grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=cache_b, include_wb=False)
    print(f"3) write-back OMITTED: stale-replay incidents={stale_replay_count(bad, oracle_cut)} (>0)")
    assert stale_replay_count(bad, oracle_cut) > 0

    # 4) the five operators -> a TreeModel.
    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    print(f"4) TreeModel: {tree.n} skeleton nodes, {len(tree.segments())} segments")
    print(f"   radius [{tree.radius.min():.3f},{tree.radius.max():.3f}]  "
          f"heartwood nodes={int((tree.r_heart > 0).sum())}  "
          f"max reaction={tree.reaction.max():.2f}")
    print(f"   trunk base radius (node 0) = {tree.radius[0]:.3f} (thickest, oldest+most loaded)")
    assert tree.radius[0] == tree.radius.max() or tree.radius[0] > tree.radius.mean()
    assert (tree.r_heart <= tree.r_bark + 1e-12).all()
    print("\ngrowth self-checks passed.")
