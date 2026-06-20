"""V1.8 calibration scratch (UNCOMMITTED). Confirms the criteria are robust across seeds, dims,
and cut scenarios before the notebook freezes them. Not a test."""
import sys
sys.path.insert(0, ".")
import growth as g


def biggest_branch(nodes, depth=1):
    cand = {}
    for n in nodes:
        if n[5] == g.LAYER_WOOD and len(n[0]) >= depth:
            rb = n[0][:depth]
            cand[rb] = cand.get(rb, 0) + 1
    return max(cand, key=cand.get) if cand else None


for dim in (2, 3):
    print(f"=== dim={dim} ===")
    for seed in range(5):
        gp = g.GrowthParams(dim=dim)
        T, LOD = gp.max_gen, gp.max_order

        # A: memo trace == fresh over a dense (time,LOD) grid
        trace = g.full_trace(gp, seed)
        mism = 0
        for t in range(T + 1):
            for lod in range(LOD + 1):
                if g.evaluate(trace, t, lod) != g.grow(t, lod, gp=gp, seed=seed, cache=None):
                    mism += 1
        det = g.trace_digest(g.full_trace(gp, seed)) == g.trace_digest(g.full_trace(gp, seed))

        # B: single cut (biggest depth-1 branch) + a multi-cut write-back
        intact = g.grow(T, LOD, gp=gp, seed=seed, cache=None)
        b1 = biggest_branch(intact, 1)
        b2 = biggest_branch(intact, 2)
        scenarios = {"single": {b1: 1}}
        if b2 is not None:
            scenarios["multi"] = {b1: 2, b2: 1}

        worst_good_stale, min_bad_stale, heal_ok = 0, 10 ** 9, True
        for name, cuts in scenarios.items():
            oracle = g.grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=None)
            cg = {}; g.grow(T, LOD, cuts={}, gp=gp, seed=seed, cache=cg, include_wb=True)
            good = g.grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=cg, include_wb=True)
            cb = {}; g.grow(T, LOD, cuts={}, gp=gp, seed=seed, cache=cb, include_wb=False)
            bad = g.grow(T, LOD, cuts=cuts, gp=gp, seed=seed, cache=cb, include_wb=False)
            worst_good_stale = max(worst_good_stale, g.stale_replay_count(good, oracle),
                                   0 if good == oracle else 1)
            min_bad_stale = min(min_bad_stale, g.stale_replay_count(bad, oracle))
            heal_ok = heal_ok and g.callus_count(oracle) > 0

        print("  seed=%d  memo_mism=%d det=%s  good_stale=%d  min_bad_stale=%d  heal=%s"
              % (seed, mism, det, worst_good_stale, min_bad_stale, heal_ok))
print("DONE")
