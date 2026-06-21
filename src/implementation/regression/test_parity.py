"""
Regression parity tests: the clean ported package reproduces the FROZEN verification oracles
(protocol §8: "convert the proven cores into a regression test that must keep passing").

For verbatim-ported mechanisms (homogenization, the fire bus step, the octree, the determinism
reductions, the Jensen rate) the port must match the oracle BIT-FOR-BIT (same numpy ops, same
order) or within solver tolerance. For the EXTENDED mechanisms (growth, restriction) we instead
guard self-consistency / the verified properties, since they add capability beyond the oracle.

Run:  PYTHONPATH=src/implementation:src/verification/oracles python -m regression.test_parity
(or via pytest with the same paths). Each check prints PASS/▲; any failure raises.
"""
import os
import sys
import pathlib

import numpy as np

# put BOTH the package and the frozen oracles on the path.
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src" / "implementation"))
sys.path.insert(0, str(_REPO / "src" / "verification" / "oracles"))

from nebula.core import determinism as det                       # noqa: E402


def test_determinism_reductions():
    """Ported determinism reductions reproduce the oracle's bit-exact behaviour."""
    import determinism as odet
    keys, vals = odet.make_problem(M=200_000, K=32, seed=3)
    vs, ss, se = odet.canonical_sort(keys, vals)
    o = odet.cpu_fixed_order(vs, ss, se, 32)
    nvs, nss, nse = det.canonical_sort(keys, vals)
    n = det.cpu_fixed_order(nvs, nss, nse, 32)
    assert det.bitwise_equal(o, n), "fixed-order reduction differs from oracle"
    oi, _ = odet.cpu_integer_exact(keys, vals, 32, 1e3)
    ni, _ = det.cpu_integer_exact(keys, vals, 32, 1e3)
    assert np.array_equal(oi, ni), "integer-exact reduction differs from oracle"
    return "determinism reductions bit-exact vs oracle"


def test_homogenization():
    """Ported Voigt/Reuss/gap/directional_estimate match the oracle exactly."""
    import homogenization as oh
    from nebula.restriction import homogenization as nh
    f = [0.34, 0.33, 0.33]
    C = [oh.isotropic_stiffness(E, 0.3) for E in (2.0, 9.0, 12.0)]
    for fn in ("voigt_bound", "reuss_bound"):
        assert np.allclose(getattr(oh, fn)(f, C), getattr(nh, fn)(f, C), atol=0, rtol=0), fn
    Cv, Cr = oh.voigt_bound(f, C), oh.reuss_bound(f, C)
    assert det.bitwise_equal(oh.relative_gap(Cv, Cr), nh.relative_gap(Cv, Cr))
    assert det.bitwise_equal(oh.directional_estimate(f, C, 2), nh.directional_estimate(f, C, 2))
    return "homogenization bit-exact vs oracle"


def test_fire_bus_step():
    """The field-agnostic bus step reproduces the oracle bus_runtime.step_split bit-for-bit
    (same operator order). This is the core conservation/composition mechanism (V0.3/V1.1)."""
    import bus_runtime as obr
    import fire_operators as ofo
    from nebula.core import buses as nbuses
    from nebula.operators import fire as nfo

    op = ofo.FireParams()
    st = obr.make_state(8, T0=600.0, gas0=0.1, o2=0.2)
    # oracle path
    o_new, o_led, o_aud, o_gov = obr.step_split(st, op, dt=1e-3)
    # ported path (same canonical order as the oracle's CONTRIBUTION_OPS dict order)
    dom = nfo.fire_domain(nfo.FireParams())
    n_new, n_led, n_aud, n_gov = nbuses.step(dom, st, 1e-3, op_order=nfo.ORACLE_OP_ORDER)
    for f in ("T", "m_s", "gas", "o2", "char", "q", "S"):
        assert det.bitwise_equal(o_new[f], n_new[f]), f"field {f} differs from oracle"
    for b in ("energy", "mass", "o2", "charge"):
        assert abs(o_aud[b] - n_aud[b]) <= 1e-15 + 1e-12 * abs(o_aud[b]), f"audit {b} differs"
    assert abs(o_gov - n_gov) <= 1e-12 * (abs(o_gov) + 1), "gov residual differs"
    return "fire bus step bit-exact vs oracle (state + audit + gov)"


def test_octree():
    """Ported octree build + BH traversal matches the oracle interaction counts & potentials."""
    import octree as oot
    from nebula.adaptive import octree as not_
    rng = np.random.default_rng(1)
    pts = rng.random((3000, 3)); mas = rng.random(3000) + 0.1
    q = rng.random((100, 3))
    ot_tree = oot.build(pts, mas); ot_lin = oot.dfs_linearize(ot_tree)
    nt_tree = not_.build(pts, mas); nt_lin = not_.dfs_linearize(nt_tree)
    op, oi, _ = oot.bh_field(ot_lin, q, theta=0.5)
    npp, ni, _ = not_.bh_field(nt_lin, q, theta=0.5)
    assert np.array_equal(oi, ni), "BH interaction counts differ"
    assert det.bitwise_equal(op, npp), "BH potentials differ"
    return "octree build + BH bit-exact vs oracle"


def test_jensen():
    """Ported Jensen rate machinery matches the oracle (pure Arrhenius)."""
    import jensen_rate as oj
    from nebula.restriction import jensen as nj
    n = 24
    f = np.broadcast_to((350.0 + 650.0 * (np.arange(n) + 0.5) / n)[:, None, None], (n, n, n))
    for A, Ta in ((3e6, 9000.0), (4e5, 7000.0)):
        assert abs(oj.true_mean_rate(f, A, Ta) - nj.true_mean_rate(f, A, Ta)) <= 1e-9
        assert abs(oj.variance_corrected_rate(f, A, Ta) - nj.variance_corrected_rate(f, A, Ta)) <= 1e-9
        assert abs(oj.variance_error_scalar(f.mean(), f.var(), Ta)
                   - nj.variance_error_scalar(f.mean(), f.var(), Ta)) <= 1e-9
    return "jensen rate matches oracle"


def test_growth_determinism():
    """Extended growth: bit-reproducible (hashed sub-seeds) and the V1.8 memo==fresh property."""
    from nebula.operators import growth as g
    gp = g.GrowthParams(dim=3)
    d1 = g.trace_digest(g.full_trace(gp, seed=5))
    d2 = g.trace_digest(g.full_trace(gp, seed=5))
    assert d1 == d2, "growth not bit-reproducible"
    trace = g.full_trace(gp, seed=5)
    for t in (6, 12, gp.max_gen):
        for lod in range(gp.max_order + 1):
            assert g.evaluate(trace, t, lod) == g.grow(t, lod, gp=gp, seed=5, cache=None)
    return "growth deterministic + memo==fresh (V1.8)"


CHECKS = [test_determinism_reductions, test_homogenization, test_fire_bus_step,
          test_octree, test_jensen, test_growth_determinism]


def run_all():
    ok = True
    for chk in CHECKS:
        try:
            msg = chk()
            print(f"  PASS  {chk.__name__}: {msg}")
        except AssertionError as e:
            ok = False
            print(f"  FAIL  {chk.__name__}: {e}")
    return ok


if __name__ == "__main__":
    print("=== regression parity: ported package vs frozen oracles ===")
    assert run_all(), "regression parity FAILED"
    print("\nall regression parity checks passed.")
