"""
Connectivity channel g_perc (Decision #15; Risk: percolation). Verified by V2.2 (CONSTRAIN +
GRADED FIX): volume-fraction homogenization (Voigt/Reuss/gap) is PROVABLY blind to connectivity
-- byte-identical for a percolating seam and a matched scattered control ~8x stiffer -- because
the bounds are one-point/fraction-only. The fix folds connectivity INTO the trust scalar with a
graded directional scalar-conductance residual g_perc (rank rho~0.90 with the true DNS knockdown,
100% pair discrimination, ~0.2x the elastic-DNS cost), plus a 26-connectivity span check as the
hard backstop wherever seams form (char, cracks).

The conductance residual solves a scalar Laplace problem div(kappa grad phi)=0 with kappa_i=E_i:
a PDE on the ACTUAL phase field, so it SEES the geometry (justified as an elastic surrogate by the
cross-property bounds, Torquato / Gibiansky-Torquato). UNLIKE Voigt/Reuss it separates a soft seam
percolating normal to an axis from a scattered cluster of equal fraction.

Ported (production functions only) from the frozen oracle src/verification/oracles/percolation.py;
the verification seam battery (violent_cells deps) stays in the oracle. The 26-conn span check is
reimplemented standalone via scipy.ndimage (no Tier-2 dependency). Optional cupy GPU CG.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import label, generate_binary_structure

try:
    import cupy as _cp
    import cupyx.scipy.sparse as _csp
    import cupyx.scipy.sparse.linalg as _cspla
    _HAS_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_GPU = False

_CG_RTOL = 1e-10
_CG_MAXITER = 20000


def _phase_conductivity(grid, materials):
    """Per-voxel scalar conductivity kappa_i = E_i (the cross-property analogue of stiffness)."""
    Es = np.array([E for (E, _) in materials], float)
    return Es[np.asarray(grid)].astype(np.float64)


def _solve_spd(A, b, use_gpu=True):
    """Solve the SPD scalar system A x = b — GPU Jacobi-PCG (cupy) or CPU sparse direct."""
    if use_gpu and _HAS_GPU:
        Ag = _csp.csr_matrix(A.astype(np.float64))
        diag = Ag.diagonal()
        Minv = _cspla.LinearOperator(Ag.shape, matvec=lambda v: v / diag)
        x, info = _cspla.cg(Ag, _cp.asarray(b), M=Minv, rtol=_CG_RTOL, atol=0.0, maxiter=_CG_MAXITER)
        if info != 0:
            raise RuntimeError(f"conduction CG did not converge (info={info})")
        return _cp.asnumpy(x)
    return spla.spsolve(A.tocsc(), b)


def directional_conductance(grid, materials, use_gpu=True):
    """Effective scalar conductance per axis via periodic finite-volume homogenization (kappa_i=E_i).

    Cell-centred 7-point Laplacian with HARMONIC-MEAN face conductances (series), periodic BC, one
    node pinned. UNLIKE Voigt/Reuss it SEES connectivity: a percolating soft seam normal to axis d
    collapses K_eff[d] toward the series floor. The Laplacian is assembled once, reused per direction.
    Returns K_eff (3,).
    """
    kappa = _phase_conductivity(grid, materials)
    n = kappa.shape[0]
    N = n ** 3
    idx = np.arange(N).reshape(n, n, n)
    kflat = kappa.ravel()
    edge_lo, edge_hi, edge_g = {}, {}, {}
    rows, cols, vals = [], [], []
    for a in range(3):
        lo = idx.ravel()
        hi = np.roll(idx, -1, axis=a).ravel()
        kb = np.roll(kappa, -1, axis=a).ravel()
        gg = 2.0 * kflat * kb / (kflat + kb)             # harmonic-mean face conductance (series)
        edge_lo[a], edge_hi[a], edge_g[a] = lo, hi, gg
        rows += [lo, hi, lo, hi]; cols += [lo, hi, hi, lo]; vals += [gg, gg, -gg, -gg]
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N)).tocsr()
    free = np.arange(1, N)
    Lff = L[free][:, free]
    K_eff = np.zeros(3)
    for d in range(3):
        b = np.zeros(N)
        np.add.at(b, edge_hi[d], edge_g[d])
        np.add.at(b, edge_lo[d], -edge_g[d])
        phi = np.zeros(N)
        phi[free] = _solve_spd(Lff, -b[free], use_gpu=use_gpu)
        energy = 0.0
        for a in range(3):
            dphi = phi[edge_hi[a]] - phi[edge_lo[a]] + (1.0 if a == d else 0.0)
            energy += float(np.dot(edge_g[a], dphi * dphi))
        K_eff[d] = energy / N
    return K_eff


def wiener_bounds(grid, materials):
    """Fraction-only scalar conductance bounds (the scalar Voigt/Reuss; connectivity-blind).

    K_arith = sum f_i kappa_i (parallel/upper); K_harm = 1/sum(f_i/kappa_i) (series/lower)."""
    phases = np.asarray(grid).ravel()
    P = len(materials)
    f = np.bincount(phases, minlength=P) / phases.size
    kappa = np.array([E for (E, _) in materials], float)
    return float(np.dot(f, kappa)), float(1.0 / np.dot(f, 1.0 / kappa))


def connectivity_residual(grid, materials, use_gpu=True):
    """g_perc[d] = (K_arith - K_eff[d]) / (K_arith - K_harm) in [0,1] — the graded, directional
    connectivity signal the V-R gap cannot provide. ~0 for a scattered cluster, ~1 for a soft seam
    percolating normal to d. Returns (3,)."""
    K_eff = directional_conductance(grid, materials, use_gpu=use_gpu)
    K_arith, K_harm = wiener_bounds(grid, materials)
    denom = max(K_arith - K_harm, 1e-300)
    return np.clip((K_arith - K_eff) / denom, 0.0, 1.0)


# ---- the 26-connectivity span backstop (standalone; scipy.ndimage, no Tier-2 deps) ----

def _soft_phase(materials):
    return int(np.argmin([E for (E, _) in materials]))


def _spans(mask, axis):
    """True iff a single 26-connected component of `mask` touches both `axis` faces."""
    lab, nlab = label(mask, structure=generate_binary_structure(3, 3))   # 26-connectivity
    if nlab == 0:
        return False
    lo = set(np.unique(np.take(lab, 0, axis=axis))) - {0}
    hi = set(np.unique(np.take(lab, mask.shape[axis] - 1, axis=axis))) - {0}
    return len(lo & hi) > 0


def percolates_load_plane(grid, materials, phase=None):
    """Hard backstop: the DEFECT phase spans across x OR z (the load plane; cells are y-extruded
    so a y-span is trivial). 26-connectivity catches thin diagonal soft paths. `phase` selects the
    defect (char / crack); default is the weakest phase. Pass `phase` so an authored soft LAYER
    (e.g. bark) is not mistaken for a percolating crack."""
    p = _soft_phase(materials) if phase is None else int(phase)
    mask = np.asarray(grid) == p
    return bool(_spans(mask, 0) or _spans(mask, 2))


def spanning_fraction(grid, materials, phase=None):
    """Graded companion: fraction of defect-phase voxels in a face-spanning (x or z) 26-conn cluster."""
    soft = _soft_phase(materials) if phase is None else int(phase)
    mask = np.asarray(grid) == soft
    tot = int(mask.sum())
    if tot == 0:
        return 0.0
    best = 0
    lab, nlab = label(mask, structure=generate_binary_structure(3, 3))
    for axis in (0, 2):
        lo = set(np.unique(np.take(lab, 0, axis=axis))) - {0}
        hi = set(np.unique(np.take(lab, mask.shape[axis] - 1, axis=axis))) - {0}
        for c in (lo & hi):
            best = max(best, int((lab == c).sum()))
    return best / tot


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print(f"conductance backend: {'GPU (cupy CG)' if _HAS_GPU else 'CPU (sparse LU)'}")
    n = 16
    mats = [(10.0, 0.3), (10.0 / 60.0, 0.3)]   # wood + soft (char-like) phase

    # a thin connected soft seam normal to x vs a matched scattered control (identical fractions).
    # thickness 1 keeps the soft fraction below the 26-conn scatter-spanning threshold (V2.2: a
    # thicker seam raises the fraction and a random scatter can spuriously span -- why g_perc is primary).
    seam = np.zeros((n, n, n), np.int64); seam[n // 2:n // 2 + 1, :, :] = 1
    rng = np.random.default_rng(0)
    ctrl = seam.ravel().copy(); ctrl = ctrl[rng.permutation(ctrl.size)].reshape(n, n, n)

    wb_s = wiener_bounds(seam, mats); wb_c = wiener_bounds(ctrl, mats)
    print(f"1) identical fractions -> identical Wiener bounds: {np.allclose(wb_s, wb_c)}")
    gp_s = connectivity_residual(seam, mats); gp_c = connectivity_residual(ctrl, mats)
    print(f"2) g_perc seam {np.round(gp_s,3)} (max {gp_s.max():.3f}) vs control max {gp_c.max():.3f}")
    assert gp_s.max() > gp_c.max(), "seam must be more conductance-penalized than control"
    print(f"3) 26-conn span: seam={percolates_load_plane(seam, mats)} control={percolates_load_plane(ctrl, mats)}"
          f"  seam spanning-frac={spanning_fraction(seam, mats):.2f}")
    assert percolates_load_plane(seam, mats) and not percolates_load_plane(ctrl, mats)

    homo = np.zeros((n, n, n), np.int64)       # single phase -> g_perc == 0, K_eff == E
    K0 = directional_conductance(homo, mats)
    print(f"4) homogeneous K_eff={np.round(K0,3)} (==10), g_perc={np.round(connectivity_residual(homo, mats),3)}")
    assert np.allclose(K0, 10.0, rtol=1e-3)
    print("\npercolation self-checks passed.")
