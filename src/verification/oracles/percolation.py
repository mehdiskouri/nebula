"""
Percolation / off-axis seam helpers — the V2.2 danger case (Risk: percolation; Decision #15).

A thin CONNECTED low-stiffness seam destroys effective stiffness out of proportion to its volume
fraction. Volume-fraction homogenization (Voigt/Reuss and hence `relative_gap`) is **blind** to this:
the bounds depend ONLY on phase fractions, so they are identical for a percolating seam and a
scattered cluster of equal volume, and identical across seam orientation. The architecture's claim is
that an axis-aligned seam still self-reports (a principal-axis directional estimate captures the weak
direction) while an OFF-AXIS seam does not — establishing a connectivity span check as a mandatory
hard refine trigger.

This module supplies the V2.2 battery and the two proxy/oracle quantities the notebook compares,
reusing — WITHOUT editing — `violent_cells` (`seam_cell`, `percolates`), `homogenization`
(`voigt_bound`/`reuss_bound`/`relative_gap`/`directional_estimate`/`isotropic_stiffness`), and the DNS
oracle `dns_elasticity_3d.effective_stiffness`. Pure numpy (+ the GPU DNS path).

Voigt convention [11,22,33,23,13,12], engineering shear — matches homogenization.py.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import violent_cells as vc
from homogenization import (isotropic_stiffness, voigt_bound, reuss_bound, relative_gap,
                            directional_estimate)

# Optional GPU backend (cupy) — reuses the same Jacobi-PCG path the DNS oracle uses, so the
# graded connectivity proxy rides the existing GPU infrastructure (dns_elasticity_3d._solve_free).
try:
    import cupy as _cp
    import cupyx.scipy.sparse as _csp
    import cupyx.scipy.sparse.linalg as _cspla
    _HAS_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_GPU = False

_CG_RTOL = 1e-10        # scalar Laplace is far better conditioned than the elastic system
_CG_MAXITER = 20000


@dataclass
class SeamCell:
    """A seam (or control) cell plus the volume-fraction quantities the proxy reads."""
    grid: np.ndarray
    materials: list
    fractions: np.ndarray
    C_phases: list
    angle_deg: float = None
    kind: str = "seam"


def _bundle(grid, materials, angle_deg=None, kind="seam"):
    phases = np.asarray(grid).ravel()
    P = len(materials)
    fr = np.bincount(phases, minlength=P) / phases.size
    Cph = [isotropic_stiffness(E, nu) for (E, nu) in materials]
    return SeamCell(np.asarray(grid), materials, fr, Cph, angle_deg, kind)


def seam_cell_at(n, angle_deg, thickness=3, contrast=60.0):
    """Thin CONNECTED low-stiffness seam crossing the cell at `angle_deg` in the x-z plane.

    Wraps `violent_cells.seam_cell`. thickness defaults to 3 so a 45 deg band stays 6-connected
    (a 1-2 voxel diagonal band can be only corner-connected and would not percolate).
    """
    grid, materials = vc.seam_cell(n, angle_deg, thickness=thickness, contrast=contrast)
    return _bundle(grid, materials, angle_deg, "seam")


def shuffled_control(seam: SeamCell, seed):
    """Matched non-percolating control: the SAME voxels permuted to random positions.

    Identical phase fractions -> BYTE-IDENTICAL Voigt/Reuss/gap as the seam, but the connected path
    is destroyed (soft fraction is below the ~0.31 simple-cubic site-percolation threshold, so a
    random shuffle almost surely does not span). This is the airtight control: the volume-fraction
    proxy cannot tell it apart from the seam, yet its true stiffness is far higher.
    """
    rng = np.random.default_rng(seed)
    flat = seam.grid.ravel().copy()
    flat = flat[rng.permutation(flat.size)]
    return _bundle(flat.reshape(seam.grid.shape), seam.materials, seam.angle_deg, "shuffled")


def percolates_xz(cell: SeamCell):
    """The proposed hard trigger: soft phase spans the cell across x OR z (the load plane).

    Cells are y-extruded, so the meaningful crack percolation is in the x-z plane (mirrors
    `surrogate_gnn.fallback_flags`). Returns bool.
    """
    return bool(vc.percolates(cell.grid, cell.materials, axis=0)
                or vc.percolates(cell.grid, cell.materials, axis=2))


def gap_vector(cell: SeamCell):
    """The per-direction Voigt-Reuss gap (the trust scalar) — a function of fractions ONLY."""
    Cv = voigt_bound(cell.fractions, cell.C_phases)
    Cr = reuss_bound(cell.fractions, cell.C_phases)
    return relative_gap(Cv, Cr)


def best_axis_proxy_error(cell: SeamCell, C_dns):
    """min over the 3 principal layer axes of ||directional_estimate(axis) - C_dns||_F / ||C_dns||_F.

    The shipped orthotropic proxy does as well as the best principal-axis layering choice. For an
    axis-aligned seam ONE axis reproduces the DNS tensor (V0.1 layered-exactness) -> ~0 error; for an
    off-axis seam NO axis-aligned orthotropic tensor can match the rotated DNS tensor (which carries
    a large off-diagonal axial-axial coupling) -> the best-axis error stays large. Returns
    (min_rel_error, best_axis).
    """
    errs = [np.linalg.norm(directional_estimate(cell.fractions, cell.C_phases, a) - C_dns)
            / np.linalg.norm(C_dns) for a in (0, 1, 2)]
    return float(min(errs)), int(np.argmin(errs))


def uniaxial_modulus(C, direction):
    """Young's modulus along a unit `direction` for a 6x6 stiffness C (Voigt, engineering shear).

    Unit uniaxial STRESS sigma = n (x) n; E(n) = 1 / (n . eps . n) with eps from the compliance.
    For a seam this is minimal along the seam NORMAL (the true weak direction) at every angle.
    """
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    S = np.linalg.inv(C)
    sigV = np.array([n[0] ** 2, n[1] ** 2, n[2] ** 2, n[1] * n[2], n[0] * n[2], n[0] * n[1]])
    e = S @ sigV                                       # engineering strain (Voigt)
    eps = np.array([[e[0], e[5] / 2, e[4] / 2],
                    [e[5] / 2, e[1], e[3] / 2],
                    [e[4] / 2, e[3] / 2, e[2]]])
    return 1.0 / float(n @ eps @ n)


def seam_normal(angle_deg):
    """Unit seam normal in the x-z plane for `seam_cell` (matches its `nx=sin, nz=-cos`)."""
    th = np.deg2rad(angle_deg)
    return np.array([np.sin(th), 0.0, -np.cos(th)])


def min_principal_modulus(C):
    """Weakest uniaxial modulus over the 3 PRINCIPAL axes — what a per-principal-axis proxy 'sees'."""
    return min(uniaxial_modulus(C, e) for e in np.eye(3))


def min_modulus_xz(C, n_dirs=181):
    """True weakest uniaxial modulus over all directions in the x-z plane (the seam's plane).

    For a seam this is the genuine structural weakness; it is low at EVERY angle. The gap between
    this and `min_principal_modulus` is the off-axis blind spot: at 45 deg the weak direction is not
    a principal axis, so the principal view far overestimates the true minimum.
    """
    th = np.linspace(0.0, np.pi, n_dirs)
    return min(uniaxial_modulus(C, [np.cos(t), 0.0, np.sin(t)]) for t in th)


# ============================================================================================
# THE GRADED CONNECTIVITY FIX — a cheap directional scalar-conductance proxy.
#
# Voigt/Reuss are fraction-only (one-point statistics) and so are PROVABLY blind to connectivity:
# byte-identical for a percolating seam and a matched scattered control. The cure is a feature that
# is NOT fraction-only. We solve a scalar Laplace problem div(kappa grad phi)=0 with kappa_i = E_i:
# a PDE on the ACTUAL phase field, so it *sees* the geometry. Justification it tracks the elastic
# knockdown: the rigorous cross-property bounds (Torquato; Gibiansky-Torquato) link effective
# conductivity to effective elastic moduli — a cheap scalar solve is an informative surrogate for the
# expensive elastic RVE. Cost: 1 scalar DOF/voxel (vs 3 vector DOF x 6 strain load cases) on a
# well-conditioned SPD Laplacian, reusing the same Jacobi-PCG GPU path.
# ============================================================================================
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

    Scalar analogue of `dns_elasticity_3d.effective_stiffness`: a cell-centred 7-point Laplacian with
    HARMONIC-MEAN face conductances (series across each face), periodic BC, one node pinned to remove
    the constant nullspace; `K_eff[d]` is the energy under a unit macro gradient along axis d. UNLIKE
    Voigt/Reuss this is a PDE solve on the actual phase field, so it SEES connectivity — a percolating
    soft seam normal to d collapses `K_eff[d]` toward the series floor, while a scattered cluster of
    equal fraction does not. The graph Laplacian is assembled ONCE and reused across the 3 directions.
    Returns `K_eff` (3,).
    """
    kappa = _phase_conductivity(grid, materials)
    n = kappa.shape[0]
    N = n ** 3
    idx = np.arange(N).reshape(n, n, n)
    kflat = kappa.ravel()
    edge_lo, edge_hi, edge_g = {}, {}, {}
    rows, cols, vals = [], [], []
    for a in range(3):                                   # one periodic edge per node along each axis
        lo = idx.ravel()
        hi = np.roll(idx, -1, axis=a).ravel()
        kb = np.roll(kappa, -1, axis=a).ravel()
        g = 2.0 * kflat * kb / (kflat + kb)              # harmonic-mean face conductance (series)
        edge_lo[a], edge_hi[a], edge_g[a] = lo, hi, g
        rows += [lo, hi, lo, hi]; cols += [lo, hi, hi, lo]; vals += [g, g, -g, -g]
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N)).tocsr()
    free = np.arange(1, N)                                # pin node 0
    Lff = L[free][:, free]
    K_eff = np.zeros(3)
    for d in range(3):
        # macro drop across each axis-d edge = 1 (unit gradient); RHS b = sum g*Delta*(e_hi - e_lo)
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
    """Fraction-only scalar conductance bounds — the scalar analogue of Voigt/Reuss, connectivity-blind.

    `K_arith = sum f_i kappa_i` (parallel / Wiener upper), `K_harm = 1/sum(f_i/kappa_i)` (series /
    Wiener lower). Depend on fractions ONLY, so a seam and its matched control share both. Returns
    `(K_arith, K_harm)`.
    """
    phases = np.asarray(grid).ravel()
    P = len(materials)
    f = np.bincount(phases, minlength=P) / phases.size
    kappa = np.array([E for (E, _) in materials], float)
    return float(np.dot(f, kappa)), float(1.0 / np.dot(f, 1.0 / kappa))


def connectivity_residual(grid, materials, use_gpu=True):
    """The graded, directional connectivity signal the V-R gap CANNOT provide (per axis, in [0,1]).

    `g_perc[d] = (K_arith - K_eff[d]) / (K_arith - K_harm)` — the fraction of the conductance Wiener
    gap the ACTUAL geometry uses up toward the series/percolation floor along axis d. ~0 for a
    scattered cluster (conduction stays near the parallel value), ~1 for a soft seam percolating
    normal to d (conduction forced to the series floor). A seam and its matched control share
    `K_arith`,`K_harm` (same fractions) AND the V-R gap, yet `g_perc` separates them: this is the
    connectivity information the fraction-only descriptor lacks. Graded (smooth through the
    percolation threshold) and directional. Returns (3,).
    """
    K_eff = directional_conductance(grid, materials, use_gpu=use_gpu)
    K_arith, K_harm = wiener_bounds(grid, materials)
    denom = max(K_arith - K_harm, 1e-300)
    return np.clip((K_arith - K_eff) / denom, 0.0, 1.0)


def spanning_cluster_fraction(cell, axis=None, connectivity=3):
    """Graded discrete companion to the boolean trigger: fraction of the SOFT phase that belongs to
    a face-spanning cluster. 0 if nothing spans; -> 1 for a fully connected seam. Uses 26-connectivity
    by default (the hardened rule), so it registers thin diagonal soft paths a 6-connectivity test
    misses. Reuses `violent_cells.spanning_cluster_fraction`.
    """
    return vc.spanning_cluster_fraction(cell.grid, cell.materials, axis=axis, connectivity=connectivity)


def spanning_fraction_loadplane(grid, materials, connectivity=3):
    """Soft-phase spanning fraction in the LOAD plane: max over x- and z-span (cells are y-extruded,
    so a y-span is trivial and excluded). A topological connectivity measure used as a REGIME-AWARE
    BACKSTOP (not a descriptor channel): with 26-connectivity it catches thin diagonal soft paths the
    6-rule misses, but it is rule/threshold-dependent — on DENSE soft fractions a 26-connected random
    scatter can spuriously span (the 26-conn site threshold ~0.20 sits below a thick seam's fraction),
    which is exactly why the *conductance* residual (physics, not topology) is the descriptor signal.
    """
    return max(vc.spanning_cluster_fraction(grid, materials, axis=0, connectivity=connectivity),
               vc.spanning_cluster_fraction(grid, materials, axis=2, connectivity=connectivity))


def percolates_xz_hard(cell, connectivity=3):
    """Hardened backstop: 26-connectivity span across x OR z (the load plane). Catches thin/diagonal
    soft paths that the default 6-connectivity `percolates_xz` misses (removing the `thickness=3`
    crutch). Returns bool.
    """
    return bool(vc.percolates(cell.grid, cell.materials, axis=0, connectivity=connectivity)
                or vc.percolates(cell.grid, cell.materials, axis=2, connectivity=connectivity))


if __name__ == "__main__":
    from dns_elasticity_3d import effective_stiffness, _HAS_GPU
    np.set_printoptions(precision=4, suppress=True)
    print(f"DNS backend: {'GPU (cupy CG)' if _HAS_GPU else 'CPU (sparse LU)'}")

    n = 16
    angles = [0, 15, 30, 45, 60, 75, 90]

    # 1) the connectivity trigger: percolating seam True at EVERY angle; matched control False.
    # 2) gap is connectivity-blind: EXACTLY identical for a seam and its matched shuffled control
    #    (same fractions); and only mildly orientation-dependent (the discretized seam's voxel count
    #    drifts a little with angle — the airtight claim is the exact seam==control identity).
    det, fp, gaps = [], [], []
    for th in angles:
        seam = seam_cell_at(n, th, thickness=3, contrast=60.0)
        ctrl = shuffled_control(seam, seed=th)
        det.append(percolates_xz(seam))
        fp.append(percolates_xz(ctrl))
        g_seam, g_ctrl = gap_vector(seam), gap_vector(ctrl)
        assert np.array_equal(g_seam, g_ctrl), "gap must be IDENTICAL (same fractions) seam vs control"
        gaps.append(g_seam)
    gaps = np.array(gaps)
    gap_drift = float(np.abs(gaps - gaps.mean(0)).max() / gaps.mean())
    print(f"1) trigger: percolating-seam detection {sum(det)}/{len(angles)}; "
          f"matched-control false-positives {sum(fp)}/{len(angles)}")
    print(f"2) gap blindness: seam==control EXACTLY at every angle; "
          f"across-angle gap drift = {gap_drift:.1%} (discretization only)")
    assert all(det) and not any(fp), "trigger must catch all seams and no controls"

    # 3) DNS sanity (one axis-aligned + one off-axis): the seam is far softer along its normal than
    #    the matched control, and the off-axis best-axis proxy error >> the axis-aligned one.
    for th in (0, 45):
        seam = seam_cell_at(n, th, thickness=3, contrast=60.0)
        ctrl = shuffled_control(seam, seed=100 + th)
        C_seam = effective_stiffness(seam.grid, seam.materials)
        C_ctrl = effective_stiffness(ctrl.grid, ctrl.materials)
        E_seam = uniaxial_modulus(C_seam, seam_normal(th))
        E_ctrl = uniaxial_modulus(C_ctrl, seam_normal(th))
        perr, axis = best_axis_proxy_error(seam, C_seam)
        print(f"3) theta={th:2d}: E_normal seam={E_seam:.3f} vs control={E_ctrl:.3f} "
              f"(ratio {E_seam/E_ctrl:.2f}); best-axis proxy error={perr:.3f} (axis {axis})")
        assert E_seam < E_ctrl, "percolating seam must be softer than the matched control"

    # 4) THE GRADED FIX — the conductance residual g_perc sees what the gap cannot, under IDENTICAL
    #    fractions (identical Wiener bounds AND identical V-R gap): the percolating seam is strictly
    #    more conductance-penalized than its matched scattered control at EVERY orientation. (Note the
    #    control's g_perc stays low even where a 26-conn topological span would FALSE-POSITIVE on it —
    #    the physics signal does not over-count dense scatter; the boolean span would.)
    print("\n4) graded conductance residual g_perc (seam vs matched control, identical fractions):")
    for th in (0, 45, 90):
        seam = seam_cell_at(n, th, thickness=3, contrast=60.0)
        ctrl = shuffled_control(seam, seed=200 + th)
        gp_s = connectivity_residual(seam.grid, seam.materials)
        gp_c = connectivity_residual(ctrl.grid, ctrl.materials)
        assert np.allclose(wiener_bounds(seam.grid, seam.materials),
                           wiener_bounds(ctrl.grid, ctrl.materials)), "Wiener bounds match (same f)"
        print(f"   theta={th:2d}: g_perc seam max {gp_s.max():.3f} {np.round(gp_s,3)} vs "
              f"control max {gp_c.max():.3f} {np.round(gp_c,3)}")
        assert gp_s.max() > gp_c.max(), "seam must be more conductance-penalized than its matched control"

    # 5) homogeneous-cell sanity: K_eff == phase conductivity exactly; g_perc == 0.
    homo = seam_cell_at(n, 0, thickness=3, contrast=1.0)             # contrast 1 -> single phase
    K0 = directional_conductance(homo.grid, homo.materials)
    print(f"5) homogeneous K_eff = {np.round(K0,4)} (expect ~10 = E_wood, isotropic)")
    assert np.allclose(K0, 10.0, rtol=1e-3), "homogeneous conductance must equal the phase value"

    # 6) hardened 26-connectivity backstop catches a THIN diagonal seam the 6-connectivity rule misses,
    #    with no control false-positive (a thickness-1 seam is sub-threshold for a 26-conn scatter).
    thin = seam_cell_at(n, 45, thickness=1, contrast=60.0)
    thin_ctrl = shuffled_control(thin, seed=303)
    print(f"6) thin 45deg seam (thickness 1): 6-conn percolates_xz={percolates_xz(thin)} -> "
          f"26-conn percolates_xz_hard={percolates_xz_hard(thin)}; control 26-conn={percolates_xz_hard(thin_ctrl)}")
    assert percolates_xz_hard(thin) and not percolates_xz_hard(thin_ctrl), \
        "26-conn backstop must catch the thin diagonal seam and not its matched control"
    print("\nALL percolation self-checks PASSED")
