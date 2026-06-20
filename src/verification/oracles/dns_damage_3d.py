"""
Damage/softening DNS oracle — the VIOLENT-REGIME ground truth (protocol §7, V2.1/Risk #1).

The linear DNS in `dns_elasticity_3d.py` is the keystone oracle, but Voigt-Reuss is a
*linear-elasticity* theorem: in the large-deformation / actively-fracturing regime the
analytic bound is not merely loose, it is **invalid** (the true effective response leaves
the [Reuss, Voigt] bracket as damage localizes). V2.1 lives exactly there. This module
extends the proven linear periodic-homogenization machinery into an **incremental
secant-damage** solver, producing genuine softening ground truth the cheap proxy CANNOT
bracket — the regime where the architecture admits "there is no cheap analytic bound" and
the RVE<->surrogate handoff must decide pay-for-RVE vs trust-surrogate.

Model (standard local isotropic continuum damage, monotone / dissipative):
  - One scalar damage d_e in [0, d_max] per voxel; secant stiffness C_e = (1 - d_e) C_e^0.
  - Damage is driven by an energy-norm equivalent strain  eps_eq = sqrt(eps . M . eps),
    M = diag(1,1,1,1/2,1/2,1/2)  (engineering-shear consistent).
  - History variable  kappa_e = max over the load path of eps_eq  (irreversibility).
  - Exponential softening law  d(kappa) = 1 - (k0/kappa) exp(-(kappa-k0)/(kf-k0))  for
    kappa > k0, capped at d_max < 1 (keeps K non-singular). Monotone in kappa ->
    d. monotone -> dissipation  D = psi0 * d_dot >= 0  (free energy psi = (1-d) psi0).
  - A macro strain is ramped in `n_increments` along a load direction; each increment is
    fixed-point iterated (re-solve with updated secant stiffness) to convergence.

Reuses, WITHOUT editing, the linear solver's element/assembly/solve internals
(`element_stiffness`, `_edof`, `_element_true_coords`, `_solve_free`, `_B_and_detJ`, `_E0`)
and `failure.von_mises`. Backend = the same cupy GPU CG path (CPU sparse-LU fallback).

Determinism: fixed element order, fixed increment schedule, seeded microstructure. CPU path
bit-reproducible; GPU CG reproducible to CG tolerance (declared regime, per V0.5).
Voigt convention [11,22,33,23,13,12], engineering shear, matches homogenization.py.
"""
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from homogenization import isotropic_stiffness, voigt_bound, reuss_bound
from dns_elasticity_3d import (
    element_stiffness, _edof, _element_true_coords, _solve_free, _B_and_detJ, _E0, _HAS_GPU,
)

if _HAS_GPU:
    import cupy as _cp
    import cupyx.scipy.sparse as _csp
    import cupyx.scipy.sparse.linalg as _cspla

# Energy-norm metric for the equivalent strain (engineering shear halved).
_M_EPS = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

# Damage solves don't need the keystone oracle's 1e-11: the violent-regime outcome (peak
# strength / dissipation) is insensitive well below this, and warm-starting across the near-
# identical fixed-point iterations makes each solve cheap.
DMG_CG_RTOL = 1e-8


@dataclass
class DamageParams:
    """Constitutive + path parameters for the secant-damage solve."""
    k0: float = 1.2e-3          # equivalent-strain damage onset (kappa_0)
    kf: float = 6.0e-3          # softening modulus param (kf > k0; larger = more ductile)
    d_max: float = 0.999        # damage cap (keeps the secant system non-singular)
    n_increments: int = 24      # load steps to max_strain
    max_strain: float = 6.0e-3  # peak macro strain amplitude along load_dir
    load_dir: int = 0           # Voigt channel loaded (0=uniaxial 11)
    fp_iters: int = 30          # max fixed-point iterations per increment
    fp_tol: float = 1e-5        # convergence on max |delta d| per increment
    use_gpu: bool = True


# Full per-path result. R_true (the "violent-regime effective response") is the bundle
# (C_secant_final, peak_stress, dissipated_energy, residual_modulus); the surrogate
# predicts a scalar reduction of it and the handoff rule is judged on its error.
DamageResult = namedtuple("DamageResult", [
    "strain_curve", "stress_curve",          # macro response along load_dir
    "d_final", "kappa_final",                # per-voxel terminal damage / history
    "C_secant_final",                        # (6,6) secant effective stiffness at path end
    "peak_stress", "peak_strain",            # macro peak load + the strain it occurs at
    "residual_modulus", "dissipated_energy",  # post-peak secant slope + total dissipation
    "C0_linear",                             # (6,6) undamaged effective tensor (linear DNS)
])


def _damage(kappa, p: DamageParams):
    """Exponential-softening damage d(kappa) in [0, d_max], monotone non-decreasing."""
    kappa = np.asarray(kappa, float)
    d = np.zeros_like(kappa)
    m = kappa > p.k0
    d[m] = 1.0 - (p.k0 / kappa[m]) * np.exp(-(kappa[m] - p.k0) / (p.kf - p.k0))
    return np.minimum(np.maximum(d, 0.0), p.d_max)


def _pattern(n, edof):
    """Precompute the fixed COO (rows, cols) once — they never change across the load path."""
    rows = np.repeat(edof, 24, axis=1).reshape(-1)
    cols = np.tile(edof, (1, 24)).reshape(-1)
    return rows, cols


def _assemble_pat(n, Ke_elem, rows, cols):
    """Global stiffness from per-element blocks using a precomputed COO pattern."""
    ndof = 3 * n ** 3
    return sp.coo_matrix((Ke_elem.reshape(-1), (rows, cols)), shape=(ndof, ndof)).tocsc()


def _solve_free_warm(Kff, F_free, x0, rtol):
    """Single-RHS SPD solve, warm-started at x0 (GPU CG; CPU sparse-LU fallback, no warm start)."""
    if _HAS_GPU:
        Kg = _csp.csr_matrix(Kff.astype(np.float64))
        diag = Kg.diagonal()
        Minv = _cspla.LinearOperator(Kg.shape, matvec=lambda v: v / diag)
        b = _cp.asarray(F_free)
        x0g = None if x0 is None else _cp.asarray(x0)
        x, info = _cspla.cg(Kg, b, x0=x0g, M=Minv, rtol=rtol, atol=0.0, maxiter=50000)
        return _cp.asnumpy(x)
    return _solve_free(Kff, F_free[:, None], use_gpu=False)[:, 0]


def _solve_macro(n, Ke_elem, edof, rows, cols, chi0, B0, free, x0, rtol=DMG_CG_RTOL):
    """Solve one applied macro strain (chi0 precomputed); return (centroid strain, free solution).

    The returned free-DOF solution warm-starts the next (near-identical) fixed-point solve, which
    collapses the CG iteration count over a load path.
    """
    ndof = 3 * n ** 3
    fe = -np.einsum("eij,ej->ei", Ke_elem, chi0)                      # (n^3,24)
    F = np.zeros(ndof)
    np.add.at(F, edof.reshape(-1), fe.reshape(-1))
    Kff = _assemble_pat(n, Ke_elem, rows, cols)[free][:, free]
    a_free = _solve_free_warm(Kff, F[free], x0, rtol)
    a = np.zeros(ndof); a[free] = a_free
    eps = np.einsum("ij,ej->ei", B0, chi0 + a[edof])                  # (n^3,6) centroid strain
    return eps, a_free


def _effective_from_Ke(n, Ke_elem, edof, use_gpu):
    """Secant effective 6x6 tensor for a given per-element (damaged) stiffness field."""
    ndof = 3 * n ** 3
    coords = _element_true_coords(n)
    chi0 = np.empty((n ** 3, 24, 6))
    for a in range(6):
        chi0[:, :, a] = (coords @ _E0[a].T).reshape(n ** 3, 24)
    fe = -np.einsum("eij,ejb->eib", Ke_elem, chi0)
    F = np.zeros((ndof, 6))
    np.add.at(F, edof.reshape(-1), fe.reshape(-1, 6))
    rows, cols = _pattern(n, edof)
    K = _assemble_pat(n, Ke_elem, rows, cols)
    free = np.arange(3, ndof)
    Kff = K[free][:, free]
    a_fl = np.zeros((ndof, 6))
    a_fl[free, :] = _solve_free(Kff, F[free, :], use_gpu=use_gpu)
    w = chi0 + a_fl[edof]
    Kw = np.einsum("eij,ejb->eib", Ke_elem, w)
    C = np.einsum("eia,eib->ab", w, Kw) / float(n ** 3)
    return 0.5 * (C + C.T)


def run_path(phase_grid, phase_materials, p: DamageParams = DamageParams()):
    """Ramp a macro strain to failure on a heterogeneous cell; return the violent response.

    phase_grid     : (n,n,n) int phase ids.
    phase_materials: list of (E, nu) per phase id.
    Returns a DamageResult; `R_true` consumers read C_secant_final + the scalar outcomes.
    """
    phase_grid = np.asarray(phase_grid)
    n = phase_grid.shape[0]
    assert phase_grid.shape == (n, n, n), "cell must be cubic"
    edof = _edof(n)
    phases = phase_grid.ravel(order="C")

    C_phases = np.stack([isotropic_stiffness(E, nu) for (E, nu) in phase_materials])  # (P,6,6)
    Ke_base = np.stack([element_stiffness(C) for C in C_phases])                      # (P,24,24)
    Ke0_elem = Ke_base[phases]                                                        # (n^3,24,24)
    Cph_elem = C_phases[phases]                                                       # (n^3,6,6)

    # precompute the path-invariant assembly pattern, macro field, and free DOFs ONCE
    rows, cols = _pattern(n, edof)
    free = np.arange(3, 3 * n ** 3)
    B0, _ = _B_and_detJ((0.0, 0.0, 0.0))
    macroT = np.einsum("a,aij->ij", np.eye(6)[p.load_dir], _E0)
    chi0_unit = (_element_true_coords(n) @ macroT.T).reshape(n ** 3, 24)   # for unit amplitude

    # undamaged linear effective tensor (the bound is computed against this state)
    C0 = _effective_from_Ke(n, Ke0_elem, edof, p.use_gpu)

    kappa = np.zeros(n ** 3)
    d = np.zeros(n ** 3)
    strain_curve, stress_curve = [], []
    dissipated = 0.0
    psi0_prev = np.zeros(n ** 3)
    d_prev = np.zeros(n ** 3)
    x0 = None                                                              # CG warm-start carrier

    for k in range(1, p.n_increments + 1):
        amp = p.max_strain * k / p.n_increments
        chi0 = amp * chi0_unit
        # fixed-point on the secant stiffness at this (fixed) applied strain
        for _ in range(p.fp_iters):
            Ke_elem = (1.0 - d)[:, None, None] * Ke0_elem
            eps, x0 = _solve_macro(n, Ke_elem, edof, rows, cols, chi0, B0, free, x0)
            eps_eq = np.sqrt((eps ** 2 * _M_EPS).sum(axis=1))
            kappa_new = np.maximum(kappa, eps_eq)
            d_new = _damage(kappa_new, p)
            if np.max(np.abs(d_new - d)) < p.fp_tol:
                kappa, d = kappa_new, d_new
                break
            kappa, d = kappa_new, d_new
        # macro stress (volume average of damaged stress) and dissipation increment
        sigma_e = (1.0 - d)[:, None] * np.einsum("eij,ej->ei", Cph_elem, eps)
        sigma_avg = sigma_e.mean(axis=0)
        psi0 = 0.5 * np.einsum("ei,ei->e", eps, np.einsum("eij,ej->ei", Cph_elem, eps))
        dissipated += float(np.sum(0.5 * (psi0 + psi0_prev) * (d - d_prev)))  # trapz, >=0
        psi0_prev, d_prev = psi0, d.copy()
        strain_curve.append(amp)
        stress_curve.append(float(sigma_avg[p.load_dir]))

    strain_curve = np.array(strain_curve)
    stress_curve = np.array(stress_curve)
    C_sec = _effective_from_Ke(n, (1.0 - d)[:, None, None] * Ke0_elem, edof, p.use_gpu)

    ipk = int(np.argmax(stress_curve))
    peak_stress = float(stress_curve[ipk])
    peak_strain = float(strain_curve[ipk])
    # post-peak secant slope (residual modulus): last point vs peak
    if ipk < len(strain_curve) - 1:
        residual_modulus = float((stress_curve[-1] - stress_curve[ipk])
                                 / (strain_curve[-1] - strain_curve[ipk]))
    else:
        residual_modulus = float(stress_curve[-1] / strain_curve[-1])

    return DamageResult(
        strain_curve=strain_curve, stress_curve=stress_curve,
        d_final=d.reshape(n, n, n), kappa_final=kappa.reshape(n, n, n),
        C_secant_final=C_sec, peak_stress=peak_stress, peak_strain=peak_strain,
        residual_modulus=residual_modulus, dissipated_energy=dissipated, C0_linear=C0,
    )


def vr_brackets(phase_grid, phase_materials):
    """Voigt/Reuss diagonal bounds on the UNDAMAGED cell (the bound that goes invalid)."""
    phases = np.asarray(phase_grid).ravel()
    P = len(phase_materials)
    frac = np.bincount(phases, minlength=P) / phases.size
    C_phases = [isotropic_stiffness(E, nu) for (E, nu) in phase_materials]
    Cv = voigt_bound(frac, C_phases)
    Cr = reuss_bound(frac, C_phases)
    return np.diag(Cv).copy(), np.diag(Cr).copy()


if __name__ == "__main__":
    import time
    np.set_printoptions(precision=4, suppress=True)
    print("backend check via dns_elasticity_3d (_solve_free) — running self-checks\n")

    # 1) HOMOGENEOUS bar, uniaxial strain: field is uniform, so DNS macro stress must equal
    #    the closed-form (1 - d(eps_eq(eps))) * C11 * eps at every increment.
    n = 10
    grid = np.zeros((n, n, n), dtype=np.int64)
    E, nu = 10.0, 0.3
    p = DamageParams(n_increments=20, max_strain=6e-3, use_gpu=True)
    t = time.time(); res = run_path(grid, [(E, nu)], p); dt = time.time() - t
    C11 = isotropic_stiffness(E, nu)[0, 0]
    eps_eq_path = res.strain_curve * np.sqrt(_M_EPS[0])     # uniaxial: eps_eq = |eps11|
    d_path = _damage(eps_eq_path, p)
    sigma_closed = (1.0 - d_path) * C11 * res.strain_curve
    rel = np.abs(res.stress_curve - sigma_closed) / np.max(np.abs(sigma_closed))
    print("1) homogeneous softening bar vs closed form:")
    print(f"   max rel stress error = {rel.max():.2e}   ({dt:.2f}s, n={n})")
    print(f"   peak stress={res.peak_stress:.5f} at eps={res.peak_strain:.2e}; "
          f"softening tail present={res.stress_curve[-1] < res.peak_stress}")
    assert rel.max() < 1e-3, "homogeneous damage curve must match closed form"

    # 2) Monotone damage + non-negative dissipation (thermodynamic admissibility).
    print("2) irreversibility / dissipation:")
    print(f"   d in [0, {res.d_final.max():.3f}]  (cap {p.d_max}); "
          f"dissipated energy = {res.dissipated_energy:.3e}")
    assert res.dissipated_energy >= 0.0, "dissipation must be >= 0"
    assert res.d_final.max() > 0.0, "bar must damage under the ramp"

    # 3) VIOLENT-REGIME point: a char-wedge cell leaves the Voigt-Reuss bracket as it softens.
    import cells
    c = cells.char_wedge_cell(n=12, depth=0.6, contrast=60.0)
    res2 = run_path(c.grid, c.materials, DamageParams(n_increments=18, max_strain=7e-3))
    dv, dr = vr_brackets(c.grid, c.materials)
    d0 = np.diag(res2.C0_linear); ds = np.diag(res2.C_secant_final)
    ld = p.load_dir
    inside0 = dr[ld] - 1e-9 <= d0[ld] <= dv[ld] + 1e-9
    inside_s = dr[ld] - 1e-9 <= ds[ld] <= dv[ld] + 1e-9
    print("3) char wedge — Voigt-Reuss validity along load axis (channel %d):" % ld)
    print(f"   Reuss={dr[ld]:.3f}  C0_linear={d0[ld]:.3f}  Voigt={dv[ld]:.3f}  -> in-bracket={inside0}")
    print(f"   C_secant(damaged)={ds[ld]:.3f}  -> in-bracket={inside_s}")
    print(f"   => undamaged bound valid ({inside0}); damaged response below Reuss "
          f"(bound INVALID) = {not inside_s}")
    assert inside0, "undamaged linear effective stiffness must lie in the V-R bracket"
    assert not inside_s, "damaged response should fall BELOW Reuss — the violent regime"
    print("\nALL dns_damage_3d self-checks PASSED")
