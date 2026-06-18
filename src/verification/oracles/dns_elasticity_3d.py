"""
DNS micro-solver: 3D voxel periodic homogenization (the keystone oracle, protocol §7).

This is the independent ground truth for V0.1 (and reused by V0.2/V1.3/V2.2/V2.3):
a direct fine-scale linear-elasticity solve of a fully-resolved heterogeneous unit
cell, returning the TRUE effective 6x6 stiffness tensor. It is deliberately slow,
simple, and unshippable — its only job is to be trusted so the cheap Voigt-Reuss
proxy never has to be trusted on faith.

Method (standard computational homogenization, periodic BC):
  - one trilinear hex (H8) element per voxel, 3 DOF/node, unit element size.
  - periodic boundary conditions enforced by wrap-around node identification
    (node id = corner index modulo n on each axis) -> exactly n^3 unique nodes.
  - macro field chi0_a = E0_a . x for the 6 unit Voigt strains a = 11,22,33,23,13,12.
  - solve K a_a = -K chi0_a for the periodic fluctuation (one node pinned to remove
    the rigid-translation nullspace); corrected field w_a = chi0_a + a_a.
  - C_eff[a,b] = (1/V) w_a^T K w_b   (energy form; exact for homogeneous cells).

Backend: the SPD reduced system is solved on the GPU (RTX-class) via Jacobi-
preconditioned conjugate gradient (`cupy`), which is matrix-free-fast and sidesteps
the catastrophic fill-in a 3D sparse *direct* factorization suffers. Falls back to a
CPU sparse direct solve if cupy is unavailable.

Determinism: element/DOF traversal and triplet assembly are in fixed lexicographic
order. The CPU path is bit-reproducible; the GPU CG path is reproducible to the CG
tolerance (a declared tolerance-regime per protocol V0.5, not bit-exact). Voigt
convention matches homogenization.py.
"""
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from homogenization import isotropic_stiffness

# Local strain "localization" of a solved cell: eps_loc[e] (6x6) maps a macroscopic
# Voigt strain vector to the element's centroid total strain (column a = response to
# unit macro strain a). With phases + C_phases it reconstructs the local stress field
# under ANY macro load from the single set of 6 unit-strain solves. (Used by V0.2.)
Localization = namedtuple("Localization", ["eps_loc", "phases", "C_phases"])

# Optional GPU backend (cupy). Detected once at import.
try:
    import cupy as _cp
    import cupyx.scipy.sparse as _csp
    import cupyx.scipy.sparse.linalg as _cspla
    _HAS_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_GPU = False

CG_RTOL = 1e-11        # CG relative tolerance (sets the GPU determinism regime)
CG_MAXITER = 50000

# 2-point Gauss rule on [-1,1]^3.
_G = 1.0 / np.sqrt(3.0)
_GAUSS = np.array([(a, b, c) for a in (-_G, _G) for b in (-_G, _G) for c in (-_G, _G)])

# Local node natural coordinates, node index = a + 2b + 4c  (a,b,c in {0,1} -> +-1).
_NAT = np.array([(2 * (i & 1) - 1, 2 * ((i >> 1) & 1) - 1, 2 * ((i >> 2) & 1) - 1)
                 for i in range(8)], dtype=float)

# Unit-Voigt-strain macro tensors E0_a (3x3, symmetric; engineering shear = 1).
_E0 = np.zeros((6, 3, 3))
_E0[0, 0, 0] = _E0[1, 1, 1] = _E0[2, 2, 2] = 1.0
_E0[3, 1, 2] = _E0[3, 2, 1] = 0.5   # 23
_E0[4, 0, 2] = _E0[4, 2, 0] = 0.5   # 13
_E0[5, 0, 1] = _E0[5, 1, 0] = 0.5   # 12


def _B_and_detJ(xi):
    """Strain-displacement matrix B (6x24) and detJ at natural point xi, unit element."""
    x, y, z = xi
    # shape-function natural derivatives (8x3)
    dN = np.empty((8, 3))
    dN[:, 0] = 0.125 * _NAT[:, 0] * (1 + _NAT[:, 1] * y) * (1 + _NAT[:, 2] * z)
    dN[:, 1] = 0.125 * (1 + _NAT[:, 0] * x) * _NAT[:, 1] * (1 + _NAT[:, 2] * z)
    dN[:, 2] = 0.125 * (1 + _NAT[:, 0] * x) * (1 + _NAT[:, 1] * y) * _NAT[:, 2]
    # unit element side 1 -> Jacobian = 0.5 I, dN/dx = 2 dN/dxi, detJ = 1/8
    dNx = 2.0 * dN
    detJ = 0.125
    B = np.zeros((6, 24))
    B[0, 0::3] = dNx[:, 0]
    B[1, 1::3] = dNx[:, 1]
    B[2, 2::3] = dNx[:, 2]
    B[3, 1::3] = dNx[:, 2]; B[3, 2::3] = dNx[:, 1]
    B[4, 0::3] = dNx[:, 2]; B[4, 2::3] = dNx[:, 0]
    B[5, 0::3] = dNx[:, 1]; B[5, 1::3] = dNx[:, 0]
    return B, detJ


def element_stiffness(C):
    """24x24 H8 element stiffness for constitutive matrix C (6x6), unit element."""
    Ke = np.zeros((24, 24))
    for xi in _GAUSS:
        B, detJ = _B_and_detJ(xi)
        Ke += (B.T @ C @ B) * detJ      # Gauss weight = 1
    return Ke


def _edof(n):
    """(n^3, 24) global DOF indices per element, periodic wrap-around connectivity."""
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()          # lexicographic element order
    edof = np.empty((n ** 3, 24), dtype=np.int64)
    for loc in range(8):
        a, b, c = loc & 1, (loc >> 1) & 1, (loc >> 2) & 1
        node = (((ii + a) % n) * n + ((jj + b) % n)) * n + ((kk + c) % n)
        edof[:, 3 * loc:3 * loc + 3] = 3 * node[:, None] + np.array([0, 1, 2])
    return edof


def _assemble(n, phase_grid, Ke_phases):
    """Assemble the periodic global stiffness K (sparse) in fixed order."""
    edof = _edof(n)
    phases = phase_grid.ravel(order="C")                     # matches lexicographic order
    rows = np.repeat(edof, 24, axis=1).reshape(-1)
    cols = np.tile(edof, (1, 24)).reshape(-1)
    vals = np.concatenate([Ke_phases[p].reshape(-1) for p in phases])
    ndof = 3 * n ** 3
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsc()
    return K, edof


def _element_true_coords(n):
    """(n^3, 8, 3) TRUE (un-wrapped) corner coordinates per element.

    The DOF map (edof) wraps around for periodicity, but the macro field E.x must
    use each element's real corner positions so boundary-crossing elements carry
    the correct uniform strain rather than a spurious jump.
    """
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    base = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(float)  # (n^3,3)
    coords = np.empty((n ** 3, 8, 3))
    for loc in range(8):
        off = np.array([loc & 1, (loc >> 1) & 1, (loc >> 2) & 1], dtype=float)
        coords[:, loc, :] = base + off
    return coords


def _solve_free(Kff, F_free, use_gpu=True):
    """Solve Kff @ X = F_free for all 6 RHS columns. SPD system.

    GPU path: Jacobi-preconditioned CG on the device (cupy). CPU path: one sparse
    LU factorization reused across the 6 right-hand sides.
    """
    if use_gpu and _HAS_GPU:
        Kg = _csp.csr_matrix(Kff.astype(np.float64))
        diag = Kg.diagonal()
        Minv = _cspla.LinearOperator(Kg.shape, matvec=lambda v: v / diag)
        X = np.empty_like(F_free)
        for a in range(F_free.shape[1]):
            b = _cp.asarray(F_free[:, a])
            x, info = _cspla.cg(Kg, b, M=Minv, rtol=CG_RTOL, atol=0.0, maxiter=CG_MAXITER)
            if info != 0:
                raise RuntimeError(f"GPU CG did not converge (rhs {a}, info={info})")
            X[:, a] = _cp.asnumpy(x)
        return X
    lu = spla.splu(Kff.tocsc())
    X = np.empty_like(F_free)
    for a in range(F_free.shape[1]):
        X[:, a] = lu.solve(F_free[:, a])
    return X


def effective_stiffness(phase_grid, phase_materials, use_gpu=True, return_localization=False):
    """TRUE effective 6x6 stiffness of a heterogeneous unit cell via DNS.

    phase_grid     : (n,n,n) int array of phase ids (cubic cell).
    phase_materials: list of (E, nu) indexed by phase id.
    Returns C_eff (6,6), symmetrized. If return_localization, returns
    (C_eff, Localization) where the localization recovers the local stress field
    under any macro load without re-solving.
    """
    phase_grid = np.asarray(phase_grid)
    n = phase_grid.shape[0]
    assert phase_grid.shape == (n, n, n), "cell must be cubic"
    ndof = 3 * n ** 3

    Ke_phases = np.stack(
        [element_stiffness(isotropic_stiffness(E, nu)) for (E, nu) in phase_materials])
    K, edof = _assemble(n, phase_grid, Ke_phases)
    phases = phase_grid.ravel(order="C")
    Ke = Ke_phases[phases]                                     # (n^3, 24, 24)

    # per-element macro field chi0_e (n^3, 24, 6) from TRUE corner coords
    coords = _element_true_coords(n)                           # (n^3, 8, 3)
    chi0 = np.empty((n ** 3, 24, 6))
    for a in range(6):
        u = coords @ _E0[a].T                                  # (n^3, 8, 3)
        chi0[:, :, a] = u.reshape(n ** 3, 24)

    # global rhs F = -K chi0, assembled per element (scatter-add over periodic DOFs)
    fe = -np.einsum("eij,ejb->eib", Ke, chi0)                  # (n^3, 24, 6)
    F = np.zeros((ndof, 6))
    np.add.at(F, edof.reshape(-1), fe.reshape(-1, 6))

    # solve K a = F for the periodic fluctuation; pin node 0 to kill rigid translation
    free = np.arange(3, ndof)
    Kff = K[free][:, free]
    a_fluct = np.zeros((ndof, 6))
    a_fluct[free, :] = _solve_free(Kff, F[free, :], use_gpu=use_gpu)

    # corrected field w_e = chi0_e + a[edof_e]; effective tensor via element energy
    w = chi0 + a_fluct[edof]                                   # (n^3, 24, 6)
    Kw = np.einsum("eij,ejb->eib", Ke, w)                      # (n^3, 24, 6)
    C = np.einsum("eia,eib->ab", w, Kw) / float(n ** 3)
    C = 0.5 * (C + C.T)

    if not return_localization:
        return C
    # localization: centroid total strain per element, eps_loc[e] = B0 @ w_e
    B0, _ = _B_and_detJ((0.0, 0.0, 0.0))
    eps_loc = np.einsum("ij,ejb->eib", B0, w)                  # (n^3, 6, 6)
    return C, Localization(eps_loc=eps_loc, phases=phases,
                           C_phases=[isotropic_stiffness(E, nu) for (E, nu) in phase_materials])


if __name__ == "__main__":
    import time
    np.set_printoptions(precision=3, suppress=True)
    print(f"backend: {'GPU (cupy CG)' if _HAS_GPU else 'CPU (sparse LU)'}\n")

    # 1) homogeneous cell -> C_eff must equal the phase stiffness exactly.
    n = 24
    grid = np.zeros((n, n, n), dtype=np.int64)
    C_true = isotropic_stiffness(10.0, 0.3)
    t = time.time(); C_dns = effective_stiffness(grid, [(10.0, 0.3)]); dt = time.time() - t
    print("1) homogeneous identity:  ||C_dns - C_true|| / ||C_true|| = %.2e   (%.2fs, n=%d)"
          % (np.linalg.norm(C_dns - C_true) / np.linalg.norm(C_true), dt, n))

    # 2) high-contrast two-layer stack -> must match the Backus closed form.
    from analytic import laminate_stiffness
    f = [0.5, 0.5]; moduli = [10.0, 0.1]; nus = [0.3, 0.3]; axis = 2   # contrast 100
    edge = n // 2
    grid = np.zeros((n, n, n), dtype=np.int64); grid[:, :, edge:] = 1
    t = time.time(); C_dns = effective_stiffness(grid, list(zip(moduli, nus))); dt = time.time() - t
    C_ana = laminate_stiffness(f, moduli, nus, axis)
    print("2) two-layer vs Backus:   ||C_dns - C_ana|| / ||C_ana|| = %.2e   (%.2fs)"
          % (np.linalg.norm(C_dns - C_ana) / np.linalg.norm(C_ana), dt))
    print("   across-layer normal C33: dns=%.4f  analytic=%.4f" % (C_dns[2, 2], C_ana[2, 2]))
    print("   in-plane normal     C11: dns=%.4f  analytic=%.4f" % (C_dns[0, 0], C_ana[0, 0]))
    print("   in-plane shear      C66: dns=%.4f  analytic=%.4f" % (C_dns[5, 5], C_ana[5, 5]))

    # 3) reproducibility on a repeat run (GPU CG: to CG tolerance, not bit-exact).
    C_a = effective_stiffness(grid, list(zip(moduli, nus)))
    C_b = effective_stiffness(grid, list(zip(moduli, nus)))
    print("3) reproducibility: max rel diff on repeat = %.2e  (CG rtol = %.0e)"
          % (np.abs(C_a - C_b).max() / np.abs(C_a).max(), CG_RTOL))
