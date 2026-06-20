"""
3D linear-elastic boundary-value FE solver + load-case domains + the SIMP topology-
optimization oracle — the independent ground truth for V1.7 (protocol §V1.7; Decision #22;
ARCHITECTURE §III.7 "the morphogenetic scaffold").

V1.7 falsifies the architecture's "origin of structure" claim: that a *skeleton is a
precipitate* — stress-driven deposition under load (Wolff's law as a generative rule, the
same rule the tree uses for reaction wood) — not an authored atlas. The cheap production
mechanism (Wolff SED remodeling, in `wolff.py`) is judged against the mature, trusted
global optimizer implemented here: **SIMP compliance minimization**. If Wolff yields
disconnected / non-load-bearing junk, or compliance far worse than SIMP at matched volume,
the mechanism fails → REDESIGN (use topology optimization directly as the precipitation
operator). This module is that oracle, and it also houses the **shared FE solver + domains**
both paths use, so the comparison is physics-identical and therefore fair.

Why a new FE solver (not `dns_elasticity_3d.py`)
------------------------------------------------
`dns_elasticity_3d.py` is *periodic homogenization* (wrap-around BCs, 6 unit-strain solves,
energy-form effective tensor). V1.7 needs a *boundary-value problem*: real Dirichlet
supports + applied loads + a compliance scalar. We REUSE its trusted, determinism-vetted
pieces — `element_stiffness` (H8, the same `_NAT` local-node convention) and the cupy-CG /
CPU-LU backend pattern — and `isotropic_stiffness` from `homogenization.py`; only the
assembly (non-periodic, with BCs) and the SIMP optimizer are new.

Method
------
- Voxel grid `dims=(nx,ny,nz)` of trilinear H8 elements, 3 DOF/node, unit element size.
  Node id `nid(i,j,k)=(i*(ny+1)+j)*(nz+1)+k`; element order is C-order of `(nx,ny,nz)`,
  matching `rho.ravel()`. Local-node offsets follow `dns_elasticity_3d._NAT`
  (`loc=a+2b+4c`), so `element_stiffness` plugs in unchanged.
- SIMP material interpolation `E(ρ)=Emin+ρ^p·(E0−Emin)`; since isotropic stiffness is linear
  in `E` at fixed `ν`, the per-element matrix is `E(ρ)·Ke0` with one precomputed `Ke0`.
- SPD reduced system (free DOFs) solved by Jacobi-preconditioned CG on `cupy` (the RTX path),
  CPU sparse-LU fallback. CG tolerance regime per V0.5 (reproducible to tolerance, not bit).
- SIMP loop: solve → compliance `c=Fᵀu` + sensitivity → Sigmund sensitivity filter → OC
  (optimality-criteria) bisection update, fixed traversal order → deterministic.

Load-case domains (design-INDEPENDENT loads → clean, apples-to-apples comparison)
--------------------------------------------------------------------------------
- `cantilever` : x=0 face fully fixed; downward (−z) load spread along the free-end mid edge.
  The textbook topology-opt benchmark, used to calibrate that the comparison machinery is
  sound on a case with a known answer.
- `creature`   : two "feet" pads on the bottom face fully fixed; a **gravity body load** = a
  fixed nominal tissue mass (`ρ_tissue≡1` over the whole domain) × `g`, lumped to nodes in
  −z. The Decision-#22 biped. Self-weight-follows-design is deliberately avoided: the tissue
  mass is fixed and the *skeleton* precipitates to carry it, which keeps the load
  design-independent (so SIMP minimizes a well-posed compliance and the matched-volume
  comparison is exact). `support_alpha` is the **seraph radiance support law-domain**
  (ARCH §III.9 / Decision #26): an upward body potential scaling the net gravity by
  `(1−support_alpha)`; `support_alpha=1` ⇒ zero net load ⇒ near-nothing precipitates.

Pure numpy/scipy + optional cupy. Self-contained beyond the two reused helpers.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import convolve, label, generate_binary_structure

from homogenization import isotropic_stiffness
from dns_elasticity_3d import element_stiffness

# Optional GPU backend (cupy) — same detection/pattern as dns_elasticity_3d.
try:
    import cupy as _cp
    import cupyx.scipy.sparse as _csp
    import cupyx.scipy.sparse.linalg as _cspla
    _HAS_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_GPU = False

CG_RTOL = 1e-10        # CG relative tolerance (sets the GPU determinism regime, per V0.5)
CG_MAXITER = 30000


@dataclass(frozen=True)
class FEParams:
    E0: float = 1.0        # solid Young's modulus
    Emin: float = 1e-9     # void (Ersatz) modulus — keeps the system non-singular
    nu: float = 0.3        # Poisson ratio
    penal: float = 3.0     # SIMP penalization exponent
    rmin: float = 1.5      # density/sensitivity filter radius (in elements)


# ---------------- mesh & connectivity bookkeeping ----------------

def n_dof(dims):
    nx, ny, nz = dims
    return 3 * (nx + 1) * (ny + 1) * (nz + 1)


def node_id(dims, i, j, k):
    nx, ny, nz = dims
    return (i * (ny + 1) + j) * (nz + 1) + k


def edof_map(dims):
    """(nel, 24) global DOF indices per element; element order = C-order of (nx,ny,nz).

    Local node `loc` has offset (a,b,c)=(loc&1,(loc>>1)&1,(loc>>2)&1) — identical to
    `dns_elasticity_3d._NAT`, so `element_stiffness` rows/cols line up.
    """
    nx, ny, nz = dims
    ex, ey, ez = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    ex, ey, ez = ex.ravel(), ey.ravel(), ez.ravel()
    edof = np.empty((nx * ny * nz, 24), dtype=np.int64)
    for loc in range(8):
        a, b, c = loc & 1, (loc >> 1) & 1, (loc >> 2) & 1
        node = ((ex + a) * (ny + 1) + (ey + b)) * (nz + 1) + (ez + c)
        edof[:, 3 * loc:3 * loc + 3] = 3 * node[:, None] + np.array([0, 1, 2])
    return edof


# ---------------- FE assembly & solve ----------------

def assemble(dims, rho, Ke0, params, edof):
    """Assemble the global stiffness K (CSR) for SIMP density field `rho` (shape dims)."""
    Escale = params.Emin + rho.ravel() ** params.penal * (params.E0 - params.Emin)
    vals = (Escale[:, None, None] * Ke0).reshape(-1)
    rows = np.repeat(edof, 24, axis=1).reshape(-1)
    cols = np.tile(edof, (1, 24)).reshape(-1)
    ndof = n_dof(dims)
    return sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()


def _cg_solve(Kff, b, use_gpu=True):
    """Solve SPD Kff x = b (single RHS). Jacobi-preconditioned CG on GPU, LU on CPU."""
    if use_gpu and _HAS_GPU:
        Kg = _csp.csr_matrix(Kff.astype(np.float64))
        diag = Kg.diagonal()
        Minv = _cspla.LinearOperator(Kg.shape, matvec=lambda v: v / diag)
        x, info = _cspla.cg(Kg, _cp.asarray(b), M=Minv, rtol=CG_RTOL, atol=0.0,
                            maxiter=CG_MAXITER)
        if info != 0:
            raise RuntimeError(f"GPU CG did not converge (info={info})")
        return _cp.asnumpy(x)
    lu = spla.splu(Kff.tocsc())
    return lu.solve(b)


def fe_solve(dims, rho, domain, params, Ke0=None, edof=None, use_gpu=True):
    """Solve the BVP for density `rho` under `domain=(fixed_dofs, F)`.

    Returns (u, compliance). `compliance = Fᵀu = uᵀKu` (twice the strain energy).
    """
    if Ke0 is None:
        Ke0 = element_stiffness(isotropic_stiffness(params.E0, params.nu)) / params.E0
    if edof is None:
        edof = edof_map(dims)
    fixed, F = domain
    K = assemble(dims, rho, Ke0, params, edof)
    ndof = n_dof(dims)
    free = np.setdiff1d(np.arange(ndof), fixed, assume_unique=False)
    u = np.zeros(ndof)
    u[free] = _cg_solve(K[free][:, free], F[free], use_gpu=use_gpu)
    return u, float(F @ u)


def element_energy(dims, u, edof, rho, Ke0, params):
    """Per-element energies (both shaped dims).

    Ue0 = uₑᵀ Ke0 uₑ  (unit-modulus element energy; the SIMP sensitivity kernel AND the
    Wolff strain-energy-density stimulus). Ue = E(ρ)·Ue0 (actual energy; Σ Ue = compliance).
    """
    ue = u[edof]                                   # (nel, 24)
    Ue0 = np.einsum("ei,ij,ej->e", ue, Ke0, ue)
    Escale = params.Emin + rho.ravel() ** params.penal * (params.E0 - params.Emin)
    return Ue0.reshape(dims), (Escale * Ue0).reshape(dims)


# ---------------- density filter & OC update ----------------

def make_filter(rmin):
    """Cone-weight neighbourhood kernel H_ef = max(0, rmin − dist) for the density filter."""
    r = int(np.floor(rmin))
    c = np.arange(-r, r + 1)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return np.maximum(0.0, rmin - np.sqrt(X ** 2 + Y ** 2 + Z ** 2))


def sensitivity_filter(rho, dc, ker, Hs):
    """Sigmund (2001) mesh-independence sensitivity filter."""
    return convolve(rho * dc, ker, mode="constant") / (Hs * np.maximum(1e-3, rho))


def oc_update(rho, dc, volfrac, move=0.2):
    """Optimality-criteria bisection update for a fixed volume fraction (deterministic)."""
    l1, l2 = 0.0, 1e9
    g = np.sqrt(np.maximum(0.0, -dc))
    while (l2 - l1) > 1e-9 * (l1 + l2 + 1e-30):
        lmid = 0.5 * (l1 + l2)
        rnew = np.clip(rho * g / np.sqrt(lmid), rho - move, rho + move)
        rnew = np.clip(rnew, 0.0, 1.0)
        if rnew.mean() > volfrac:
            l1 = lmid
        else:
            l2 = lmid
    return rnew


# ---------------- the SIMP oracle ----------------

def simp_optimize(dims, domain, volfrac, params=FEParams(), n_iter=60, use_gpu=True,
                  rho0=None, tol=1e-3, verbose=False):
    """SIMP compliance minimization — the independent oracle. Returns (rho, history)."""
    Ke0 = element_stiffness(isotropic_stiffness(params.E0, params.nu)) / params.E0
    edof = edof_map(dims)
    ker = make_filter(params.rmin)
    Hs = convolve(np.ones(dims), ker, mode="constant")
    rho = np.full(dims, volfrac) if rho0 is None else rho0.copy()
    hist = []
    for it in range(n_iter):
        u, c = fe_solve(dims, rho, domain, params, Ke0=Ke0, edof=edof, use_gpu=use_gpu)
        Ue0, _ = element_energy(dims, u, edof, rho, Ke0, params)
        dc = -params.penal * rho ** (params.penal - 1) * (params.E0 - params.Emin) * Ue0
        dc = sensitivity_filter(rho, dc, ker, Hs)
        rho = oc_update(rho, dc, volfrac)
        hist.append(c)
        if verbose:
            print(f"  it {it:3d}  c={c:.4e}  vol={rho.mean():.3f}")
        if it > 5 and abs(hist[-2] - hist[-1]) <= tol * abs(hist[-1]):
            break
    return rho, np.array(hist)


# ---------------- design scoring & connectivity ----------------

def binarize(rho, volfrac):
    """Threshold a density field to exactly `volfrac` solid (top-quantile). {0,1} field."""
    thr = np.quantile(rho, 1.0 - volfrac)
    return (rho > thr).astype(float)


def compliance_of(dims, rho, domain, params=FEParams(), use_gpu=True):
    """Compliance Fᵀu of an arbitrary (e.g. binary) design under `domain`."""
    _, c = fe_solve(dims, rho, domain, params, use_gpu=use_gpu)
    return c


def connectivity(rho_bin, support_elem_mask, frac_threshold=0.95):
    """Connectivity diagnostics of a binary solid set (26-connectivity).

    The load-bearing skeleton is the LARGEST 26-connected component (any minor floating
    speckle is not part of it and would be cleaned at export). The structure is "connected
    & load-bearing" when that dominant component holds ≥ `frac_threshold` of the solid
    material AND reaches the supports (a continuous load path exists).

    Returns dict: n_components, frac_in_largest (largest comp / total solid),
    touches_support, connected.
    """
    struct = generate_binary_structure(3, 3)          # 26-connectivity
    solid = rho_bin > 0.5
    lbl, n = label(solid, structure=struct)
    total = int(solid.sum())
    if total == 0:
        return {"n_components": 0, "frac_in_largest": 0.0,
                "touches_support": False, "connected": False}
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    frac = sizes[largest] / total
    touches = bool(((lbl == largest) & support_elem_mask).any())
    return {"n_components": int(n), "frac_in_largest": float(frac),
            "touches_support": touches,
            "connected": bool(frac >= frac_threshold and touches)}


# ---------------- load-case domains ----------------

def build_cantilever(dims, load=1.0):
    """x=0 face fully fixed; downward (−z) load spread along the free-end mid-height edge.

    Returns (domain=(fixed_dofs, F), support_elem_mask).
    """
    nx, ny, nz = dims
    ndof = n_dof(dims)
    fixed = []
    for j in range(ny + 1):
        for k in range(nz + 1):
            n = node_id(dims, 0, j, k)
            fixed += [3 * n, 3 * n + 1, 3 * n + 2]
    F = np.zeros(ndof)
    end_nodes = [node_id(dims, nx, j, nz // 2) for j in range(ny + 1)]
    for n in end_nodes:
        F[3 * n + 2] = -load / len(end_nodes)
    support_mask = np.zeros(dims, dtype=bool)
    support_mask[0, :, :] = True                       # elements against the clamped face
    return (np.array(sorted(set(fixed)), dtype=np.int64), F), support_mask


def build_creature(dims, g=1.0, support_alpha=0.0, rho_tissue=1.0, foot=None,
                   ability_load=0.0):
    """Two feet pads on the bottom (k=0) fixed; gravity body load (−z) over the whole domain.

    Net gravity is scaled by (1 − support_alpha) — the seraph radiance support law-domain.
    Body load uses a FIXED nominal tissue mass (ρ_tissue over the full domain), so the load
    is design-independent. Optional `ability_load` adds a downward point force at the top
    centre. Returns (domain=(fixed_dofs, F), support_elem_mask).
    """
    nx, ny, nz = dims
    ndof = n_dof(dims)
    edof = edof_map(dims)
    if foot is None:
        foot = max(1, nx // 6)
    # feet: bottom-face nodes in the two end pads, all DOF fixed
    fixed = []
    foot_x = list(range(0, foot + 1)) + list(range(nx - foot, nx + 1))
    for i in foot_x:
        for j in range(ny + 1):
            n = node_id(dims, i, j, 0)
            fixed += [3 * n, 3 * n + 1, 3 * n + 2]
    # gravity body load: per-element mass ρ_tissue·V (V=1) × g_eff, 1/8 to each node, in −z
    g_eff = g * (1.0 - support_alpha)
    w = rho_tissue * g_eff
    fe = np.zeros((nx * ny * nz, 24))
    fe[:, 2::3] = -w / 8.0
    F = np.zeros(ndof)
    np.add.at(F, edof.reshape(-1), fe.reshape(-1))
    if ability_load:
        n = node_id(dims, nx // 2, ny // 2, nz)
        F[3 * n + 2] -= ability_load
    support_mask = np.zeros(dims, dtype=bool)
    support_mask[[i for i in foot_x if i < nx], :, 0] = True
    return (np.array(sorted(set(fixed)), dtype=np.int64), F), support_mask


if __name__ == "__main__":
    import time
    np.set_printoptions(precision=4, suppress=True)
    print(f"backend: {'GPU (cupy CG)' if _HAS_GPU else 'CPU (sparse LU)'}\n")
    P = FEParams()

    # 1) cantilever SIMP reproduces the textbook load-path: compliance drops & is ~monotone.
    dims = (32, 12, 6)
    dom, supp = build_cantilever(dims, load=1.0)
    t = time.time()
    rho, hist = simp_optimize(dims, dom, volfrac=0.3, params=P, n_iter=50)
    dt = time.time() - t
    drops = np.mean(np.diff(hist) <= 1e-9 * np.abs(hist[:-1]))   # fraction of non-increasing steps
    print("1) cantilever SIMP (dims=%s, vf=0.3, %d its, %.1fs)" % (dims, len(hist), dt))
    print("   compliance: start=%.4e  final=%.4e  reduction=%.1fx  monotone-frac=%.2f"
          % (hist[0], hist[-1], hist[0] / hist[-1], drops))
    assert hist[-1] < hist[0], "SIMP failed to reduce compliance"
    assert drops > 0.9, "SIMP compliance not (near-)monotone"

    # 2) binarized SIMP design is connected and load-bearing (touches the clamp).
    rb = binarize(rho, volfrac=0.3)
    con = connectivity(rb, supp)
    print("2) SIMP binary design: %s" % con)
    assert con["connected"], "SIMP design not connected/load-bearing"

    # 3) determinism: a repeat run matches to the CG tolerance regime (V0.5).
    rho2, hist2 = simp_optimize(dims, dom, volfrac=0.3, params=P, n_iter=50)
    rel = np.abs(hist[-1] - hist2[-1]) / abs(hist[-1])
    print("3) determinism: final-compliance rel diff on repeat = %.2e  (CG rtol=%.0e)"
          % (rel, CG_RTOL))
    assert rel < 1e-6, "non-deterministic SIMP result"

    # 4) creature gravity scaling + seraph support-field sanity (load-only, no opt).
    dimc = (24, 10, 16)
    dom_e, _ = build_creature(dimc, g=1.0)
    dom_s, _ = build_creature(dimc, g=1.0, support_alpha=1.0)
    print("4) creature body-load |F|: earth-g=%.3e  seraph(alpha=1)=%.3e"
          % (np.linalg.norm(dom_e[1]), np.linalg.norm(dom_s[1])))
    assert np.linalg.norm(dom_s[1]) < 1e-12, "seraph support did not cancel gravity"
    print("\nall topology_opt self-checks passed.")
