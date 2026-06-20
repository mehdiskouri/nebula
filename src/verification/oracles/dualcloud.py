"""
Dual-cloud skinning fidelity — the mechanism under test for V1.9 (protocol §V1.9; Decision #6;
ARCHITECTURE §III.1 "Rendering is decoupled: the dual cloud").

The architecture decouples appearance from physics: a COARSE physics cloud (mass/stress/state,
wired into the hypergraph) drives a DENSE render cloud (Gaussian splats bound by skinning weights
that ride the deformation of their physics neighbours). "Simulate thousands of nodes; render
millions of splats." V1.9 falsifies the economy: does a dense render cloud skinned to a coarse
physics cloud reproduce the FULL-RESOLUTION deformation within a visual tolerance, at a large
node-count reduction, with error that grows GRACEFULLY (no popping) as deformation increases?
Failure class is CONSTRAIN (cap supported deformation per coarse resolution; add adaptive
physics-cloud refinement), not KILL.

Independent oracle — an exact continuum deformation field
---------------------------------------------------------
The claim under test is the **skinning operator** (does a coarse cloud, skinned, reproduce the
fine deformation), so the oracle is the full-resolution deformation itself. We use an EXACT
smooth continuum deformation field φ (pure bending into a constant-curvature arc, helical twist,
or stretch). This is the strongest possible oracle for a skinning test: it is the true deformed
position of *every* render point at *any* resolution, with zero solver noise, fully controllable
severity — and, crucially, it lets us check skinning of MILLIONS of render points against exact
truth on the GPU. (Physics-solver fidelity is a separate concern, already validated by V1.7's FE
work; deliberately isolating skinning from solver error is the point.)

The mechanism
-------------
- A coarse physics cloud and a dense render cloud are two lattice samplings of the same beam.
- Both are deformed by φ; the coarse cloud's per-node rotation is estimated by polar
  decomposition (shape matching) from its OWN neighbour graph — exactly what the shippable
  system has (it does not know φ).
- Shippable cheap path: rotation-aware **linear-blend skinning** p = Σ wᵢ (Rᵢ (p₀−Xᵢ) + xᵢ).
- The **translation-only foil**: p = p₀ + Σ wᵢ (xᵢ − Xᵢ) — no per-node rotation → the
  candy-wrapper collapse, worst at high node-reduction (sparse frames) and large rotation.
- Error = per-render-point distance to φ(p₀), normalized by beam length.

numpy is the deterministic reference; the Warp GPU helpers (skin_lbs_gpu) skin the dense cloud at
full scale on the RTX 4090. Reuses determinism.rel_diff for the GPU↔numpy cross-check.
"""
from dataclasses import dataclass

import numpy as np

# Optional GPU backend (Warp) — same detection pattern as octree_gpu / determinism.
try:
    import warp as wp
    wp.init()
    _HAS_WARP = wp.get_cuda_device_count() > 0
except Exception:
    _HAS_WARP = False


@dataclass(frozen=True)
class BeamParams:
    L: float = 10.0          # beam length (x)
    W: float = 4.0           # cross-section side (y, z)
    nx: int = 13             # coarse lattice resolution along length
    ny: int = 4
    nz: int = 4
    neighbor_r: float = 1.8  # rotation-neighbour radius in units of lattice spacing


# ---------------- lattice ----------------

def make_lattice(bp: BeamParams, scale=1):
    """Rest positions of a beam lattice at `scale`× the coarse resolution. Returns (X, dims, spacing)."""
    nx = (bp.nx - 1) * scale + 1
    ny = (bp.ny - 1) * scale + 1
    nz = (bp.nz - 1) * scale + 1
    xs = np.linspace(0.0, bp.L, nx)
    ys = np.linspace(0.0, bp.W, ny)
    zs = np.linspace(0.0, bp.W, nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    X = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return X, (nx, ny, nz), bp.L / (nx - 1)


def neighbor_lists(X, spacing, neighbor_r):
    """Per-node neighbour index lists within neighbor_r*spacing (for rotation estimation)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    pairs = tree.query_pairs(neighbor_r * spacing, output_type="ndarray")
    nbr = [[] for _ in range(len(X))]
    for a, b in pairs:
        nbr[a].append(b); nbr[b].append(a)
    return [np.array(v, dtype=np.int64) for v in nbr]


# ---------------- the exact continuum deformation field (the oracle) ----------------

def _Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def deform_field(P, bp: BeamParams, mode="bend", severity=0.0):
    """Exact smooth deformation of points P (N,3). `severity`: total bend/twist angle (rad) or
    fractional stretch. Returns deformed positions — the full-resolution ground truth for any P."""
    P = np.asarray(P, float)
    s = P[:, 0]                                   # axis coordinate (arclength)
    yc, zc = P[:, 1] - bp.W / 2, P[:, 2] - bp.W / 2   # cross-section offsets
    out = np.empty_like(P)
    if mode == "bend":
        kappa = severity / bp.L                   # constant curvature; a = kappa*s
        a = kappa * s
        if abs(severity) < 1e-9:
            cx, cy = s, np.zeros_like(s)
        else:
            cx, cy = np.sin(a) / kappa, (1 - np.cos(a)) / kappa
        # frame: e1 tangent, e2 in-plane normal, e3 = z; cross-section rides e2,e3
        out[:, 0] = cx + (-np.sin(a)) * yc
        out[:, 1] = cy + (np.cos(a)) * yc
        out[:, 2] = P[:, 2]
    elif mode == "twist":
        phi = severity * s / bp.L                 # twist grows linearly along the axis
        cphi, sphi = np.cos(phi), np.sin(phi)
        out[:, 0] = s
        out[:, 1] = bp.W / 2 + cphi * yc - sphi * zc
        out[:, 2] = bp.W / 2 + sphi * yc + cphi * zc
    elif mode == "stretch":
        out[:, 0] = s * (1.0 + severity)
        out[:, 1] = P[:, 1]; out[:, 2] = P[:, 2]
    else:
        raise ValueError(mode)
    return out


# ---------------- per-node rotation (shape matching) ----------------

def node_rotations(X, x, nbr):
    """Polar-decomposition rotation per node from rest vs current neighbour offsets."""
    n = len(X)
    R = np.tile(np.eye(3), (n, 1, 1))
    for i in range(n):
        js = nbr[i]
        if len(js) < 3:
            continue
        A = (x[js] - x[i]).T @ (X[js] - X[i])
        U, _, Vt = np.linalg.svd(A)
        Ri = U @ Vt
        if np.linalg.det(Ri) < 0:
            U[:, -1] *= -1
            Ri = U @ Vt
        R[i] = Ri
    return R


# ---------------- skinning ----------------

def bind_weights(P0, Xc, k=4):
    """Bind each dense rest point to its k nearest coarse rest nodes (inverse-distance weights)."""
    from scipy.spatial import cKDTree
    dist, idx = cKDTree(Xc).query(P0, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    w = 1.0 / (dist ** 2 + 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    return idx.astype(np.int64), w


def skin_lbs(P0, idx, w, Xc, xc, Rc):
    """Rotation-aware linear-blend skinning: p = Σ wᵢ (Rᵢ (p₀−Xᵢ) + xᵢ)."""
    off = P0[:, None, :] - Xc[idx]
    contrib = np.einsum("nkij,nkj->nki", Rc[idx], off) + xc[idx]
    return np.einsum("nk,nki->ni", w, contrib)


def skin_translation(P0, idx, w, Xc, xc):
    """Translation-only foil: p = p₀ + Σ wᵢ (xᵢ − Xᵢ)  (no per-node rotation)."""
    return P0 + np.einsum("nk,nki->ni", w, xc[idx] - Xc[idx])


# ---------------- GPU skinning (Warp) — for the full-scale test ----------------

if _HAS_WARP:
    @wp.kernel
    def _lbs_kernel(P0: wp.array(dtype=wp.vec3), idx: wp.array2d(dtype=wp.int32),
                    w: wp.array2d(dtype=wp.float32), Xc: wp.array(dtype=wp.vec3),
                    xc: wp.array(dtype=wp.vec3), Rc: wp.array(dtype=wp.mat33),
                    K: wp.int32, out: wp.array(dtype=wp.vec3)):
        n = wp.tid()
        p = wp.vec3(0.0, 0.0, 0.0)
        for j in range(K):
            i = idx[n, j]
            p = p + w[n, j] * (Rc[i] * (P0[n] - Xc[i]) + xc[i])
        out[n] = p


def skin_lbs_gpu(P0, idx, w, Xc, xc, Rc):
    """GPU rotation-aware LBS over the dense cloud (the embarrassingly-parallel render path)."""
    if not _HAS_WARP:
        raise RuntimeError("no GPU")
    K = idx.shape[1]
    d = "cuda"
    P0d = wp.array(P0.astype(np.float32), dtype=wp.vec3, device=d)
    idxd = wp.array(idx.astype(np.int32), dtype=wp.int32, device=d)
    wd = wp.array(w.astype(np.float32), dtype=wp.float32, device=d)
    Xcd = wp.array(Xc.astype(np.float32), dtype=wp.vec3, device=d)
    xcd = wp.array(xc.astype(np.float32), dtype=wp.vec3, device=d)
    Rcd = wp.array(Rc.astype(np.float32), dtype=wp.mat33, device=d)
    out = wp.zeros(len(P0), dtype=wp.vec3, device=d)
    wp.launch(_lbs_kernel, dim=len(P0), inputs=[P0d, idxd, wd, Xcd, xcd, Rcd, K, out], device=d)
    wp.synchronize()
    return out.numpy().astype(np.float64)


# ---------------- end-to-end experiment helper ----------------

def run_case(bp: BeamParams, dense_scale, mode, severity, k=4, gpu=False):
    """Skin a dense render cloud (driven by the coarse physics cloud) and compare to φ (the oracle).

    Coarse rotations are estimated from the coarse cloud's own neighbour graph (the shippable
    system does not know φ). Returns per-point errors (fraction of beam length) for LBS & the foil.
    """
    Xc, _, sc = make_lattice(bp, scale=1)
    xc = deform_field(Xc, bp, mode, severity)
    nbrc = neighbor_lists(Xc, sc, bp.neighbor_r)
    Rc = node_rotations(Xc, xc, nbrc)

    Xd, _, _ = make_lattice(bp, scale=dense_scale)
    truth = deform_field(Xd, bp, mode, severity)          # the full-resolution oracle

    idx, w = bind_weights(Xd, Xc, k=k)
    p_lbs = skin_lbs_gpu(Xd, idx, w, Xc, xc, Rc) if gpu else skin_lbs(Xd, idx, w, Xc, xc, Rc)
    p_tr = skin_translation(Xd, idx, w, Xc, xc)

    e_lbs = np.linalg.norm(p_lbs - truth, axis=1) / bp.L
    e_tr = np.linalg.norm(p_tr - truth, axis=1) / bp.L
    return dict(Nc=len(Xc), Nd=len(Xd), reduction=len(Xd) / len(Xc),
                lbs_mean=float(e_lbs.mean()), lbs_max=float(e_lbs.max()),
                lbs_p95=float(np.percentile(e_lbs, 95)),
                tr_mean=float(e_tr.mean()), tr_max=float(e_tr.max()),
                xc=xc, Xc=Xc, Xd=Xd, truth=truth, p_lbs=p_lbs, p_tr=p_tr, Rc=Rc,
                idx=idx, w=w)


if __name__ == "__main__":
    import determinism as det

    bp = BeamParams()
    # 1) fidelity: rotation-aware LBS tracks the exact field at >=10x reduction under a big bend.
    r = run_case(bp, dense_scale=3, mode="bend", severity=np.deg2rad(90))
    print("1) bend 90deg, reduction=%.1fx (Nc=%d Nd=%d)" % (r["reduction"], r["Nc"], r["Nd"]))
    print("   LBS  err mean=%.4f max=%.4f p95=%.4f  (fraction of beam length)"
          % (r["lbs_mean"], r["lbs_max"], r["lbs_p95"]))
    print("   foil err mean=%.4f max=%.4f" % (r["tr_mean"], r["tr_max"]))
    assert r["reduction"] >= 10.0 and r["lbs_mean"] < 0.02, "fidelity/reduction failed"

    # 2) necessity: the translation-only foil collapses under twist (candy-wrapper) & sharp bend.
    rt = run_case(bp, dense_scale=3, mode="twist", severity=np.deg2rad(180))
    print("2) twist 180deg: LBS mean=%.4f  foil mean=%.4f  (foil/LBS=%.1fx)"
          % (rt["lbs_mean"], rt["tr_mean"], rt["tr_mean"] / rt["lbs_mean"]))
    assert rt["tr_mean"] > 2 * rt["lbs_mean"] and r["tr_mean"] > 2 * r["lbs_mean"], "foil not worse"

    # 3) graceful: LBS error monotone & jump-free as the bend grows.
    errs = [run_case(bp, dense_scale=3, mode="bend", severity=np.deg2rad(a))["lbs_mean"]
            for a in (0, 15, 30, 45, 60, 75, 90)]
    print("3) graceful bend sweep LBS mean errs:", [round(e, 4) for e in errs])
    assert np.diff(errs).max() < 0.01, "popping: large jump"

    # 4) determinism + GPU == numpy at modest N.
    r2 = run_case(bp, dense_scale=3, mode="bend", severity=np.deg2rad(90))
    print("4) determinism rel diff (repeat) = %.2e" % det.rel_diff(r["p_lbs"], r2["p_lbs"]))
    assert det.rel_diff(r["p_lbs"], r2["p_lbs"]) < 1e-12
    if _HAS_WARP:
        g = skin_lbs_gpu(r["Xd"], r["idx"], r["w"], r["Xc"], r["xc"], r["Rc"])
        print("   GPU==numpy rel diff = %.2e" % det.rel_diff(g, r["p_lbs"]))
        assert det.rel_diff(g, r["p_lbs"]) < 1e-5, "GPU LBS disagrees with numpy"
    print("\ndualcloud self-checks passed.")
