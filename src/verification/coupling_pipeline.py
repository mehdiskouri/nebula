"""
Coupling-operator full pipeline (ARCHITECTURE §III.8; Decisions #24, #25) — V1.4.

Builds the geometric vitruvian/spectral pipeline on top of the proven symmetry core
(symmetry_basis.py):

    reference + skeleton + thickness knob  ->  within-part lift  ->  symmetry-adapted GFT
    ->  joint coefficient tensor Ĉ  ->  reconstructed 3-D form  ->  silhouette.

* Within-part lift: each bone is a local medial axis; its surface is a generalized cylinder
  r(s,θ). The silhouette gives the IN-PLANE half-width w(s); the THICKNESS KNOB κ supplies the
  provably-unrecoverable DEPTH axis (Decision #25) — so κ never changes the front silhouette.
  Each bone's w(s) is a degree-(NS-1) polynomial + a few θ-harmonics (micro detail) -> a per-node
  shape-coefficient vector cᵢ (each child node owns its incoming bone; the root owns none).
* Across-part GFT: stack {cᵢ} as channels on the graph and apply the symmetry-adapted basis ->
  Ĉ[m,k]  (m = graph/irrep mode by Laplacian frequency, k = within-part mode). Truncation = LOD.
* C¹ quilt: a constrained least-squares over the per-bone w-coefficients enforces, at every
  parent->child seam, C⁰ (shared ring radius) and C¹ (arc-length radius slope) continuity — the
  Interface-hyperedge / hanging-node fix reused for geometry.

Synthetic ground-truth-by-construction: a known symmetric creature is the reference; the pipeline
round-trips it and is judged on silhouette IoU, graceful truncation, macro-vs-micro response,
symmetry-lock, and seam C¹ residual. Pure numpy. Reuses symmetry_basis (regression-safe).
"""
import numpy as np

import symmetry_basis as sb

NS = 4          # w(s) polynomial coefficients per bone (a0 + a1 s + a2 s^2 + a3 s^3)
NTH = 2         # θ-harmonic amplitudes per bone (cos 2θ, cos 3θ) — the micro-detail channels
K = NS + NTH    # channels per node
THETA_H = (2, 3)


# ============================ skeletons ============================

def _frames(pos, parent):
    """Per child-node local frame (axis tangent t, in-plane normal n, depth axis d)."""
    N = len(pos); fr = {}
    for n in range(N):
        p = parent[n]
        if p is None:
            continue
        a, b = pos[p], pos[n]
        t = b - a; L = np.linalg.norm(t) + 1e-12; t = t / L
        # in-plane normal: perpendicular to t in the image (x,y) plane
        nrm = np.array([-t[1], t[0], 0.0])
        if np.linalg.norm(nrm) < 1e-6:
            nrm = np.array([1.0, 0.0, 0.0])
        nrm = nrm / np.linalg.norm(nrm)
        d = np.cross(t, nrm); d = d / (np.linalg.norm(d) + 1e-12)   # depth axis
        fr[n] = dict(a=a, b=b, t=t, n=nrm, d=d, L=L)
    return fr


def biped_skeleton():
    """17-joint bilateral biped (matches coupling_operator_core.py), mirror-symmetric in x."""
    names = ['pelvis', 'spine', 'chest', 'neck', 'head',
             'L_sho', 'L_elb', 'L_hand', 'R_sho', 'R_elb', 'R_hand',
             'L_hip', 'L_knee', 'L_foot', 'R_hip', 'R_knee', 'R_foot']
    pos = np.array([
        [0, 0, 0], [0, 1.0, 0], [0, 2.0, 0], [0, 2.5, 0], [0, 3.0, 0],
        [0.5, 2.0, 0], [1.2, 1.9, 0], [1.9, 1.8, 0],
        [-0.5, 2.0, 0], [-1.2, 1.9, 0], [-1.9, 1.8, 0],
        [0.25, -0.1, 0], [0.3, -1.3, 0], [0.35, -2.5, 0],
        [-0.25, -0.1, 0], [-0.3, -1.3, 0], [-0.35, -2.5, 0]], float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7), (2, 8), (8, 9), (9, 10),
             (0, 11), (11, 12), (12, 13), (0, 14), (14, 15), (15, 16)]
    parent = {1: 0, 2: 1, 3: 2, 4: 3, 5: 2, 6: 5, 7: 6, 8: 2, 9: 8, 10: 9,
              11: 0, 12: 11, 13: 12, 14: 0, 15: 14, 16: 15, 0: None}
    swap = {5: 8, 6: 9, 7: 10, 11: 14, 12: 15, 13: 16}
    perm = list(range(17))
    for x, y in swap.items():
        perm[x], perm[y] = y, x
    L = sb.laplacian(17, edges); P = sb.permutation_matrix(perm)
    return dict(kind="Z2", N=17, names=names, pos=pos, edges=edges, parent=parent,
                perm=perm, swap=swap, L=L, sym=P, frames=_frames(pos, parent))


def seraph_skeleton():
    """19-node C6 seraph (matches coupling_operator_c6.py): core + 6 wings + halo ring."""
    pos = np.zeros((19, 3))
    for w in range(6):
        ang = w * np.pi / 3.0
        rad, mid, tip = 1 + 3 * w, 2 + 3 * w, 3 + 3 * w
        c, s = np.cos(ang), np.sin(ang)
        pos[rad] = [0.6 * c, 0.6 * s, 0.0]
        pos[mid] = [1.3 * c, 1.3 * s, 0.0]
        pos[tip] = [2.1 * c, 2.1 * s, 0.0]
    edges, parent = [], {0: None}
    for w in range(6):
        rad, mid, tip = 1 + 3 * w, 2 + 3 * w, 3 + 3 * w
        edges += [(0, rad), (rad, mid), (mid, tip), (rad, 1 + 3 * ((w + 1) % 6))]
        parent[rad] = 0; parent[mid] = rad; parent[tip] = mid
    pc = list(range(19))
    for w in range(6):
        nw = (w + 1) % 6
        pc[1 + 3 * w], pc[2 + 3 * w], pc[3 + 3 * w] = 1 + 3 * nw, 2 + 3 * nw, 3 + 3 * nw
    L = sb.laplacian(19, edges); R = sb.permutation_matrix(pc)
    return dict(kind="C6", N=19, pos=pos, edges=edges, parent=parent, perm=pc,
                L=L, sym=R, order=6, frames=_frames(pos, parent))


# ============================ within-part shape & targets ============================

def _wpoly(coeff_a, s):
    """w(s) = Σ a_j s^j  (coeff_a length NS)."""
    return sum(coeff_a[j] * s ** j for j in range(NS))


def biped_target():
    """Symmetric ground-truth biped: per-bone [a0..a3, th2, th3]. L/R bones identical."""
    C = np.zeros((17, K))
    # (child node) -> (base radius a0, taper a1 = end-start); torso thick, limbs taper.
    # Buried connector bones (clavicle 5/8, hip-link 11/14) sit INSIDE the torso silhouette, so
    # their target ~= the local body envelope (the lift can only measure the envelope there).
    rad = {1: (0.34, 0.0), 2: (0.36, 0.0), 3: (0.24, 0.0), 4: (0.42, 0.0),      # torso/neck/head
           5: (0.245, 0.0), 6: (0.16, -0.03), 7: (0.11, -0.03),                 # left arm (clavicle buried)
           11: (0.20, 0.0), 12: (0.19, -0.04), 13: (0.12, -0.03)}               # left leg (hip-link buried)
    for L, Rn in [(5, 8), (6, 9), (7, 10), (11, 14), (12, 15), (13, 16)]:       # mirror to right
        rad[Rn] = rad[L]
    for n, (a0, a1) in rad.items():
        C[n, 0] = a0; C[n, 1] = a1
    return C


def seraph_target():
    """Symmetric ground-truth seraph: identical wing bones (C6), a core blob."""
    C = np.zeros((19, K))
    for w in range(6):
        rad, mid, tip = 1 + 3 * w, 2 + 3 * w, 3 + 3 * w
        C[rad, 0] = 0.18; C[rad, 1] = -0.02
        C[mid, 0] = 0.16; C[mid, 1] = -0.04
        C[tip, 0] = 0.10; C[tip, 1] = -0.03
    return C


# ============================ reconstruction & silhouette ============================

def bone_solid_points(skel, C, n, kappa, ns=18, nth=30, nr=7):
    """Filled cross-section sample points (3-D) of bone owned by child node `n`."""
    fr = skel["frames"][n]; a, t, nn, d, Lb = fr["a"], fr["t"], fr["n"], fr["d"], fr["L"]
    coeff_a = C[n, :NS]; th = C[n, NS:]
    ss = np.linspace(0, 1, ns); thetas = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    us = np.linspace(0.15, 1.0, nr)
    pts = []
    for s in ss:
        axis = a + s * (fr["b"] - a)
        w = _wpoly(coeff_a, s)
        for theta in thetas:
            rho = w * (1.0 + sum(th[h] * np.cos(THETA_H[h] * theta) for h in range(NTH)))
            for u in us:
                pts.append(axis + u * rho * np.cos(theta) * nn + u * kappa * rho * np.sin(theta) * d)
    return np.array(pts)


def all_points(skel, C, kappa):
    pts = [bone_solid_points(skel, C, n, kappa) for n in range(skel["N"]) if skel["parent"][n] is not None]
    return np.vstack(pts)


def silhouette(pts3d, res=320, bounds=None):
    """Rasterize projected (x,y) points to a SOLID binary mask (front view, depth dropped).
    Returns (mask, bounds). Holes from finite sampling are closed (dilate → fill)."""
    from scipy import ndimage
    xy = pts3d[:, :2]
    if bounds is None:
        lo = xy.min(0) - 0.2; hi = xy.max(0) + 0.2
        c = (lo + hi) / 2; half = (hi - lo).max() / 2
        bounds = (c - half, c + half)
    lo, hi = bounds
    ij = np.floor((xy - lo) / (hi - lo) * (res - 1)).astype(int)
    ij = np.clip(ij, 0, res - 1)
    mask = np.zeros((res, res), bool)
    mask[ij[:, 1], ij[:, 0]] = True
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=2))
    return mask, bounds


def iou(m1, m2):
    inter = np.logical_and(m1, m2).sum(); union = np.logical_or(m1, m2).sum()
    return float(inter) / float(union + 1e-12)


def _in_mask(xy, mask, bounds, res):
    lo, hi = bounds
    j, i = np.floor((xy - lo) / (hi - lo) * (res - 1)).astype(int)
    if 0 <= i < res and 0 <= j < res:
        return mask[i, j]
    return False


def _seg_dist(p, a, b):
    """Distance from point p to segment a-b (all 2-D)."""
    ab = b - a; t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def _nearest_bone(p, skel, bones):
    """The bone (child node) whose medial-axis segment is nearest to point p (the vitruvian
    anchor's supervision: each silhouette region belongs to its painted bone)."""
    best, bn = 1e18, None
    for n in bones:
        fr = skel["frames"][n]
        dseg = _seg_dist(p, fr["a"][:2], fr["b"][:2])
        if dseg < best:
            best, bn = dseg, n
    return bn


def lift_from_silhouette(skel, mask, bounds, ns_samples=11):
    """THE WITHIN-PART LIFT (protocol §V1.4): measure each bone's in-plane half-width w(s) by
    marching perpendicular to the medial axis through the reference silhouette, then fit a
    degree-(NS-1) polynomial. Depth (θ-harmonics) is unrecoverable from one view -> left to the
    thickness knob; lifted θ-channels are 0. Returns node coefficient matrix C (N,K)."""
    res = mask.shape[0]; pix = (bounds[1] - bounds[0]).max() / res; step = pix * 0.3
    bones = [n for n in range(skel["N"]) if skel["parent"][n] is not None]
    C = np.zeros((skel["N"], K))
    ss = np.linspace(0.30, 0.70, ns_samples)        # clean bone interior (avoid joint overlap)
    deg = 2                                          # stable low-degree fit -> no wild extrapolation
    for n in bones:
        fr = skel["frames"][n]; a, b, nn = fr["a"], fr["b"], fr["n"]
        ws = []
        for s in ss:
            axis = (a + s * (b - a))[:2]
            half = []
            for sgn in (+1.0, -1.0):
                dist = 0.0
                for _ in range(int(0.9 / step)):
                    q = axis + (dist + step) * sgn * nn[:2]
                    # march while still inside the silhouette AND inside THIS bone's anchor region
                    if not _in_mask(q, mask, bounds, res) or _nearest_bone(q, skel, bones) != n:
                        break
                    dist += step
                half.append(dist + 0.5 * pix)        # half-pixel edge debias (true edge is mid-pixel)
            ws.append(0.5 * (half[0] + half[1]))
        coeff = np.polynomial.polynomial.polyfit(ss, ws, deg)      # ascending powers -> a_j
        C[n, :len(coeff)] = coeff
    return C


def chamfer(p1, p2, sample=400, seed=0):
    """Symmetric mean nearest-neighbor distance between two 2-D point clouds (xy)."""
    rng = np.random.default_rng(seed)
    a = p1[rng.choice(len(p1), min(sample, len(p1)), replace=False), :2]
    b = p2[rng.choice(len(p2), min(sample, len(p2)), replace=False), :2]
    da = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1).mean()
    db = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(1).mean()
    return float(0.5 * (da + db))


# ============================ symmetry-adapted GFT ============================

def make_basis(skel):
    """Return (basis, eigvals, labels). basis columns are graph modes (real Z2 / complex Cn)."""
    if skel["kind"] == "Z2":
        w, V, char = sb.adapted_basis_real(skel["L"], skel["sym"])
        return V, w, char                  # labels = +1 SYM / -1 ANTI
    w, U, m = sb.adapted_basis_cyclic(skel["L"], skel["sym"], skel["order"])
    return U, w, m                         # labels = angular momentum m


def gft_forward(C, basis):
    return basis.conj().T @ C              # Ĉ[m,k]


def gft_inverse(Chat, basis):
    return np.real(basis @ Chat)


def truncate(Chat, eigvals, n_graph, k_keep):
    """Keep the n_graph lowest-frequency graph modes and the k_keep lowest within-part channels."""
    out = np.zeros_like(Chat)
    order = np.argsort(eigvals)[:n_graph]
    out[order, :k_keep] = Chat[order, :k_keep]
    return out


# ============================ C1 quilt stitch ============================

def seam_list(skel):
    """(parent_bone_node, child_bone_node) pairs at every non-root parent->child joint."""
    seams = []
    children = {n: [] for n in range(skel["N"])}
    for n in range(skel["N"]):
        p = skel["parent"][n]
        if p is not None:
            children[p].append(n)
    for n in range(skel["N"]):
        if skel["parent"][n] is None:
            continue                       # n's incoming bone is the parent bone
        for c in children[n]:
            seams.append((n, c))           # bone(n) -> bone(c) share joint n
    return seams


def _seam_constraints(skel):
    """Linear constraint matrix C over stacked w-coeffs x (bones × NS) for C0 + C1 at seams."""
    bones = [n for n in range(skel["N"]) if skel["parent"][n] is not None]
    idx = {n: i for i, n in enumerate(bones)}
    nb = len(bones); rows = []
    for (pn, cn) in seam_list(skel):
        Lp = skel["frames"][pn]["L"]; Lc = skel["frames"][cn]["L"]
        # C0:  w_p(1) - w_c(0) = 0     ->  Σ_j a_p,j  -  a_c,0 = 0
        r0 = np.zeros(nb * NS)
        for j in range(NS):
            r0[idx[pn] * NS + j] += 1.0
        r0[idx[cn] * NS + 0] -= 1.0
        rows.append(r0)
        # C1:  w_p'(1)/Lp - w_c'(0)/Lc = 0  ->  (Σ_j j a_p,j)/Lp - (a_c,1)/Lc = 0
        r1 = np.zeros(nb * NS)
        for j in range(NS):
            r1[idx[pn] * NS + j] += j / Lp
        r1[idx[cn] * NS + 1] -= 1.0 / Lc
        rows.append(r1)
    return np.array(rows), bones, idx


def quilt_stitch(C, skel):
    """Project the per-bone w-coefficients onto the C0+C1 seam-continuity subspace (nearest to the
    lifted target). Returns (C_stitched, seam_residual_after, seam_gap_before)."""
    Cmat, bones, idx = _seam_constraints(skel)
    x0 = np.array([C[n, j] for n in bones for j in range(NS)])
    gap_before = float(np.linalg.norm(Cmat @ x0))
    # equality-constrained least squares: min ||x-x0|| s.t. Cmat x = 0  -> x = x0 - Cᵀ(CCᵀ)⁻¹C x0
    CCt = Cmat @ Cmat.T
    x = x0 - Cmat.T @ np.linalg.solve(CCt + 1e-12 * np.eye(len(CCt)), Cmat @ x0)
    res_after = float(np.linalg.norm(Cmat @ x))
    Cs = C.copy()
    for n in bones:
        Cs[n, :NS] = x[idx[n] * NS:idx[n] * NS + NS]
    return Cs, res_after, gap_before


# ============================ convenience round-trip ============================

def reconstruct(skel, Chat, basis, kappa):
    return all_points(skel, gft_inverse(Chat, basis), kappa)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for name, sk_fn, tgt_fn in [("biped", biped_skeleton, biped_target),
                                ("seraph", seraph_skeleton, seraph_target)]:
        sk = sk_fn(); C0 = tgt_fn(); kappa = 0.6
        basis, eig, lab = make_basis(sk)
        # reference silhouette from the ground-truth creature
        mask_ref, bnd = silhouette(all_points(sk, C0, kappa))
        # FULL PIPELINE: lift from the silhouette -> GFT -> inverse -> reconstruct
        C_lift = lift_from_silhouette(sk, mask_ref, bnd)
        Chat = gft_forward(C_lift, basis)
        C_rec = gft_inverse(Chat, basis)
        mask_rec, _ = silhouette(all_points(sk, C_rec, kappa), bounds=bnd)
        print(f"\n{name}: lift→GFT→reconstruct silhouette IoU = {iou(mask_ref, mask_rec):.4f}  "
              f"(ref area {mask_ref.sum()} px)")
        print(f"  GFT round-trip coeff error = {np.abs(C_rec - C_lift).max():.2e}")
        Cs, res_after, gap_before = quilt_stitch(C_lift, sk)
        print(f"  C¹ quilt: seam gap before = {gap_before:.3e}  residual after = {res_after:.2e}")
