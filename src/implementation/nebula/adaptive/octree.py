"""
Morton-linearized octree + Barnes-Hut far-field — the "one tree, three hats"
(ARCHITECTURE §III.2; Decision #8). Verified by V0.4 (PASS): n_total exponent 0.161 (log),
n_active exponent 1.009 (linear), <=0.65x dense at full activation; descent depth = 1.02 x
log8(n) (early termination demonstrably fires).

Build: normalize points to [0,1)^3, compute 3-D Morton (Z-order) codes, sort, build a
linearized octree by recursively partitioning the sorted range into 8 octants. Each node
carries a COARSE PROXY = aggregate mass + center-of-mass (the Barnes-Hut monopole), computed
bottom-up. The opening criterion size/distance < theta turns the proxy into the log factor: a
distant query stops descending once a node's reduced model is accurate enough -- soundness
rests on the proxy being a valid bounded-error reduced model (V0.1's premise, one domain over).

Ported verbatim-in-behaviour from the frozen oracle src/verification/oracles/octree.py. The
same flat DFS-linearized arrays drive the GPU traversal in octree_gpu.py. Pure numpy.
"""
from dataclasses import dataclass

import numpy as np

MORTON_BITS = 21          # bits per axis (3*21 = 63-bit code fits in uint64)


@dataclass
class Octree:
    points: np.ndarray; masses: np.ndarray
    level: np.ndarray; lo: np.ndarray; hi: np.ndarray
    center: np.ndarray; half: np.ndarray; child: np.ndarray; is_leaf: np.ndarray
    mass: np.ndarray; com: np.ndarray; size: np.ndarray; n_nodes: int


def _part1by2(n):
    """Spread the low 21 bits of n into every 3rd bit (uint64)."""
    n = n.astype(np.uint64) & np.uint64(0x1fffff)
    n = (n | (n << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    n = (n | (n << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    n = (n | (n << np.uint64(8)))  & np.uint64(0x100f00f00f00f00f)
    n = (n | (n << np.uint64(4)))  & np.uint64(0x10c30c30c30c30c3)
    n = (n | (n << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return n


def morton_codes(unit_pts):
    """3-D Morton codes for points in [0,1)^3 (interleave x,y,z; bit0=x,1=y,2=z)."""
    q = np.clip((unit_pts * (1 << MORTON_BITS)).astype(np.int64), 0, (1 << MORTON_BITS) - 1)
    return (_part1by2(q[:, 0]) | (_part1by2(q[:, 1]) << np.uint64(1))
            | (_part1by2(q[:, 2]) << np.uint64(2)))


def build(points, masses, leaf_capacity=16, max_level=MORTON_BITS):
    """Build the Morton-linearized octree with bottom-up monopole aggregation."""
    points = np.asarray(points, float)
    masses = np.asarray(masses, float)
    n = len(points)
    lo_box = points.min(0); span = (points.max(0) - lo_box).max()
    span = span if span > 0 else 1.0
    unit = (points - lo_box) / (span * (1 + 1e-9))
    codes = morton_codes(unit)
    order = np.argsort(codes, kind="stable")
    pts = points[order]; mas = masses[order]; codes = codes[order]

    L_, LO_, HI_, CX_, HALF_, LEAF_, CHILD_ = [], [], [], [], [], [], []

    def new_node(level, lo, hi, center, half):
        idx = len(L_)
        L_.append(level); LO_.append(lo); HI_.append(hi)
        CX_.append(center); HALF_.append(half); LEAF_.append(False)
        CHILD_.append([-1] * 8)
        return idx

    root = new_node(0, 0, n, np.array([0.5, 0.5, 0.5]), 0.5)
    stack = [root]
    while stack:
        nd = stack.pop()
        lo, hi, level = LO_[nd], HI_[nd], L_[nd]
        if (hi - lo) <= leaf_capacity or level >= max_level:
            LEAF_[nd] = True
            continue
        shift = np.uint64(3 * (max_level - 1 - level))
        oct_of = ((codes[lo:hi] >> shift) & np.uint64(7)).astype(np.int64)
        bounds = np.searchsorted(oct_of, np.arange(9))
        center = CX_[nd]; half = HALF_[nd]; chalf = half * 0.5
        any_child = False
        for o in range(8):
            a, b = lo + bounds[o], lo + bounds[o + 1]
            if b <= a:
                continue
            bx, by, bz = o & 1, (o >> 1) & 1, (o >> 2) & 1
            cc = center + (np.array([bx, by, bz]) - 0.5) * half
            cidx = new_node(level + 1, a, b, cc, chalf)
            CHILD_[nd][o] = cidx
            stack.append(cidx)
            any_child = True
        if not any_child:
            LEAF_[nd] = True

    n_nodes = len(L_)
    level = np.array(L_); LO = np.array(LO_); HI = np.array(HI_)
    center = np.array(CX_); half = np.array(HALF_)
    child = np.array(CHILD_); is_leaf = np.array(LEAF_)
    size = 2.0 * half * span

    mass = np.zeros(n_nodes); com = np.zeros((n_nodes, 3))
    csum = np.cumsum(np.concatenate([[0.0], mas]))
    cxsum = np.cumsum(np.concatenate([np.zeros((1, 3)), pts * mas[:, None]]), axis=0)
    for nd in range(n_nodes - 1, -1, -1):
        if is_leaf[nd]:
            a, b = LO[nd], HI[nd]
            m = csum[b] - csum[a]
            mass[nd] = m
            com[nd] = (cxsum[b] - cxsum[a]) / m if m > 0 else center[nd] * span + lo_box
        else:
            m = 0.0; mc = np.zeros(3)
            for o in range(8):
                ci = child[nd, o]
                if ci >= 0:
                    m += mass[ci]; mc += mass[ci] * com[ci]
            mass[nd] = m
            com[nd] = mc / m if m > 0 else center[nd] * span + lo_box

    return Octree(pts, mas, level, LO, HI, center, half, child, is_leaf,
                  mass, com, size, n_nodes)


@dataclass
class LinTree:
    """DFS-preorder linearization with escape pointers for stackless traversal."""
    com: np.ndarray; mass: np.ndarray; size: np.ndarray
    is_leaf: np.ndarray; lo: np.ndarray; hi: np.ndarray
    escape: np.ndarray; level: np.ndarray
    points: np.ndarray; masses: np.ndarray; n_nodes: int


def dfs_linearize(tree):
    """Reorder a built Octree into DFS preorder + escape pointers (LinTree)."""
    n = tree.n_nodes
    order = np.empty(n, np.int64); new_of = np.empty(n, np.int64)
    stack = [0]; k = 0
    while stack:
        nd = stack.pop(); order[k] = nd; new_of[nd] = k; k += 1
        if not tree.is_leaf[nd]:
            for o in range(7, -1, -1):
                ci = tree.child[nd, o]
                if ci >= 0:
                    stack.append(ci)
    size_sub = np.ones(n, np.int64)
    parent_new = np.full(n, -1, np.int64)
    for nd in range(tree.n_nodes):
        if not tree.is_leaf[nd]:
            for o in range(8):
                ci = tree.child[nd, o]
                if ci >= 0:
                    parent_new[new_of[ci]] = new_of[nd]
    for newidx in range(n - 1, 0, -1):
        size_sub[parent_new[newidx]] += size_sub[newidx]
    escape = np.arange(n, dtype=np.int64) + size_sub
    return LinTree(tree.com[order], tree.mass[order], tree.size[order],
                   tree.is_leaf[order], tree.lo[order], tree.hi[order],
                   escape, tree.level[order], tree.points, tree.masses, n)


def bh_field(lin, queries, theta=0.5, eps=1e-4, G=1.0):
    """Barnes-Hut potential phi(q) = sum_j G m_j / |q - p_j| (softened), stackless.

    Returns (phi, interactions, depth): interactions = exact work/query, depth = max level visited.
    """
    queries = np.asarray(queries, float)
    nq = len(queries)
    phi = np.zeros(nq); inter = np.zeros(nq, np.int64); depth = np.zeros(nq, np.int64)
    com = lin.com; mass = lin.mass; size = lin.size; is_leaf = lin.is_leaf
    lo = lin.lo; hi = lin.hi; escape = lin.escape; lvl = lin.level
    pts = lin.points; mas = lin.masses
    eps2 = eps * eps
    for i in range(nq):
        q = queries[i]; p = 0.0; cnt = 0; dmax = 0; idx = 0
        while idx < lin.n_nodes:
            if lvl[idx] > dmax:
                dmax = lvl[idx]
            d = np.sqrt((com[idx, 0]-q[0])**2 + (com[idx, 1]-q[1])**2
                        + (com[idx, 2]-q[2])**2 + eps2)
            if is_leaf[idx]:
                a, b = lo[idx], hi[idx]
                for j in range(a, b):
                    dj = np.sqrt((pts[j, 0]-q[0])**2 + (pts[j, 1]-q[1])**2
                                 + (pts[j, 2]-q[2])**2 + eps2)
                    p += G * mas[j] / dj
                cnt += (b - a); idx = escape[idx]
            elif size[idx] / d < theta:
                p += G * mass[idx] / d; cnt += 1; idx = escape[idx]
            else:
                idx += 1
        phi[i] = p; inter[i] = cnt; depth[i] = dmax
    return phi, inter, depth


def direct_field(points, masses, queries, eps=1e-4, G=1.0, chunk=512):
    """Exact all-pairs potential (the O(n_q . n) dense oracle + correctness reference)."""
    points = np.asarray(points, float); masses = np.asarray(masses, float)
    queries = np.asarray(queries, float)
    phi = np.empty(len(queries))
    eps2 = eps * eps
    for s in range(0, len(queries), chunk):
        Q = queries[s:s + chunk]
        d = np.sqrt(((Q[:, None, :] - points[None, :, :]) ** 2).sum(-1) + eps2)
        phi[s:s + chunk] = G * (masses[None, :] / d).sum(1)
    return phi


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 5000
    pts = rng.random((n, 3)); mas = rng.random(n) + 0.1
    tree = build(pts, mas)
    print(f"1) build n={n} nodes={tree.n_nodes} leaves={int(tree.is_leaf.sum())} "
          f"root mass err={abs(tree.mass[0]-mas.sum()):.2e}")
    lin = dfs_linearize(tree)
    q = rng.random((200, 3)); exact = direct_field(pts, mas, q)
    print("2) BH-vs-direct relative error & mean interactions/query by theta:")
    for theta in (1.0, 0.5, 0.3):
        phi, inter, depth = bh_field(lin, q, theta=theta)
        relerr = np.abs(phi - exact) / np.abs(exact)
        print(f"   theta={theta}: mean relerr={relerr.mean():.2e}  interactions/q={inter.mean():.1f}  "
              f"depth={depth.mean():.1f}")
