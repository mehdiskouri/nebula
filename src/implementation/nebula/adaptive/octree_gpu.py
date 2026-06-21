"""
GPU Barnes-Hut traversal (NVIDIA Warp) over the SAME Morton-linearized octree
(ARCHITECTURE Part V: the Taichi/Warp GPU substrate). One thread per query walks the
DFS-linearized node arrays (octree.LinTree) stackless via escape pointers -- identical logic
to octree.bh_field. Used by V0.4 to push n to 10^6+ and corroborate the scaling in wall-clock.

Ported verbatim-in-behaviour from the frozen oracle src/verification/oracles/octree_gpu.py.
"""
import numpy as np

try:
    import warp as wp
    wp.init()
    _HAS_WARP = wp.get_cuda_device_count() > 0
except Exception:
    _HAS_WARP = False


if _HAS_WARP:
    @wp.kernel
    def _bh_kernel(com: wp.array(dtype=wp.vec3), mass: wp.array(dtype=wp.float32),
                   size: wp.array(dtype=wp.float32), is_leaf: wp.array(dtype=wp.int32),
                   lo: wp.array(dtype=wp.int32), hi: wp.array(dtype=wp.int32),
                   escape: wp.array(dtype=wp.int32),
                   pts: wp.array(dtype=wp.vec3), pm: wp.array(dtype=wp.float32),
                   queries: wp.array(dtype=wp.vec3), n_nodes: wp.int32,
                   theta: wp.float32, eps2: wp.float32, G: wp.float32,
                   phi: wp.array(dtype=wp.float32)):
        i = wp.tid()
        q = queries[i]
        p = float(0.0)
        idx = int(0)
        while idx < n_nodes:
            c = com[idx]
            dx = c[0] - q[0]; dy = c[1] - q[1]; dz = c[2] - q[2]
            d = wp.sqrt(dx * dx + dy * dy + dz * dz + eps2)
            if is_leaf[idx] == 1:
                a = lo[idx]; b = hi[idx]
                for j in range(a, b):
                    pj = pts[j]
                    ex = pj[0] - q[0]; ey = pj[1] - q[1]; ez = pj[2] - q[2]
                    dj = wp.sqrt(ex * ex + ey * ey + ez * ez + eps2)
                    p += G * pm[j] / dj
                idx = int(escape[idx])
            elif size[idx] / d < theta:
                p += G * mass[idx] / d
                idx = int(escape[idx])
            else:
                idx = idx + 1
        phi[i] = p

    @wp.kernel
    def _bh_count_kernel(com: wp.array(dtype=wp.vec3), size: wp.array(dtype=wp.float32),
                         is_leaf: wp.array(dtype=wp.int32), lo: wp.array(dtype=wp.int32),
                         hi: wp.array(dtype=wp.int32), escape: wp.array(dtype=wp.int32),
                         level: wp.array(dtype=wp.int32),
                         queries: wp.array(dtype=wp.vec3), n_nodes: wp.int32,
                         theta: wp.float32, eps2: wp.float32,
                         work: wp.array(dtype=wp.int32), depth: wp.array(dtype=wp.int32)):
        i = wp.tid()
        q = queries[i]
        w = int(0); dmax = int(0); idx = int(0)
        while idx < n_nodes:
            if level[idx] > dmax:
                dmax = level[idx]
            c = com[idx]
            dx = c[0] - q[0]; dy = c[1] - q[1]; dz = c[2] - q[2]
            d = wp.sqrt(dx * dx + dy * dy + dz * dz + eps2)
            if is_leaf[idx] == 1:
                w += hi[idx] - lo[idx]
                idx = int(escape[idx])
            elif size[idx] / d < theta:
                w += 1
                idx = int(escape[idx])
            else:
                idx = idx + 1
        work[i] = w; depth[i] = dmax


def _to_vec3(a):
    return wp.array(np.ascontiguousarray(a, dtype=np.float32), dtype=wp.vec3, device="cuda")


def upload(lin):
    """Upload a LinTree's arrays to the GPU once (reused across query launches)."""
    return dict(
        com=_to_vec3(lin.com),
        mass=wp.array(lin.mass.astype(np.float32), dtype=wp.float32, device="cuda"),
        size=wp.array(lin.size.astype(np.float32), dtype=wp.float32, device="cuda"),
        is_leaf=wp.array(lin.is_leaf.astype(np.int32), dtype=wp.int32, device="cuda"),
        lo=wp.array(lin.lo.astype(np.int32), dtype=wp.int32, device="cuda"),
        hi=wp.array(lin.hi.astype(np.int32), dtype=wp.int32, device="cuda"),
        escape=wp.array(lin.escape.astype(np.int32), dtype=wp.int32, device="cuda"),
        pts=_to_vec3(lin.points),
        pm=wp.array(lin.masses.astype(np.float32), dtype=wp.float32, device="cuda"),
        level=wp.array(lin.level.astype(np.int32), dtype=wp.int32, device="cuda"),
        n_nodes=int(lin.n_nodes),
    )


def bh_count_gpu(dev, queries, theta=0.5, eps=1e-4):
    """Per-query interaction count and max descent depth on the GPU (the work metric)."""
    qd = _to_vec3(queries)
    nq = len(queries)
    work = wp.zeros(nq, dtype=wp.int32, device="cuda")
    depth = wp.zeros(nq, dtype=wp.int32, device="cuda")
    wp.launch(_bh_count_kernel, dim=nq, inputs=[
        dev["com"], dev["size"], dev["is_leaf"], dev["lo"], dev["hi"], dev["escape"],
        dev["level"], qd, dev["n_nodes"], float(theta), float(eps * eps), work, depth],
        device="cuda")
    wp.synchronize()
    return work.numpy(), depth.numpy()


def bh_field_gpu(dev, queries, theta=0.5, eps=1e-4, G=1.0, synchronize=True):
    """Evaluate the Barnes-Hut potential at `queries` on the GPU. Returns phi (numpy)."""
    qd = _to_vec3(queries)
    nq = len(queries)
    phi = wp.zeros(nq, dtype=wp.float32, device="cuda")
    wp.launch(_bh_kernel, dim=nq, inputs=[
        dev["com"], dev["mass"], dev["size"], dev["is_leaf"], dev["lo"], dev["hi"],
        dev["escape"], dev["pts"], dev["pm"], qd, dev["n_nodes"],
        float(theta), float(eps * eps), float(G), phi], device="cuda")
    if synchronize:
        wp.synchronize()
    return phi.numpy()


if __name__ == "__main__":
    import time
    from . import octree as ot
    print(f"Warp GPU available: {_HAS_WARP}")
    assert _HAS_WARP, "no CUDA device for Warp"
    rng = np.random.default_rng(0)
    n = 20000
    pts = rng.random((n, 3)); mas = rng.random(n) + 0.1
    tree = ot.build(pts, mas); lin = ot.dfs_linearize(tree)
    q = rng.random((1000, 3))
    cpu, _, _ = ot.bh_field(lin, q, theta=0.5)
    dev = upload(lin)
    gpu = bh_field_gpu(dev, q, theta=0.5)
    direct = ot.direct_field(pts, mas, q)
    print(f"1) GPU vs CPU max rel diff = {np.abs(gpu-cpu).max()/np.abs(cpu).max():.2e}")
    print(f"   GPU vs direct mean rel err = {(np.abs(gpu-direct)/np.abs(direct)).mean():.2e}")
    print("2) GPU full-activation wall-clock:")
    for nn in (50000, 200000, 1000000):
        P = rng.random((nn, 3)); M = rng.random(nn) + 0.1
        tr = ot.build(P, M); ln = ot.dfs_linearize(tr); dv = upload(ln)
        Q = P.astype(np.float32)
        bh_field_gpu(dv, Q[:1000], theta=0.5)
        t0 = time.time(); bh_field_gpu(dv, Q, theta=0.5); dt = time.time() - t0
        print(f"   n={nn:>8d}  nodes={tr.n_nodes:>8d}  {nn/dt/1e6:.2f}M queries/s")
