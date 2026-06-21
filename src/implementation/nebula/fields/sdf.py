"""
Signed distance field of the grown tree (ARCHITECTURE Foundational thesis; §III.2).

Internally Nebula is volumetric/implicit, never mesh-native: the solid tree is the union of
tapered capsules (round cones) swept along the growth skeleton with the cambium radius r(s).
The SDF is what marching cubes meshes at export (geometry.mesh_export) and what the
restriction operator homogenizes (heterogeneous bark/sapwood/heartwood cells). Growth writes
this implicit substrate; it never deposits explicit fine mesh vertices (Decision #11).

`tree_phase_grid` is the bridge to the single trust currency: it classifies each voxel of a
coarse cell into bark / sapwood / heartwood by nearest-segment radial position, so the
restriction operator sees a real layered cell (V0.1: layered media are Voigt/Reuss-exact in
the principal directions -- concentric rings are exactly that case).

Pure numpy (vectorized per-segment over its AABB); deterministic.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

# phase ids (air + the three wood layers; char is added by the fire sim, not here)
PHASE_AIR, PHASE_BARK, PHASE_SAPWOOD, PHASE_HEARTWOOD = 0, 1, 2, 3


@dataclass
class SDFGrid:
    """A signed distance field on a regular grid. values < 0 inside the solid."""
    origin: np.ndarray       # world coords of voxel center (0,0,0)
    spacing: float
    values: np.ndarray       # (Nx, Ny, Nz), signed distance (negative inside)

    @property
    def shape(self):
        return self.values.shape

    def axis_coords(self):
        return [self.origin[d] + self.spacing * np.arange(self.values.shape[d]) for d in range(3)]

    def sample(self, points):
        """Trilinear sample of the SDF at world `points` (Q,3)."""
        pts = np.asarray(points, float)
        idx = (pts - self.origin) / self.spacing            # fractional voxel coords
        return map_coordinates(self.values, idx.T, order=1, mode="nearest")


def _round_cone(P, a, b, r1, r2):
    """Exact SDF of a tapered capsule (round cone) a(r1) -> b(r2), vectorized over P (Q,3).

    The standard analytic round-cone distance (Inigo Quilez); falls back to a sphere when the
    segment degenerates or the radius difference exceeds the segment length."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ba = b - a
    l2 = float(ba @ ba)
    rr = r1 - r2
    a2 = l2 - rr * rr
    if l2 < 1e-12 or a2 <= 1e-12:                           # degenerate -> union of end spheres
        return np.minimum(np.linalg.norm(P - a, axis=1) - r1,
                          np.linalg.norm(P - b, axis=1) - r2)
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    xperp = pa * l2 - np.outer(y, ba)
    x2 = np.sum(xperp * xperp, axis=1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d = np.empty(P.shape[0])
    c1 = np.sign(z) * a2 * z2 > k
    c2 = (~c1) & (np.sign(y) * a2 * y2 < k)
    c3 = ~(c1 | c2)
    d[c1] = np.sqrt(x2[c1] + z2[c1]) * il2 - r2
    d[c2] = np.sqrt(x2[c2] + y2[c2]) * il2 - r1
    d[c3] = (np.sqrt(np.maximum(x2[c3] * a2 * il2, 0.0)) + y[c3] * rr) * il2 - r1
    return d


def build_sdf(tree, spacing=None, pad=0.25):
    """Build the tree SDF grid: the union (min) of round-cone capsules over the skeleton.

    Each segment is evaluated only inside its AABB (expanded by its radius + pad) for speed;
    voxels untouched by any segment keep a large positive distance (outside).
    """
    spacing = float(spacing if spacing is not None else tree.params.spacing * 0.5)
    lo, hi = tree.bounds(pad=pad)
    shape = tuple(int(np.ceil((hi[d] - lo[d]) / spacing)) + 1 for d in range(3))
    values = np.full(shape, 1e9, dtype=float)
    xs = lo[0] + spacing * np.arange(shape[0])
    ys = lo[1] + spacing * np.arange(shape[1])
    zs = lo[2] + spacing * np.arange(shape[2])

    for (a, b, ra, rb) in tree.segments():
        rmax = max(ra, rb) + pad
        smin = np.minimum(a, b) - rmax
        smax = np.maximum(a, b) + rmax
        i0 = max(int(np.floor((smin[0] - lo[0]) / spacing)), 0)
        j0 = max(int(np.floor((smin[1] - lo[1]) / spacing)), 0)
        k0 = max(int(np.floor((smin[2] - lo[2]) / spacing)), 0)
        i1 = min(int(np.ceil((smax[0] - lo[0]) / spacing)) + 1, shape[0])
        j1 = min(int(np.ceil((smax[1] - lo[1]) / spacing)) + 1, shape[1])
        k1 = min(int(np.ceil((smax[2] - lo[2]) / spacing)) + 1, shape[2])
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            continue
        gx, gy, gz = np.meshgrid(xs[i0:i1], ys[j0:j1], zs[k0:k1], indexing="ij")
        P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        d = _round_cone(P, a, b, ra, rb).reshape(i1 - i0, j1 - j0, k1 - k0)
        block = values[i0:i1, j0:j1, k0:k1]
        np.minimum(block, d, out=block)
    return SDFGrid(np.asarray(lo, float), spacing, values)


def segment_field(points, tree):
    """For each point (Q,3): nearest segment distance to axis (rho), interpolated radii.

    Returns (rho, r_at, rbark_at, rheart_at, react_at): the radial position and the
    interpolated outer/bark/heart radii + reaction fraction at the nearest point on the
    nearest skeleton segment -- the inputs to the layer classification."""
    pts = np.asarray(points, float)
    Q = pts.shape[0]
    best = np.full(Q, np.inf)
    r_at = np.zeros(Q); rb_at = np.zeros(Q); rh_at = np.zeros(Q); rk_at = np.zeros(Q)
    for i in range(tree.n):
        j = int(tree.parent[i])
        if j < 0:
            continue
        a = tree.pos[j]; b = tree.pos[i]
        ba = b - a
        l2 = float(ba @ ba) + 1e-12
        t = np.clip(((pts - a) @ ba) / l2, 0.0, 1.0)
        proj = a + np.outer(t, ba)
        rho = np.linalg.norm(pts - proj, axis=1)
        upd = rho < best
        best[upd] = rho[upd]
        r_at[upd] = tree.radius[j] * (1 - t[upd]) + tree.radius[i] * t[upd]
        rb_at[upd] = tree.r_bark[j] * (1 - t[upd]) + tree.r_bark[i] * t[upd]
        rh_at[upd] = tree.r_heart[j] * (1 - t[upd]) + tree.r_heart[i] * t[upd]
        rk_at[upd] = tree.reaction[j] * (1 - t[upd]) + tree.reaction[i] * t[upd]
    return best, r_at, rb_at, rh_at, rk_at


def tree_phase_grid(tree, origin, spacing, shape):
    """Per-voxel material phase (air/bark/sapwood/heartwood) on a coarse grid.

    A voxel is solid where rho <= r(s); within the solid, rho>r_bark is bark, rho<r_heart is
    heartwood, else sapwood -- the concentric-ring layering the restriction operator reads.
    """
    xs = origin[0] + spacing * np.arange(shape[0])
    ys = origin[1] + spacing * np.arange(shape[1])
    zs = origin[2] + spacing * np.arange(shape[2])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    rho, r_at, rb_at, rh_at, _ = segment_field(P, tree)
    phase = np.full(P.shape[0], PHASE_AIR, dtype=np.int64)
    solid = rho <= r_at
    phase[solid] = PHASE_SAPWOOD
    phase[solid & (rho > rb_at)] = PHASE_BARK
    phase[solid & (rho < rh_at)] = PHASE_HEARTWOOD
    return phase.reshape(shape)


if __name__ == "__main__":
    from ..operators.growth import grow_tree, GrowthParams
    np.set_printoptions(precision=3, suppress=True)

    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    sdf = build_sdf(tree)
    inside = int((sdf.values < 0).sum())
    print(f"1) SDF grid {sdf.shape}  spacing={sdf.spacing:.3f}  inside voxels={inside}")
    assert inside > 0, "tree SDF has no interior"

    # the root must be inside; a far point must be outside; sign sanity.
    samp = sdf.sample(np.array([tree.pos[0], tree.pos[0] + np.array([5.0, 5.0, 5.0])]))
    print(f"2) SDF at root = {samp[0]:.3f} (<0 inside)   far point = {samp[1]:.3f} (>0 outside)")
    assert samp[0] < 0 < samp[1]

    # phase grid: concentric layering present (some heartwood, sapwood, bark).
    phase = tree_phase_grid(tree, sdf.origin, sdf.spacing * 2, tuple(s // 2 for s in sdf.shape))
    counts = {n: int((phase == p).sum()) for n, p in
              (("air", 0), ("bark", 1), ("sapwood", 2), ("heartwood", 3))}
    print(f"3) phase voxel counts: {counts}")
    assert counts["sapwood"] > 0 and counts["bark"] > 0
    print("\nsdf self-checks passed.")
