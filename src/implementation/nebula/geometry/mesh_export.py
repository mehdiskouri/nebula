"""
Mesh extraction + glTF export (ARCHITECTURE Foundational thesis; Decision #2).

The ONLY place in Nebula a mesh exists. Everything upstream is implicit (the SDF) or the
hypergraph; here marching cubes extracts the iso-surface of the grown tree SDF and writes glTF
(.glb). Crucially, COLOUR IS DERIVED, not authored (Foundational thesis: "Color and texture are
derived properties, outputs of simulation"): each vertex's colour comes from the simulated char
fraction chi (blackening), the temperature T (ember glow), and the wood layer (bark/sapwood/
heartwood) -- the time-record of the front against the fire, not a painted texture.

skimage.measure.marching_cubes for the iso-surface; trimesh + pygltflib for the .glb.
"""
import numpy as np
import trimesh
from skimage import measure
from scipy.ndimage import map_coordinates

from ..fields.sdf import segment_field

# derived base colours (RGB 0-255): the wood layers, char, and ember.
COL_BARK = np.array([101, 67, 33], float)
COL_SAPWOOD = np.array([193, 154, 107], float)
COL_HEARTWOOD = np.array([120, 72, 40], float)
COL_CHAR = np.array([22, 20, 18], float)
COL_EMBER = np.array([255, 110, 25], float)
COL_GROUND = np.array([88, 110, 58], float)


def tube_mesh(tree, nsides_trunk=16, nsides_twig=6, r_split=0.05):
    """Mesh the skeleton as tapered generalized cylinders that TAPER TO POINTS at the tips (ARCHITECTURE
    §III.8). Returns (verts (V,3), faces (F,3), vert_node (V,) → the skeleton node each vertex rides).

    Branch tips taper to a near-point (pointy twigs, not blunt sphere-capped stubs), the root↔trunk
    junction flows continuously through node 0 (no sphere base), and there are NO icosphere caps — the
    spheres at every node were the 'blob endings'. Each tube ring twists with a stable parallel frame
    so consecutive segments line up; child tubes start slightly inside the parent to hide the joint.
    """
    par = tree.parent
    nchild = np.zeros(tree.n, int)
    for i in range(tree.n):
        if par[i] >= 0:
            nchild[par[i]] += 1
    is_tip = nchild == 0
    rmesh = tree.radius.astype(float).copy()
    rmesh[is_tip] = np.maximum(0.0015, 0.015 * rmesh[is_tip])      # tips → near-points (pointy)

    verts, faces, vnode = [], [], []
    for i in range(tree.n):
        j = int(par[i])
        if j < 0:
            continue
        a, b = tree.pos[j].copy(), tree.pos[i]
        ra, rb = float(rmesh[j]), float(rmesh[i])
        axis = b - a
        L = np.linalg.norm(axis)
        if L < 1e-9:
            continue
        d = axis / L
        # overlap the child start into the parent so the joint has no crease/gap
        if par[j] >= 0:
            a = a - d * (0.6 * ra)
        ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(d, ref); u /= (np.linalg.norm(u) + 1e-12)
        v = np.cross(d, u)
        ns = nsides_trunk if max(ra, rb) > r_split else nsides_twig
        th = np.linspace(0, 2 * np.pi, ns, endpoint=False)
        ring = np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * v[None, :]
        base = len(verts)
        for k in range(ns):
            verts.append(a + ra * ring[k]); vnode.append(j)
        if is_tip[i]:                                              # collapse the tip ring to a point
            verts.append(b); vnode.append(i)
            for k in range(ns):
                faces.append([base + k, base + ((k + 1) % ns), base + ns])   # cone to the tip
        else:
            for k in range(ns):
                verts.append(b + rb * ring[k]); vnode.append(i)
            for k in range(ns):
                k2 = (k + 1) % ns
                faces.append([base + k, base + ns + k, base + ns + k2])
                faces.append([base + k, base + ns + k2, base + k2])
    return (np.asarray(verts, float), np.asarray(faces, np.int64), np.asarray(vnode, np.int64))


def _catmull_rom(P, sub):
    """Subdivide a polyline P (m,3) with Catmull-Rom into a smooth curve (sub points per segment).
    Returns the dense points and the fractional original-index per dense point (for radius interp)."""
    m = len(P)
    if m < 2:
        return P.copy(), np.zeros(len(P))
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])     # phantom endpoints
    out, frac = [], []
    for i in range(m - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        for s in range(sub):
            tt = s / sub
            t2, t3 = tt * tt, tt * tt * tt
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
            frac.append(i + tt)
    out.append(P[-1]); frac.append(m - 1.0)
    return np.array(out), np.array(frac)


def swept_tube_mesh(tree, sub=3, nsides_trunk=20, nsides_twig=7, r_split=0.045):
    """Mesh the tree as continuous SWEPT TUBES along whole branch paths — the organic, grown look.

    Per-segment cylinders crease at every node (the 'stacked cylinders'); instead each path (root→tip
    chain) is a single generalized cylinder: a Catmull-Rom-smoothed centreline, a parallel-transport
    frame (no twist, rings stay aligned so the surface is seamless), a smoothly interpolated radius
    that tapers to a point at the tip. Child paths begin inside the parent so the joints flow. Returns
    (verts, faces, vert_node)."""
    par = tree.parent; pos = tree.pos.astype(float); rad = tree.radius.astype(float)
    n = tree.n
    children = [[] for _ in range(n)]
    for i in range(n):
        if par[i] >= 0:
            children[par[i]].append(i)
    nchild = np.array([len(c) for c in children])
    primary = {i: (max(children[i], key=lambda c: rad[c]) if children[i] else -1) for i in range(n)}
    is_start = np.zeros(n, bool)
    for i in range(n):
        if par[i] < 0:
            is_start[i] = True
        elif primary[par[i]] != i:
            is_start[i] = True
    verts, faces, vnode = [], [], []
    for s in np.where(is_start)[0]:
        chain = []
        cur = int(s)
        while cur != -1:
            chain.append(cur)
            cur = primary[cur]
        nodes = np.array(chain)
        P = pos[nodes].astype(float).copy(); R = rad[nodes].copy()
        R[-1] = max(0.0015, 0.01 * R[-1])                   # taper the path tip to a point
        # overlap a short stub back INTO the parent at the CHILD radius (thin) so junctions don't
        # blob — child tubes must not inherit the thick parent radius at the joint.
        if par[s] >= 0:
            d = pos[s] - pos[par[s]]; d = d / (np.linalg.norm(d) + 1e-9)
            P = np.vstack([pos[s] - d * (1.3 * R[0]), P])
            R = np.concatenate([[R[0]], R]); nodes = np.concatenate([[int(s)], nodes])
        if len(nodes) < 2:
            continue
        Pd, frac = _catmull_rom(P, sub)
        Rd = np.interp(frac, np.arange(len(nodes)), R)
        # taper the radius CONTINUOUSLY base→tip (the pipe model leaves uniform-thick limb 'stubs');
        # blend with a base→tip taper, then force a monotone thinning so every branch comes to a point.
        tt = (frac - frac[0]) / (frac[-1] - frac[0] + 1e-9)
        Rd = 0.55 * Rd + 0.45 * (R[0] * (1 - tt) + R[-1] * tt)
        Rd = np.minimum.accumulate(Rd)
        nd = nodes[np.clip(np.round(frac).astype(int), 0, len(nodes) - 1)]
        # parallel-transport frame along the dense centreline
        tang = np.gradient(Pd, axis=0)
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0, 2]) < 0.9 else np.array([1.0, 0, 0])
        u0 = np.cross(tang[0], ref); u0 /= np.linalg.norm(u0) + 1e-9
        U = [u0]
        for t in range(1, len(Pd)):
            u = U[-1] - tang[t] * (U[-1] @ tang[t])          # project onto the new normal plane
            nrm = np.linalg.norm(u)
            U.append(u / nrm if nrm > 1e-6 else U[-1])
        U = np.array(U); V = np.cross(tang, U)
        ns = nsides_trunk if R.max() > r_split else nsides_twig
        th = np.linspace(0, 2 * np.pi, ns, endpoint=False)
        ring = (np.cos(th)[None, :, None] * U[:, None, :] + np.sin(th)[None, :, None] * V[:, None, :])
        rverts = Pd[:, None, :] + Rd[:, None, None] * ring   # (T, ns, 3)
        T = len(Pd); b0 = len(verts)
        for t in range(T):
            for k in range(ns):
                verts.append(rverts[t, k]); vnode.append(int(nd[t]))
        for t in range(T - 1):
            for k in range(ns):
                k2 = (k + 1) % ns
                a = b0 + t * ns + k; bb = b0 + t * ns + k2
                c = b0 + (t + 1) * ns + k; dd = b0 + (t + 1) * ns + k2
                faces.append([a, c, dd]); faces.append([a, dd, bb])
        tip = len(verts); verts.append(Pd[-1]); vnode.append(int(nd[-1]))   # close the tip to a point
        for k in range(ns):
            faces.append([b0 + (T - 1) * ns + k, tip, b0 + (T - 1) * ns + ((k + 1) % ns)])
    return np.asarray(verts, float), np.asarray(faces, np.int64), np.asarray(vnode, np.int64)


def leaf_cards(canopy, size=0.12, seed=7, curl=0.28):
    """Flat leaf BLADES attached to the twigs: a diamond (narrow base → widest middle → pointed tip)
    per leaf, oriented outward + a gravity droop + a phyllotactic tilt. Returns (verts, faces,
    leaf_id) where leaf_id maps each vertex to its leaf (for per-leaf char colour + dropping).
    Every blade's base sits ON its twig, so none float detached."""
    from ..core.determinism import rng_from_key
    n = canopy.n
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
    rng = rng_from_key("leafcard", seed, n)
    base = canopy.pos.astype(float)
    nrm = canopy.normal / (np.linalg.norm(canopy.normal, axis=1, keepdims=True) + 1e-9)
    # leaf direction: outward + downward droop + a little scatter (phyllotaxis already in azimuth)
    ld = nrm + np.array([0, 0, -0.45]) + 0.35 * rng.standard_normal((n, 3))
    ld /= (np.linalg.norm(ld, axis=1, keepdims=True) + 1e-9)
    up = np.array([0, 0, 1.0])
    wd = np.cross(ld, up); wn = np.linalg.norm(wd, axis=1, keepdims=True)
    wd = np.where(wn < 1e-6, np.cross(ld, np.array([1.0, 0, 0])), wd)
    wd /= (np.linalg.norm(wd, axis=1, keepdims=True) + 1e-9)
    length = (size * (0.7 + 0.6 * rng.random(n)))[:, None]
    width = 0.42 * length
    # smooth lanceolate leaf template: (s along midrib, half-width fraction) — rounded, pointed tip
    T = np.array([[0.0, 0.05], [0.0, -0.05], [0.28, 0.5], [0.28, -0.5],
                  [0.60, 0.42], [0.60, -0.42], [1.0, 0.0]])
    Tf = np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3], [4, 6, 5]])
    m = len(T)
    cn = np.cross(ld, wd)                            # leaf-card normal (for out-of-plane curl)
    cn /= (np.linalg.norm(cn, axis=1, keepdims=True) + 1e-9)
    s = T[:, 0][None, :, None]; wf = T[:, 1][None, :, None]
    # cup the blade out of plane: edges lift, midrib dips, tip droops → leaves have VOLUME, not flat
    coff = curl * (T[:, 1] ** 2 - 0.12 - 0.25 * T[:, 0])[None, :, None]
    verts = (base[:, None, :] + ld[:, None, :] * (length[:, None] * s)
             + wd[:, None, :] * (width[:, None] * wf)
             + cn[:, None, :] * (length[:, None] * coff)).reshape(-1, 3)
    idx = np.arange(n)
    faces = (Tf[None, :, :] + (m * idx)[:, None, None]).reshape(-1, 3)
    leaf_id = np.repeat(idx, m)
    return verts.astype(float), faces.astype(np.int64), leaf_id


def tube_vertex_colors(tree, vert_node, chi_v=None, T_v=None):
    """Per-vertex RGBA for the tube mesh: bark base (the only thing visible on an intact tree),
    then char chi blackening + ember T glow (derived from the simulation)."""
    n = len(vert_node)
    # subtle bark variation by node radius (thicker -> slightly darker/older bark)
    rr = tree.radius[vert_node]
    t = np.clip((rr - rr.min()) / (np.ptp(rr) + 1e-9), 0, 1)[:, None]
    col = (1 - t) * np.array([186, 150, 110.0]) + t * np.array([120, 84, 50.0])  # twig-tan -> trunk-bark
    if chi_v is not None:
        chi = np.clip(chi_v, 0, 1)[:, None]
        col = (1 - chi) * col + chi * COL_CHAR
        if T_v is not None:
            ember = (np.clip((T_v - 500) / 400, 0, 1)[:, None]) * chi
            col = (1 - ember) * col + ember * COL_EMBER
    rgba = np.empty((n, 4), np.uint8)
    rgba[:, :3] = np.clip(col, 0, 255).astype(np.uint8); rgba[:, 3] = 255
    return rgba


def extract_mesh(sdf, level=0.0):
    """Marching cubes on an SDFGrid -> (verts world-coords (V,3), faces (F,3)). Empty if no surface."""
    vals = sdf.values
    if not (vals.min() < level < vals.max()):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    verts, faces, _normals, _vals = measure.marching_cubes(
        vals, level=level, spacing=(sdf.spacing, sdf.spacing, sdf.spacing))
    verts = verts + sdf.origin[None, :]
    return verts, faces.astype(np.int64)


def _sample(grid, origin, spacing, points):
    """Trilinear sample of a scalar grid at world `points` (Q,3)."""
    idx = (np.asarray(points, float) - np.asarray(origin)) / spacing
    return map_coordinates(np.asarray(grid, float), idx.T, order=1, mode="nearest")


def derive_vertex_colors(verts, tree, chi_grid=None, T_grid=None, fire_origin=None,
                         fire_spacing=None):
    """Per-vertex RGBA (uint8) DERIVED from the simulation: layer base colour, then char chi
    blackening, then a temperature ember glow. No authored texture."""
    if len(verts) == 0:
        return np.zeros((0, 4), np.uint8)
    rho, r_at, rb_at, rh_at, _ = segment_field(verts, tree)
    col = np.tile(COL_SAPWOOD, (len(verts), 1))
    col[rho > rb_at] = COL_BARK                         # outer shell
    col[rho < rh_at] = COL_HEARTWOOD                    # old core
    if chi_grid is not None and fire_origin is not None:
        chi = np.clip(_sample(chi_grid, fire_origin, fire_spacing, verts), 0.0, 1.0)[:, None]
        col = (1.0 - chi) * col + chi * COL_CHAR        # char blackens
        if T_grid is not None:
            T = _sample(T_grid, fire_origin, fire_spacing, verts)
            ember = np.clip((T - 500.0) / 400.0, 0.0, 1.0)[:, None] * chi  # glow where hot+charring
            col = (1.0 - ember) * col + ember * COL_EMBER
    rgba = np.empty((len(verts), 4), np.uint8)
    rgba[:, :3] = np.clip(col, 0, 255).astype(np.uint8)
    rgba[:, 3] = 255
    return rgba


def ground_mesh(hf, color=COL_GROUND):
    """Triangulate a Heightfield into a colored ground mesh (trimesh)."""
    nx, ny = hf.shape
    xs = hf.origin[0] + hf.spacing * np.arange(nx)
    ys = hf.origin[1] + hf.spacing * np.arange(ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([gx.ravel(), gy.ravel(), hf.heights.ravel()], axis=1)
    idx = np.arange(nx * ny).reshape(nx, ny)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            faces.append([a, b, c]); faces.append([a, c, d])
    faces = np.asarray(faces, np.int64)
    rgba = np.tile(np.append(color, 255).astype(np.uint8), (len(verts), 1))
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)


def tree_mesh(sdf, tree, chi_grid=None, T_grid=None, fire_origin=None, fire_spacing=None, level=0.0):
    """Marching-cubes tree mesh with simulation-derived vertex colour (a trimesh.Trimesh)."""
    verts, faces = extract_mesh(sdf, level=level)
    colors = derive_vertex_colors(verts, tree, chi_grid, T_grid, fire_origin, fire_spacing)
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors, process=False)


def export_glb(path, sdf, tree, chi_grid=None, T_grid=None, fire_origin=None, fire_spacing=None,
               heightfield=None, level=0.0):
    """Export the tree (+ optional ground) to a glTF .glb. Returns the trimesh.Scene."""
    mesh = tree_mesh(sdf, tree, chi_grid, T_grid, fire_origin, fire_spacing, level=level)
    geoms = [mesh]
    if heightfield is not None:
        geoms.append(ground_mesh(heightfield))
    scene = trimesh.Scene(geoms)
    scene.export(path)
    return scene


if __name__ == "__main__":
    import tempfile, os
    from ..operators.growth import grow_tree, GrowthParams
    from ..fields.sdf import build_sdf
    from ..fields.heightfield import make_terrain

    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    sdf = build_sdf(tree)
    hf = make_terrain(seed=1, size=4.0)

    # a synthetic char/T field on a coarse fire grid around the trunk base, to exercise derived colour.
    lo, hi = tree.bounds(pad=0.1)
    fsp = 0.1
    fshape = tuple(int(np.ceil((hi[d] - lo[d]) / fsp)) + 1 for d in range(3))
    chi = np.zeros(fshape); T = np.full(fshape, 300.0)
    chi[:, :, :fshape[2] // 3] = 0.9                    # charred near the base
    T[:, :, :fshape[2] // 3] = 800.0

    verts, faces = extract_mesh(sdf)
    print(f"1) marching cubes: {len(verts)} verts, {len(faces)} faces")
    assert len(verts) > 0 and len(faces) > 0

    base = derive_vertex_colors(verts, tree)                       # wood only (no fire)
    colors = derive_vertex_colors(verts, tree, chi, T, np.asarray(lo), fsp)   # with the burn
    changed = int((np.abs(colors[:, :3].astype(int) - base[:, :3].astype(int)).sum(1) > 30).sum())
    print(f"2) derived colours: {len(colors)} RGBA; {changed} vertices changed by the burn "
          f"(char/ember) vs pristine wood -- colour is a SIMULATION OUTPUT, not authored")
    assert colors.shape == (len(verts), 4) and changed > 0

    out = os.path.join(tempfile.gettempdir(), "nebula_tree_test.glb")
    scene = export_glb(out, sdf, tree, chi, T, np.asarray(lo), fsp, heightfield=hf)
    sz = os.path.getsize(out)
    reloaded = trimesh.load(out)
    print(f"3) exported {out} ({sz} bytes); reloaded geometry count = {len(reloaded.geometry)}")
    assert sz > 0 and len(reloaded.geometry) >= 1
    print("\nmesh_export self-checks passed.")
