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
