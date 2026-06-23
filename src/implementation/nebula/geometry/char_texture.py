"""
Char "alligator" crackle — texture as a DERIVED simulation output (ARCHITECTURE Part I; V3.6).

Charred wood cracks into the characteristic polygonal "alligator skin" because the shrinking,
desiccating char layer is loaded in tension by the wood beneath. The crack-network **cell size is
set by the char-layer thickness** (the mud-crack / thermal-crack thickness law: spacing ∝ char
depth — the V3.6 oracle `shrinkage_crack_ref`), and crack DEPTH grows with the char fraction χ. So
the crackle is a DERIVED function of the char state — generated here as a Voronoi-edge (Worley)
cellular field whose feature spacing equals the predicted crack spacing — and exported as a
displacement/normal/AO map the path tracer reads (it does not appear on unburnt wood).

Deterministic (feature jitter hashed from a stable key, V1.8). Isotropic 2-D network (unlike the
vertical bark fissures of `bark_texture`).
"""
import numpy as np
from scipy.spatial import cKDTree

from ..core.determinism import rng_from_key


def crack_spacing(char_depth, c=6.0):
    """Alligator-cell spacing ∝ char depth (thickness law; matches the V3.6 oracle)."""
    return c * np.asarray(char_depth, float)


def crack_depth(chi, char_thickness, k=1.0):
    """Crack depth grows with char fraction χ, saturating at the char-layer thickness."""
    return np.clip(k * char_thickness * np.clip(chi, 0, 1), 0.0, char_thickness)


def _worley_edges(query_pts, feature_pts, crack_width):
    """Voronoi-edge (crack) intensity in [0,1]: 1 where the two nearest feature points are
    near-equidistant (a cell boundary), 0 deep inside a cell."""
    d, _ = cKDTree(feature_pts).query(query_pts, k=2)
    return np.clip(1.0 - (d[:, 1] - d[:, 0]) / np.maximum(crack_width, 1e-9), 0.0, 1.0)


def crack_field_2d(n, char_depth, world_size, seed=0, crack_width_frac=0.28):
    """A 2-D char-crack intensity raster (n×n) with cell size = crack_spacing(char_depth). Used to
    verify the thickness-law scaling (the oracle measures its dominant cell size)."""
    spacing = float(crack_spacing(char_depth))
    ncell = max(int(round(world_size / spacing)), 2)
    rng = rng_from_key("charcrack2d", seed, round(char_depth, 6))
    gi, gj = np.meshgrid(np.arange(ncell), np.arange(ncell), indexing="ij")
    fpts = (np.stack([gi.ravel(), gj.ravel()], 1) + rng.random((ncell * ncell, 2))) * spacing
    qs = np.stack(np.meshgrid(np.linspace(0, world_size, n), np.linspace(0, world_size, n),
                              indexing="ij"), -1).reshape(-1, 2)
    return _worley_edges(qs, fpts, crack_width_frac * spacing).reshape(n, n)


def char_relief(tree, verts, vert_node, chi, seed=0, char_thickness=0.01, crack_width_frac=0.28):
    """Per-vertex DERIVED char crackle on the charred surface. `chi` (V,) is the per-vertex char
    fraction. Returns displacement (inward at cracks), relief-perturbed normal, crack intensity.
    Cracks only where chi>0; cell size scales with the local char depth."""
    chi = np.clip(np.asarray(chi, float), 0, 1)
    charred = chi > 0.05
    disp = np.zeros(len(verts))
    fissure = np.zeros(len(verts))
    normal = verts - tree.pos[vert_node]
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9)
    if charred.any():
        depth = crack_depth(chi[charred], char_thickness)          # local char depth
        spacing = np.maximum(crack_spacing(depth), 1e-4)
        # 3-D Worley feature points at the median spacing over the charred surface (the cell scale)
        sp = float(np.median(spacing))
        lo = verts[charred].min(0); hi = verts[charred].max(0)
        rng = rng_from_key("charcrack3d", seed, round(sp, 5))
        dims = np.maximum(((hi - lo) / sp).astype(int) + 1, 2)
        gi = np.indices(dims).reshape(3, -1).T
        fpts = lo + (gi + rng.random(gi.shape)) * sp
        edge = _worley_edges(verts[charred], fpts, crack_width_frac * sp)
        fissure[charred] = edge
        disp[charred] = depth * edge                               # inward at cracks
    verts_relief = verts - normal * disp[:, None]
    return {"displacement": disp, "verts_relief": verts_relief, "normal": normal, "fissure": fissure}


def apply_crack_albedo(albedo, fissure, darken=0.6):
    """Darken albedo in the char cracks (deep shadow at the crack bottoms) — derived AO."""
    return albedo * (1.0 - darken * np.asarray(fissure, float)[:, None])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/nebula/src/verification/oracles")
    import shrinkage_crack_ref as sc
    from ..operators.growth import grow_tree, GrowthParams
    from .mesh_export import tube_mesh
    np.seterr(all="ignore")

    # 1) crack cell size scales with char depth (the thickness law; measured vs oracle prediction).
    print("1) measured alligator-cell size vs char depth (∝ depth):")
    sizes = []
    for h in (0.004, 0.008, 0.016):
        field = crack_field_2d(128, h, world_size=1.0, seed=7)
        est = sc.measure_cell_size(field, spacing_px=1.0 / 128)
        sizes.append(est)
        print(f"   depth {h:.3f}: predicted spacing {float(sc.crack_spacing(h)):.3f}, measured cell {est:.3f}")
    sl = np.polyfit([0.004, 0.008, 0.016], sizes, 1)[0]
    assert sizes[2] > sizes[1] > sizes[0] and sl > 0     # cell size grows with depth

    # 2) crack depth grows with χ.
    cd = crack_depth(np.array([0.0, 0.3, 0.6, 1.0]), 0.01)
    print(f"2) crack depth vs χ: {np.round(cd,4)} (increasing)")
    assert np.all(np.diff(cd) >= 0) and cd[-1] > cd[0]

    # 3) on a tree: cracks only on charred verts; deterministic.
    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    verts, faces, vnode = tube_mesh(tree)
    chi = np.where(verts[:, 2] < tree.pos[:, 2].min() + 0.3 * np.ptp(tree.pos[:, 2]), 0.9, 0.0)  # charred base
    rel = char_relief(tree, verts, vnode, chi, seed=7)
    rel2 = char_relief(tree, verts, vnode, chi, seed=7)
    print(f"3) char relief: {(rel['fissure']>0.05).sum()} cracked verts (all in charred base); "
          f"unburnt disp {rel['displacement'][chi==0].max():.4f} (≈0); deterministic "
          f"{np.array_equal(rel['displacement'], rel2['displacement'])}")
    assert rel["displacement"][chi == 0].max() == 0 and (rel["fissure"][chi > 0] > 0).any()
    assert np.array_equal(rel["displacement"], rel2["displacement"])
    print("\nchar_texture (derived alligator crackle) self-checks passed.")
