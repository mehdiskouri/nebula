"""
Bark-fissure relief — texture as a DERIVED simulation output (ARCHITECTURE Part I; V3.8).

Bark texture is not painted: secondary (radial) growth stretches the rigid outer bark over an
expanding circumference until it fails in tension into **vertical fissures**, whose depth and
spacing are set by the trunk's radius and growth increment (the V3.8 oracle `bark_morphology_ref`:
depth grows with radius & Δr; spacing ∝ bark thickness; fissures run along the grain). This module
DERIVES that relief — a per-vertex inward displacement + a perturbed normal + a fissure-darkened
albedo — from the grown tree's own state, to be EXPORTED as a displacement/normal primvar the path
tracer reads (and shown faithfully in the splat preview). No authored texture image.

Deterministic (per-node phase hashed from the lineage, V1.8). Twigs (thin) stay smooth — only the
bark-bearing trunk/branches fissure.
"""
import numpy as np

from ..core.determinism import stable_hash


def fissure_depth(radius, growth_increment, bark_thickness, c_r=3.0, c_g=50.0):
    """Derived fissure depth (matches the V3.8 oracle): grows with radius & growth, saturates at bark."""
    drive = c_r * np.asarray(radius, float) + c_g * growth_increment
    return np.clip(bark_thickness * (1.0 - np.exp(-drive)), 0.0, bark_thickness)


def fissure_count(radius, bark_thickness):
    """Number of vertical fissures around the trunk = circumference / spacing (spacing ∝ thickness)."""
    spacing = 2.0 * bark_thickness
    return np.maximum(np.round(2 * np.pi * np.asarray(radius, float) / spacing), 3).astype(int)


def _ridge(x):
    """Sharp-valleyed triangular ridge in [0,1] (fissure cross-section): 1 on ridges, 0 in cracks."""
    return 1.0 - np.abs(((x / np.pi) % 2.0) - 1.0)


def bark_relief(tree, verts, vert_node, seed=0, r_split=0.06, depth_gain=1.0):
    """Per-vertex DERIVED bark relief. Returns dict:
      displacement (V,)  inward depth (≥0) into the bark at each vertex,
      verts_relief (V,3) displaced vertex positions (along the inward radial),
      normal (V,3)       relief-perturbed outward normal,
      fissure (V,)       fissure mask/intensity in [0,1] (for albedo darkening).
    Twigs (radius ≤ r_split) get ~zero relief (smooth)."""
    gp = tree.params
    r = tree.radius[vert_node]
    nodepos = tree.pos[vert_node]
    radial = verts - nodepos
    rad_len = np.linalg.norm(radial, axis=1, keepdims=True)
    outn = radial / np.maximum(rad_len, 1e-9)
    # axis = parent->node direction (the grain / fissure direction is vertical along it)
    par = tree.parent[vert_node]
    axis = np.where((par >= 0)[:, None], tree.pos[vert_node] - tree.pos[np.maximum(par, 0)], np.array([0, 0, 1.0]))
    axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)
    # azimuth around the axis (build a stable in-plane frame)
    ref = np.where(np.abs(axis[:, 2:3]) < 0.9, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
    u = np.cross(axis, ref); u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
    v = np.cross(axis, u)
    theta = np.arctan2(np.einsum("ij,ij->i", outn, v), np.einsum("ij,ij->i", outn, u))
    # along-axis coordinate (so fissures meander slightly with height, not perfectly straight)
    s = np.einsum("ij,ij->i", verts - nodepos, axis)
    nf = fissure_count(r, gp.bark_thickness)
    # deterministic per-node phase + slight per-node fissure jitter (hashed lineage)
    phase = np.array([stable_hash("bark", seed, int(n)) % 1000 for n in vert_node]) / 1000.0 * 2 * np.pi
    pattern = _ridge(nf * (theta + 0.15 * np.sin(6.0 * s)) + phase)        # 0 in cracks, 1 on ridges
    depth = fissure_depth(r, gp.growth_per_season, gp.bark_thickness) * depth_gain
    is_bark = (r > r_split).astype(float)
    disp = is_bark * depth * (1.0 - pattern)                              # inward at the cracks
    verts_relief = verts - outn * disp[:, None]
    # perturb the normal toward the crack walls (tangential gradient of the relief)
    dpattern = np.cos(nf * theta + phase)                                 # ∂pattern/∂θ proxy
    normal = outn - (is_bark * depth * 8.0 * dpattern)[:, None] * v
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-9)
    fissure = is_bark * (1.0 - pattern)
    return {"displacement": disp, "verts_relief": verts_relief, "normal": normal, "fissure": fissure}


def apply_fissure_albedo(albedo, fissure, darken=0.45):
    """Darken albedo in the fissure valleys (less light reaches the crack bottoms) — derived AO."""
    return albedo * (1.0 - darken * np.asarray(fissure, float)[:, None])


if __name__ == "__main__":
    from ..operators.growth import grow_tree, GrowthParams
    from .mesh_export import tube_mesh
    np.seterr(all="ignore")
    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    verts, faces, vnode = tube_mesh(tree)

    rel = bark_relief(tree, verts, vnode, seed=7)
    bark = tree.radius[vnode] > 0.06
    twig = tree.radius[vnode] <= 0.06
    print(f"1) relief on {bark.sum()} bark verts, {twig.sum()} twig verts")
    print(f"   bark displacement mean {rel['displacement'][bark].mean():.4f} > twig {rel['displacement'][twig].mean():.4f}")
    assert rel["displacement"][bark].mean() > 5 * (rel["displacement"][twig].mean() + 1e-9)

    # fissure depth scales with trunk radius (derived, matches the oracle scaling)
    d = fissure_depth(np.array([0.05, 0.1, 0.2, 0.4]), 0.005, 0.018)
    print(f"2) fissure depth vs radius {np.round(d,4)} (increasing)")
    assert np.all(np.diff(d) > 0)

    # determinism
    rel2 = bark_relief(tree, verts, vnode, seed=7)
    print(f"3) determinism: identical relief = {np.array_equal(rel['displacement'], rel2['displacement'])}")
    assert np.array_equal(rel["displacement"], rel2["displacement"])

    # fissures darken albedo in the valleys
    alb = np.tile(np.array([0.4, 0.26, 0.13]), (len(verts), 1))
    alb2 = apply_fissure_albedo(alb, rel["fissure"])
    print(f"4) fissure albedo darkening: mean luminance {alb.mean():.3f} -> {alb2.mean():.3f}")
    assert alb2.mean() < alb.mean()
    print("\nbark_texture (derived fissure relief) self-checks passed.")
