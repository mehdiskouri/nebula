"""
Render-cloud generation — the dense Gaussian splats that ride the coarse physics (V1.9; V3.9).

**Role: a FAITHFUL PREVIEW, not the final render.** Nebula's deliverable is the causal state
(morphology + derived physical fields); the beauty render is a downstream path tracer (Omniverse)
consuming the USD+OpenVDB export (`geometry.export`). This splat view is the in-engine *preview* — a
transparent window onto what the simulation computed: every splat's position comes from the grown
skeleton and every colour is DERIVED (`geometry.appearance`: wood-layer albedo, char darkening,
blackbody ember). It does NOT beautify, and `show_physics()` recolours by a raw derived field
(T / χ / layer) so the preview doubles as an honest debugger of the sim.

The architecture's dual cloud: a COARSE physics cloud (the tree skeleton + the fire grid) drives the
DENSE render cloud. This module GENERATES it (never captured — a captured cloud is a hollow shell;
the surface AND the interior revealed at char/fracture are generated) and binds it to the coarse
skeleton so it rides the XPBD topple via the verified rotation-aware LBS (`dualcloud`, V1.9). The
flame/smoke are emissive/absorptive splats from the gas/T/soot grid (blackbody + Beer–Lambert, V3.3).
Splats carry NO physics state, so the preview cannot perturb the simulation. Surfaces come from
`mesh_export.tube_mesh` (generalized cylinders — the de-blob). Output feeds `gaussian_rasterizer`.
"""
import numpy as np

from ..geometry import mesh_export as me
from ..geometry import appearance as ap


def field_colors(values, cmap="inferno", vmin=None, vmax=None):
    """Map a per-splat scalar field to RGB via a named colormap (a faithful field visualization)."""
    import matplotlib
    import matplotlib.colors as mcolors
    v = np.asarray(values, float)
    norm = mcolors.Normalize(v.min() if vmin is None else vmin, v.max() if vmax is None else vmax)
    return matplotlib.colormaps[cmap](norm(v))[:, :3]


def show_physics(cloud, values, cmap="inferno", vmin=None, vmax=None):
    """Recolour a splat cloud by a DERIVED FIELD (temperature, char χ, layer, …) — the
    'show-the-physics' diagnostic mode: the preview as an honest debugger of the simulation, NOT a
    beauty shot. Returns a copy with `color` replaced; geometry/opacity untouched."""
    out = dict(cloud)
    out["color"] = field_colors(values, cmap, vmin, vmax)
    return out


def _tangent_frame(n):
    """Two orthonormal tangents per normal n (N,3)."""
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    ref = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([0, 0, 1.0]), np.array([1.0, 0, 0]))
    t1 = np.cross(n, ref); t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12)
    t2 = np.cross(n, t1)
    return t1, t2, n


def disk_covariance(normals, s_tan, s_norm):
    """Per-splat 3×3 covariance of a flat disk: wide (s_tan) in the tangent plane, thin (s_norm)
    along the normal. Σ = s_tan²(t1 t1ᵀ + t2 t2ᵀ) + s_norm² n nᵀ."""
    t1, t2, n = _tangent_frame(np.asarray(normals, float))
    st = np.asarray(s_tan, float)[:, None, None]
    sn = np.asarray(s_norm, float)[:, None, None]
    def outer(a):
        return a[:, :, None] * a[:, None, :]
    return st ** 2 * (outer(t1) + outer(t2)) + sn ** 2 * outer(n)


def tree_splats(tree, chi_grid=None, T_grid=None, fire_origin=None, fire_spacing=None,
                size_scale=1.0):
    """Generate surface splats for the tree from its tube mesh. Returns dict of arrays + `vnode`
    (skeleton node per splat, for skinning) and base wood colors."""
    verts, faces, vnode = me.tube_mesh(tree)
    # per-vertex outward normal = radial from the skeleton node it rides
    nodepos = tree.pos[vnode]
    nrm = verts - nodepos
    nrm[np.linalg.norm(nrm, axis=1) < 1e-9] = np.array([0, 0, 1.0])
    # splat size from the local twig radius (bigger branches → bigger splats)
    r = tree.radius[vnode]
    s_tan = np.clip(0.9 * r, 0.01, None) * size_scale
    s_norm = 0.25 * s_tan
    cov = disk_covariance(nrm, s_tan, s_norm)
    # DERIVED appearance: wood-layer base, char, ember
    rho, _, rb_at, rh_at, _ = me.segment_field(verts, tree)
    base = np.tile(me.COL_SAPWOOD / 255.0, (len(verts), 1))
    base[rho > rb_at] = me.COL_BARK / 255.0
    base[rho < rh_at] = me.COL_HEARTWOOD / 255.0
    chi = Tv = None
    if chi_grid is not None and fire_origin is not None:
        chi = np.clip(me._sample(chi_grid, fire_origin, fire_spacing, verts), 0, 1)
        Tv = me._sample(T_grid, fire_origin, fire_spacing, verts) if T_grid is not None else None
    surf = ap.surface_appearance(base, T=Tv, chi=chi)
    color = surf["albedo"] + surf["emission"]            # HDR: albedo + blackbody ember
    op = np.full(len(verts), 0.97)
    return {"means": verts, "cov": cov, "color": color, "opacity": op, "vnode": vnode,
            "normal": nrm}


def canopy_splats(canopy, fuel=None, size=0.06):
    """Generate leaf splats: green foliage, browning/charring/ember as the leaf burns (if `fuel`)."""
    n = canopy.n
    nrm = canopy.normal.copy()
    area = canopy.area if fuel is None else fuel.area
    s_tan = np.sqrt(np.clip(area, 1e-6, None)) * 3.0 * size / np.sqrt(canopy.area.mean() + 1e-9)
    cov = disk_covariance(nrm, s_tan, 0.15 * s_tan)
    leaf = np.tile(np.array([0.18, 0.42, 0.12]), (n, 1))     # foliage green
    color = leaf.copy(); op = np.full(n, 0.85)
    if fuel is not None:
        chi = np.clip(fuel.char, 0, 1)[:, None]
        # browning then char then ember from the leaf temperature proxy (ignited → hot)
        brown = np.array([0.35, 0.22, 0.05])
        color = (1 - chi) * leaf + chi * np.array([0.05, 0.045, 0.04])   # → char black
        color = (1 - 0.4 * chi) * color + 0.4 * chi * brown
        hot = fuel.ignited & (fuel.mass > 0)
        color[hot] = color[hot] + ap.ember_emission(np.full(int(hot.sum()), 1200.0),
                                                     chi=np.ones(int(hot.sum())))
        op = np.where(fuel.mass > 0, 0.85, 0.0)              # burnt leaves vanish
    keep = op > 0
    return {"means": canopy.pos[keep], "cov": cov[keep], "color": color[keep],
            "opacity": op[keep], "twig_node": canopy.twig_node[keep]}


def flame_splats(state, origin, spacing, T_hot=700.0, soot_key="soot", max_splats=20000):
    """Emissive flame splats (blackbody) + absorptive smoke splats (soot), from the fire grid."""
    T = state["T"]; soot = state.get(soot_key, np.zeros_like(T))
    hot = T > T_hot
    out_means, out_cov, out_col, out_op = [], [], [], []
    if hot.any():
        ij = np.argwhere(hot)
        if len(ij) > max_splats:
            keep = np.argsort(-T[hot])[:max_splats]; ij = ij[keep]
        pos = origin[None, :] + ij * spacing
        Tv = T[ij[:, 0], ij[:, 1], ij[:, 2]]
        col = ap.ember_emission(Tv, chi=np.ones(len(Tv))) * 2.0       # HDR emissive
        s = np.full(len(pos), 0.7 * spacing)
        cov = (s[:, None, None] ** 2) * np.eye(3)[None]
        out_means.append(pos); out_cov.append(cov); out_col.append(col)
        out_op.append(np.clip((Tv - T_hot) / 600.0, 0.15, 0.7))
    smoky = soot > 0.02
    if smoky.any():
        ij = np.argwhere(smoky)
        if len(ij) > max_splats:
            keep = np.argsort(-soot[smoky])[:max_splats]; ij = ij[keep]
        pos = origin[None, :] + ij * spacing
        sv = soot[ij[:, 0], ij[:, 1], ij[:, 2]]
        col = np.tile(np.array([0.05, 0.05, 0.055]), (len(pos), 1))   # dark smoke
        s = np.full(len(pos), 1.1 * spacing)
        cov = (s[:, None, None] ** 2) * np.eye(3)[None]
        out_means.append(pos); out_cov.append(cov); out_col.append(col)
        out_op.append(ap.smoke_alpha(sv, kappa=2.0) * 0.5)
    if not out_means:
        return None
    return {"means": np.vstack(out_means), "cov": np.vstack(out_cov),
            "color": np.vstack(out_col), "opacity": np.concatenate(out_op)}


def merge(*clouds):
    """Concatenate splat clouds into one (means, cov, color, opacity) for a single render call."""
    clouds = [c for c in clouds if c is not None and len(c["means"])]
    return {k: np.concatenate([c[k] for c in clouds], 0) for k in ("means", "cov", "color", "opacity")}


# ----------------------------------------------------------- dual-cloud skinning (V1.9 reuse)
def bind_to_skeleton(splat_means, tree, k=4):
    """Bind each splat to its k nearest skeleton nodes (inverse-distance weights). Reuses the V1.9
    skinning machinery so splats ride the coarse physics cloud's deformation."""
    import sys
    sys.path.insert(0, "/workspace/nebula/src/verification/oracles")
    import dualcloud as dc
    return dc.bind_weights(splat_means, tree.pos, k=k)


def skin(splat_means, idx, w, tree, x_now, R_now):
    """Rotation-aware LBS of the splats to the deformed skeleton (x_now, per-node rot R_now)."""
    import sys
    sys.path.insert(0, "/workspace/nebula/src/verification/oracles")
    import dualcloud as dc
    return dc.skin_lbs(splat_means, idx, w, tree.pos, x_now, R_now)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/workspace/nebula/src/verification/oracles")
    from ..operators.growth import grow_tree, GrowthParams
    from ..operators import canopy as cano
    import dualcloud as dc
    np.seterr(all="ignore")

    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    sp = tree_splats(tree)
    print(f"1) tree splats: {len(sp['means'])} (cov {sp['cov'].shape}, HDR color range "
          f"[{sp['color'].min():.2f},{sp['color'].max():.2f}])")
    assert len(sp["means"]) > 1000 and sp["cov"].shape[1:] == (3, 3)

    can = cano.generate_canopy(tree, cano.CanopyParams(), seed=7)
    cs = canopy_splats(can)
    print(f"2) canopy splats: {len(cs['means'])} green leaves")
    assert len(cs["means"]) > 1000

    # 3) dual-cloud skinning: bind splats, rigidly rotate the whole skeleton, splats follow rigidly.
    idx, w = dc.bind_weights(sp["means"], tree.pos, k=4)
    th = np.deg2rad(30); Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    x_now = tree.pos @ Rz.T
    R_now = np.tile(Rz, (tree.n, 1, 1))
    skinned = dc.skin_lbs(sp["means"], idx, w, tree.pos, x_now, R_now)
    expected = sp["means"] @ Rz.T
    err = np.linalg.norm(skinned - expected, axis=1).max() / np.ptp(tree.pos[:, 2])
    print(f"3) rigid-rotation skinning error (frac of tree height) = {err:.2e}")
    assert err < 1e-9
    print("\nsplat generation + skinning self-checks passed.")
