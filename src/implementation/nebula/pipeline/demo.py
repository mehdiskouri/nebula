"""
Phase-0 visual demo (renders the tree slice to MP4).

Two complementary videos:
  - BEAUTY (pyrender, EGL offscreen): the actual 3-D tree mesh -- grow, then ignite at the base
    and watch the char climb (colour DERIVED from chi/T), then the base burns through and the tree
    TOPPLES (mesh deformed by the XPBD skeleton fall). The "tree on fire" hero shot.
  - MECHANISM (matplotlib 3-D): the machinery -- the temperature field, the trust-driven adaptive
    refinement levels filling in where it chars, and the skeleton fracturing. The "under the hood".

Reuses the verified pipeline building blocks (grow_tree, fire_domain, IMEX burn, restriction/refine,
XPBD). Headless: writes .mp4 via ffmpeg. Run:  python -m nebula.pipeline.demo --out /tmp/nebula
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # headless GL for pyrender (must precede import)

from dataclasses import dataclass, field as _field

import numpy as np

from ..operators.growth import grow_tree, GrowthParams
from ..operators import fire as fo
from ..operators import integrators as integ
from ..fields.sdf import build_sdf, tree_phase_grid, PHASE_AIR, segment_field
from ..fields.heightfield import make_terrain
from ..restriction.jensen import variance_error_scalar
from ..adaptive.refine import AdaptiveGrid, RefineParams, compute_D, proximity_field
from ..mechanics import xpbd
from ..geometry import mesh_export as me

CHI_CHAR = 0.5


@dataclass
class Recording:
    tree: object
    verts: np.ndarray            # (V,3) rest mesh vertices
    faces: np.ndarray
    nearest_node: np.ndarray     # (V,) nearest skeleton node per vertex (for topple skinning)
    heightfield: object
    fire_origin: np.ndarray
    fire_spacing: float
    vchi: list = _field(default_factory=list)   # per-burn-step (V,) vertex char fraction
    vT: list = _field(default_factory=list)      # per-burn-step (V,) vertex temperature
    T_grids: list = _field(default_factory=list)  # per-step coarse temperature (for mechanism panel)
    chi_grids: list = _field(default_factory=list)
    levels: list = _field(default_factory=list)   # per-step refinement level grid
    wood_pts: np.ndarray = None  # (W,3) wood voxel centers (mechanism scatter)
    wood_idx: tuple = None        # indices of wood voxels into the fire grid
    fall_pos: list = _field(default_factory=list)  # XPBD charred positions over the topple
    released: np.ndarray = None
    flame: list = _field(default_factory=list)      # per-step (pts (k,3), T (k,)) hot fire voxels
    coarse_block: int = 4
    fshape: tuple = None
    seed: int = 7


def record_demo(seed=7, age=None, fire_grid=28, burn_steps=48, dt=0.05, ignite_dT=800.0,
                s_break=0.25, coarse_block=4, sdf_spacing=0.03, verbose=True):
    """Run the slice while capturing per-step snapshots for rendering."""
    from scipy.spatial import cKDTree
    def log(*a):
        if verbose:
            print(*a)

    gp = GrowthParams(dim=3)
    tree = grow_tree(seed=seed, age=age, gp=gp)
    sdf = build_sdf(tree, spacing=sdf_spacing)
    hf = make_terrain(seed=seed, size=float(np.ptp(tree.pos[:, :2]) + 1.5))
    verts, faces = me.extract_mesh(sdf)
    nearest = cKDTree(tree.pos).query(verts)[1]      # nearest skeleton node per vertex (KD-tree, low mem)
    log(f"grow+mesh: {tree.n} nodes, {len(verts)} verts")

    # fire law-domain over the wood
    lo, hi = tree.bounds(pad=0.1)
    extent = float((hi - lo).max()); fsp = extent / fire_grid
    fshape = tuple(int(np.ceil((hi[d] - lo[d]) / fsp)) + 1 for d in range(3))
    phase = tree_phase_grid(tree, lo, fsp, fshape); wood = phase != PHASE_AIR
    p = fo.FireParams()
    state = {"T": np.full(fshape, 320.0), "m_s": wood.astype(float), "gas": 0.02 * wood,
             "o2": np.full(fshape, 0.23), "char": np.zeros(fshape), "q": np.zeros(fshape)}
    zc = lo[2] + fsp * np.arange(fshape[2]); zg = np.broadcast_to(zc[None, None, :], fshape)
    z0, z1 = float(tree.pos[:, 2].min()), float(tree.pos[:, 2].max())
    ig = wood & (zg < z0 + 0.18 * (z1 - z0))
    src = fo.ignition(ig, energy=p.C_V * ignite_dT * max(int(ig.sum()), 1))
    domain = fo.fire_domain(p)

    rec = Recording(tree=tree, verts=verts, faces=faces, nearest_node=nearest, heightfield=hf,
                    fire_origin=np.asarray(lo), fire_spacing=fsp, coarse_block=coarse_block,
                    fshape=fshape, seed=seed)
    wi = np.array(np.where(wood)).T                          # (W,3) wood voxel indices
    rec.wood_idx = wood
    rec.wood_pts = lo[None, :] + wi * fsp

    # adaptive grid driven by a cheap per-coarse-cell trust proxy (char fraction + Jensen eps),
    # which the pipeline confirmed tracks the true lod_trust 100% (V2.3).
    B = coarse_block
    cshape = tuple(max(s // B, 1) for s in fshape)
    ag = AdaptiveGrid.coarse(cshape, RefineParams(max_level=3))

    def coarse_fields(st, chi):
        trust = np.zeros(cshape); eps = np.zeros(cshape); Tc = np.zeros(cshape)
        for ci in range(cshape[0]):
            for cj in range(cshape[1]):
                for ck in range(cshape[2]):
                    sl = (slice(ci*B, ci*B+B), slice(cj*B, cj*B+B), slice(ck*B, ck*B+B))
                    w = wood[sl]
                    Tc[ci, cj, ck] = float(st["T"][sl].mean())
                    if w.sum() == 0:
                        continue
                    eps[ci, cj, ck] = float(variance_error_scalar(st["T"][sl].mean(), st["T"][sl].var(), p.Ta_py))
                    cf = float(((chi[sl] > CHI_CHAR) & w).mean())
                    trust[ci, cj, ck] = cf * 3.0          # char-fraction trust proxy (~lod_trust scale)
        return trust, eps, Tc

    # BURN ----------------------------------------------------------------------------------------
    vidx = np.clip(((verts - lo) / fsp).astype(int), 0, np.array(fshape) - 1)
    T_HOT = 650.0       # flame threshold
    for s in range(burn_steps):
        state = integ.step_semi_implicit(state, p, dt, sources=(src if s == 0 else None))
        chi = state["char"] / (state["char"] + state["m_s"] + 1e-12)
        rec.vchi.append(chi[vidx[:, 0], vidx[:, 1], vidx[:, 2]].copy())
        rec.vT.append(state["T"][vidx[:, 0], vidx[:, 1], vidx[:, 2]].copy())
        rec.chi_grids.append(chi[wood].copy())
        rec.T_grids.append(state["T"][wood].copy())
        # flame: hot wood voxels (the visible fire), top-k by temperature
        hot = (state["T"] > T_HOT) & wood
        hi_idx = np.argwhere(hot)
        if len(hi_idx):
            Th = state["T"][hot]
            if len(hi_idx) > 160:
                keep = np.argsort(-Th)[:160]
                hi_idx, Th = hi_idx[keep], Th[keep]
            rec.flame.append((lo[None, :] + hi_idx * fsp, Th))
        else:
            rec.flame.append((np.zeros((0, 3)), np.zeros(0)))
        trust, eps, Tc = coarse_fields(state, chi)
        prox = proximity_field(cshape, Tc > 500.0, spacing=1.0, margin=1.5)
        ag.step(compute_D(trust=trust, eps=eps, proximity=prox, rp=ag.params))
        rec.levels.append(ag.level.copy())
    log(f"burn: {burn_steps} steps, char max {chi.max():.2f}, refined cells {(ag.level>0).sum()}, "
        f"flame voxels (last) {len(rec.flame[-1][0])}")

    # FRACTURE + TOPPLE ---------------------------------------------------------------------------
    model = xpbd.from_tree(tree)
    cidx = np.clip(((tree.pos[model.edges[:, 1]] - lo) / fsp).astype(int), 0, np.array(fshape) - 1)
    char_edge = state["char"][cidx[:, 0], cidx[:, 1], cidx[:, 2]] / (
        state["char"][cidx[:, 0], cidx[:, 1], cidx[:, 2]]
        + state["m_s"][cidx[:, 0], cidx[:, 1], cidx[:, 2]] + 1e-12)
    alpha, broken = xpbd.char_to_compliance(model, char_edge, S_break=s_break)
    import collections
    adj = collections.defaultdict(list)
    for e, (a, b) in enumerate(model.edges):
        if not broken[e]:
            adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    reach = np.zeros(model.M, bool); stk = [0]; reach[0] = True
    while stk:
        u = stk.pop()
        for w in adj[u]:
            if not reach[w]:
                reach[w] = True; stk.append(w)
    rec.released = ~reach
    st = xpbd.XPBDState.at_rest(model)
    rec.fall_pos.append(st.x.copy())
    for k in range(24):
        st = xpbd.step(model, st, 8e-3, iters=12, alpha=alpha, broken=broken, anchored=~rec.released)
        if k % 2 == 1:
            rec.fall_pos.append(st.x.copy())
    log(f"fracture: {int(broken.sum())} constraints; {int(rec.released.sum())} nodes detach")
    return rec


# ============================================================================================
# BEAUTY render (pyrender, EGL)
# ============================================================================================
def _look_at(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4); M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M


def _base_layer_rgb(rec):
    """Static per-vertex wood-layer base colour (bark / sapwood / heartwood), computed once."""
    rho, r_at, rb_at, rh_at, _ = segment_field(rec.verts, rec.tree)
    col = np.tile(me.COL_SAPWOOD, (len(rec.verts), 1)).astype(float)
    col[rho > rb_at] = me.COL_BARK
    col[rho < rh_at] = me.COL_HEARTWOOD
    return col


def _blend_rgba(base, vchi=None, vT=None):
    """Blend the static wood base with the simulation: char (chi) blackens, hot char (T) embers."""
    col = base.copy()
    if vchi is not None:
        chi = np.clip(vchi, 0, 1)[:, None]
        col = (1 - chi) * col + chi * me.COL_CHAR
        if vT is not None:
            ember = (np.clip((vT - 500) / 400, 0, 1)[:, None]) * chi
            col = (1 - ember) * col + ember * me.COL_EMBER
    rgba = np.empty((len(base), 4), np.uint8)
    rgba[:, :3] = np.clip(col, 0, 255).astype(np.uint8); rgba[:, 3] = 255
    return rgba


def render_beauty(rec, out, W=900, H=700, fps=12, orbit_frames=18, hold_frames=6):
    import subprocess
    import trimesh
    import pyrender

    center = np.array([rec.tree.pos[:, 0].mean(), rec.tree.pos[:, 1].mean(),
                       0.5 * (rec.tree.pos[:, 2].min() + rec.tree.pos[:, 2].max())])
    H_tree = float(rec.tree.pos[:, 2].max() - rec.tree.pos[:, 2].min())
    radius = 2.4 * max(H_tree, 1.0)
    ground = me.ground_mesh(rec.heightfield)
    r = pyrender.OffscreenRenderer(W, H)
    fsp = rec.fire_spacing
    unit_sphere = trimesh.creation.icosphere(subdivisions=1, radius=1.0)

    def flame_mesh(pts, T):
        """One combined mesh of ember spheres at the hot voxels, coloured red->yellow by T."""
        if len(pts) == 0:
            return None
        V, Fc = unit_sphere.vertices, unit_sphere.faces; nv = len(V)
        allv, allf, allc = [], [], []
        for i, (q, t) in enumerate(zip(pts, T)):
            frac = float(np.clip((t - 650) / 450, 0, 1))            # hotter -> bigger, yellower
            col = np.array([255, int(110 + 130 * frac), int(15 + 110 * frac), 255], np.uint8)
            allv.append(V * (0.45 + 0.6 * frac) * fsp + q)
            allf.append(Fc + i * nv)
            allc.append(np.tile(col, (nv, 1)))
        return trimesh.Trimesh(vertices=np.vstack(allv), faces=np.vstack(allf),
                               vertex_colors=np.vstack(allc), process=False)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "18",
         "-pix_fmt", "yuv420p", out],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = [0]

    def render(vdef, rgba, az, flame=None):
        sc = pyrender.Scene(bg_color=[0.06, 0.07, 0.09, 1.0], ambient_light=[0.22, 0.22, 0.26])
        tm = trimesh.Trimesh(vertices=vdef, faces=rec.faces, vertex_colors=rgba, process=False)
        sc.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
        gm = trimesh.Trimesh(vertices=ground.vertices, faces=ground.faces,
                             vertex_colors=ground.visual.vertex_colors, process=False)
        sc.add(pyrender.Mesh.from_trimesh(gm, smooth=False))
        if flame is not None and len(flame[0]):
            fm = flame_mesh(*flame)
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[1.0, 0.5, 0.15, 1.0], emissiveFactor=[1.0, 0.55, 0.12],
                metallicFactor=0.0, roughnessFactor=1.0)
            sc.add(pyrender.Mesh.from_trimesh(fm, material=mat, smooth=True))
            cen = flame[0].mean(0)
            sc.add(pyrender.PointLight(color=[1.0, 0.55, 0.2], intensity=8.0 * len(flame[0])),
                   pose=np.array([[1, 0, 0, cen[0]], [0, 1, 0, cen[1]], [0, 0, 1, cen[2] + 0.1],
                                  [0, 0, 0, 1]], float))
        cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45))
        pose = _look_at(center + np.array([radius*np.cos(az), radius*np.sin(az), 0.55*radius]), center)
        sc.add(cam, pose=pose)
        sc.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.2), pose=pose)
        sc.add(pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.6),
               pose=_look_at(center + np.array([-radius, radius*0.4, radius*0.6]), center))
        color, _ = r.render(sc)
        proc.stdin.write(np.ascontiguousarray(color[..., :3], np.uint8).tobytes())
        n[0] += 1

    base = _base_layer_rgb(rec)                                     # static wood layers (once)
    nb = len(rec.vchi)
    az0 = np.deg2rad(35)
    for k in range(orbit_frames):                                   # Act 1: the grown tree
        render(rec.verts, _blend_rgba(base), az0 + 2*np.pi * 0.25 * k / orbit_frames)
    for s in range(nb):                                             # Act 2: burn, char climbs + flame
        render(rec.verts, _blend_rgba(base, rec.vchi[s], rec.vT[s]),
               az0 + np.deg2rad(90) + 2*np.pi * 0.45 * s / nb, flame=rec.flame[s])
    az = az0 + np.deg2rad(90) + 2*np.pi*0.45                        # Act 3: topple (deform by XPBD)
    rgba = _blend_rgba(base, rec.vchi[-1], rec.vT[-1])
    last_flame = rec.flame[-1]
    for fi, fp in enumerate(rec.fall_pos):
        disp = fp - rec.tree.pos
        # flames fade out as the tree falls
        fl = (last_flame[0], last_flame[1]) if fi < len(rec.fall_pos) // 2 else None
        render(rec.verts + disp[rec.nearest_node], rgba, az, flame=fl)
        az += np.deg2rad(2)
    for _ in range(hold_frames):
        render(rec.verts + (rec.fall_pos[-1] - rec.tree.pos)[rec.nearest_node], rgba, az)
    r.delete()
    proc.stdin.close(); proc.wait()
    return out, n[0]


# ============================================================================================
# MECHANISM render (matplotlib 3-D panels)
# ============================================================================================
def render_mechanism(rec, out, fps=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    wp = rec.wood_pts
    fsp = rec.fire_spacing
    blk = rec.coarse_block
    lo = rec.fire_origin

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    nb = len(rec.vchi)
    nfall = len(rec.fall_pos)
    nframes = nb + nfall

    # static skeleton segments for ax3
    segs = [(int(rec.tree.parent[i]), i) for i in range(rec.tree.n) if rec.tree.parent[i] >= 0]
    segs = np.array(segs)

    def coarse_centers(level):
        cs = level.shape
        idx = np.array(np.meshgrid(*[np.arange(s) for s in cs], indexing="ij")).reshape(3, -1).T
        return lo[None, :] + (idx * blk + 0.5 * blk) * fsp, level.ravel()

    def draw(frame):
        for ax in (ax1, ax2, ax3):
            ax.cla(); ax.set_axis_off()
        s = min(frame, nb - 1)
        # ax1 temperature
        T = rec.T_grids[s]
        ax1.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c=T, cmap="turbo", vmin=300, vmax=1100, s=6)
        ax1.set_title(f"temperature  (step {s+1}/{nb})", fontsize=11)
        # ax2 refinement levels
        cc, lev = coarse_centers(rec.levels[s])
        m = lev > 0
        if m.any():
            ax2.scatter(cc[m, 0], cc[m, 1], cc[m, 2], c=lev[m], cmap="plasma", vmin=0, vmax=3,
                        s=60, marker="s", alpha=0.8)
        ax2.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c="0.6", s=2, alpha=0.25)
        ax2.set_title("adaptive refinement (trust-driven)", fontsize=11)
        # ax3 structure / fracture
        if frame < nb:
            P = rec.tree.pos; chi = rec.vchi[s]
            # node char by nearest vertex char (approx via skeleton sample): use released later
            ax3.scatter(P[:, 0], P[:, 1], P[:, 2], c="saddlebrown", s=4)
            ax3.set_title("structure (intact)", fontsize=11)
        else:
            P = rec.fall_pos[frame - nb]
            col = np.where(rec.released, "0.1", "saddlebrown")
            for a, b in segs:
                ax3.plot([P[a, 0], P[b, 0]], [P[a, 1], P[b, 1]], [P[a, 2], P[b, 2]],
                         c=("0.1" if (rec.released[a] or rec.released[b]) else "saddlebrown"), lw=0.6)
            ax3.scatter(P[:, 0], P[:, 1], P[:, 2], c=col, s=4)
            ax3.set_title("fracture: charred base burns through -> topple", fontsize=11)
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(wp[:, 0].min(), wp[:, 0].max())
            ax.set_ylim(wp[:, 1].min(), wp[:, 1].max())
            ax.set_zlim(min(wp[:, 2].min(), rec.tree.pos[:, 2].min()),
                        rec.tree.pos[:, 2].max())
            ax.view_init(elev=18, azim=-60 + 0.6 * frame)
        fig.suptitle("Nebula Phase-0 — the tree, completely", fontsize=14)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=nframes, blit=False)
    anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    return out, nframes


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="nebula-demo")
    ap.add_argument("--out", default="demo_output/nebula", help="output path prefix")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fire-grid", type=int, default=28)
    ap.add_argument("--burn-steps", type=int, default=48)
    ap.add_argument("--no-beauty", action="store_true")
    ap.add_argument("--no-mechanism", action="store_true")
    a = ap.parse_args(argv)
    outdir = os.path.dirname(a.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    print("=== recording the Phase-0 slice ===")
    rec = record_demo(seed=a.seed, fire_grid=a.fire_grid, burn_steps=a.burn_steps)

    outs = []
    if not a.no_beauty:
        print("=== rendering beauty (pyrender/EGL) ===")
        path, nf = render_beauty(rec, a.out + "_beauty.mp4")
        print(f"  wrote {path} ({nf} frames)"); outs.append(path)
    if not a.no_mechanism:
        print("=== rendering mechanism (matplotlib) ===")
        path, nf = render_mechanism(rec, a.out + "_mechanism.mp4")
        print(f"  wrote {path} ({nf} frames)"); outs.append(path)
    print("\ndemo videos:", *outs, sep="\n  ")


if __name__ == "__main__":
    main()
