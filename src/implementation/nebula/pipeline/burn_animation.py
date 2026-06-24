"""
The burning-tree ANIMATION (physics side) — the fire SPREADING through the tree over time.

Not a camera orbit of one frozen moment: the camera is fixed and the FIRE evolves — igniting at the
base, climbing the trunk, the crown flashing, char propagating, smoke rising, leaves burning away.
Driven by `operators.tree_fire` (the front-propagation spread + the continuous flame volume) and
rendered with the Gaussian-splat rasterizer (the faithful preview). The flame is the continuous Tg
VOLUME (sampled densely), not spheres.

Run:  python -m nebula.pipeline.burn_animation --out demo_output
"""
import os
import subprocess

import numpy as np

from ..operators.growth import grow_tree, GrowthParams
from ..operators import canopy as cano
from ..operators import tree_fire as tfire
from ..geometry import mesh_export as me
from ..geometry import appearance as ap
from ..render import splat as sp
from ..render import gaussian_rasterizer as gr


def _ffmpeg(frames_dir, out_mp4, fps=18):
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", out_mp4], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def volume_flame_splats(Tg, soot, origin, spacing, T_hot=560.0, max_pts=26000):
    """Continuous flame (emissive blackbody) + smoke splats sampled from the Tg / soot VOLUME."""
    out = []
    hot = Tg > T_hot
    if hot.any():
        ij = np.argwhere(hot); Tv = Tg[hot]
        if len(ij) > max_pts:
            keep = np.argsort(-Tv)[:max_pts]; ij, Tv = ij[keep], Tv[keep]
        pos = origin[None, :] + (ij + 0.5) * spacing
        col = ap.ember_emission(Tv, chi=np.ones(len(Tv))) * 3.0
        s = np.full(len(pos), 0.95 * spacing)
        cov = (s[:, None, None] ** 2) * np.eye(3)[None]
        op = np.clip((Tv - 400) / 700, 0.18, 0.7)
        out.append({"means": pos, "cov": cov, "color": col, "opacity": op})
    smk = soot > 0.01
    if smk.any():
        ij = np.argwhere(smk); sv = soot[smk]
        if len(ij) > max_pts:
            keep = np.argsort(-sv)[:max_pts]; ij, sv = ij[keep], sv[keep]
        pos = origin[None, :] + (ij + 0.5) * spacing
        col = np.tile(np.array([0.06, 0.06, 0.07]), (len(pos), 1))
        s = np.full(len(pos), 1.3 * spacing)
        cov = (s[:, None, None] ** 2) * np.eye(3)[None]
        out.append({"means": pos, "cov": cov, "color": col,
                    "opacity": ap.smoke_alpha(sv, kappa=2.5) * 0.4})
    return sp.merge(*out) if out else None


def make_animation(out_dir="demo_output", seed=7, n_frames=64, W=720, H=900, fps=18):
    os.makedirs(out_dir, exist_ok=True)
    fr = os.path.join(out_dir, "_burn_frames"); os.makedirs(fr, exist_ok=True)
    import imageio.v2 as imageio

    tree = grow_tree(seed=seed, gp=GrowthParams(dim=3))
    can = cano.generate_canopy(tree, cano.CanopyParams(), seed=seed)
    print("=== simulating the spreading fire (front propagation) ===")
    tf, times, hist = tfire.simulate(tree, can, n_frames=n_frames)
    print(f"[anim] {tf.n} fuel elements; burn over {tf.t_end:.0f}s in {n_frames} frames")

    verts, faces, vnode = me.tube_mesh(tree)
    tsp = sp.tree_splats(tree)                       # base wood colour
    base_col = tsp["color"].copy()
    csp0 = sp.canopy_splats(can)                     # base leaf cloud (geometry + base green)
    leaf_base = csp0["color"].copy()
    nbf = tf.n_branch

    # fixed camera (the fire animates in place); a hair of drift for life
    z0, z1 = tree.pos[:, 2].min(), tree.pos[:, 2].max(); H_t = z1 - z0
    center = np.array([tree.pos[:, 0].mean(), tree.pos[:, 1].mean(), 0.52 * z1])
    radius = 2.05 * max(H_t, 1.0)

    for k, fstate in enumerate(hist):
        # tree char + ember per vertex (from the per-branch-element state)
        cn = np.zeros(tree.n); cn[tf.branch_node] = fstate["char"][:nbf]
        bn = np.zeros(tree.n, bool); bn[tf.branch_node] = fstate["burning"][:nbf]
        cv = cn[vnode]; bv = bn[vnode]
        TgV = me._sample(fstate["Tg"], tf.origin, tf.spacing, verts)
        emberT = np.where(bv, np.clip(TgV, 850, 1350), 0.0)
        surf = ap.surface_appearance(base_col, T=emberT, chi=cv)
        tree_cloud = dict(tsp); tree_cloud["color"] = surf["albedo"] + surf["emission"]

        # leaves: green -> scorched brown -> char black -> gone (opacity)
        lchar = fstate["char"][nbf:]; lburn = fstate["burning"][nbf:]
        lc = leaf_base.copy()
        c = np.clip(lchar, 0, 1)[:, None]
        brown = np.array([0.30, 0.16, 0.04])
        lc = (1 - np.clip(c * 1.6, 0, 1)) * lc + np.clip(c * 1.6, 0, 1) * brown
        lc = (1 - np.clip((c - 0.5) * 2, 0, 1)) * lc + np.clip((c - 0.5) * 2, 0, 1) * np.array([0.05, 0.045, 0.04])
        lc[lburn] = lc[lburn] + ap.ember_emission(np.full(int(lburn.sum()), 1150.0), chi=np.ones(int(lburn.sum())))
        lop = csp0["opacity"] * (lchar < 0.92)        # fully-charred leaves drop
        leaf_cloud = {"means": csp0["means"], "cov": csp0["cov"], "color": lc, "opacity": lop}

        flame = volume_flame_splats(fstate["Tg"], fstate["soot"], tf.origin, tf.spacing)
        cloud = sp.merge(tree_cloud, leaf_cloud, flame)

        az = np.deg2rad(35) + np.deg2rad(18) * k / max(n_frames - 1, 1)   # slow drift
        cam = gr.Camera(eye=(center[0] + radius * np.cos(az), center[1] + radius * np.sin(az),
                             center[2] + 0.12 * radius), target=tuple(center), W=W, H=H, fov_deg=42)
        hdr, _ = gr.render(cloud["means"], cloud["cov"], cloud["color"], cloud["opacity"], cam,
                           bg=(0.03, 0.035, 0.05))
        img = (np.clip(gr.tonemap(hdr, exposure=1.5, bloom=0.6), 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(fr, f"f{k:04d}.png"), img)
        if k in (n_frames // 3, n_frames // 2):
            imageio.imwrite(os.path.join(out_dir, f"burn_hero_{k}.png"), img)
        if k % 8 == 0:
            print(f"[anim] frame {k+1}/{n_frames}  charred {fstate['char'].mean()*100:.0f}%  "
                  f"flame Tmax {fstate['Tg'].max():.0f}K")

    out_mp4 = os.path.join(out_dir, "physics_burn_animation.mp4")
    _ffmpeg(fr, out_mp4, fps=fps)
    print("wrote", out_mp4)
    return out_mp4


if __name__ == "__main__":
    import argparse
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default="demo_output"); ap_.add_argument("--seed", type=int, default=7)
    ap_.add_argument("--frames", type=int, default=64)
    a = ap_.parse_args()
    make_animation(a.out, a.seed, a.frames)
