"""
Physics-side demo of the burning tree — the FAITHFUL PREVIEW (Tier-3 mechanisms), to MP4 + PNG.

This is Nebula's own view of what the simulation computed (the deliverable beauty render is the
downstream Omniverse path trace of the USD+VDB export). Three acts, all from the verified Tier-3
state via the Gaussian-splat rasterizer (V3.9):
  - GROW    : the grown tree (root flare + surface roots + phyllotactic canopy) — morphology.
  - BURN    : the physical flame (hot, V3.2) climbing the trunk, char + crown flash, smoke.
  - PHYSICS : "show-the-physics" — the same splats recoloured by the raw temperature field (the
              honest debugger), proving the colour is derived, not painted.

Run:  python -m nebula.pipeline.physics_demo --out demo_output
"""
import os
import subprocess

import numpy as np

from .burning_scene import run_burning_scene, SceneConfig
from ..render import splat as sp
from ..render import gaussian_rasterizer as gr
from ..geometry.mesh_export import _sample


def _ffmpeg(frames_dir, out_mp4, fps=14):
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "f%04d.png"), "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", "-crf", "18", out_mp4],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _orbit_cam(center, radius, az, W, H, elev=0.12):
    return gr.Camera(eye=(center[0] + radius * np.cos(az), center[1] + radius * np.sin(az),
                          center[2] + elev * radius), target=tuple(center), W=W, H=H, fov_deg=42)


def _render_orbit(cloud, center, radius, out_mp4, frames_dir, n=48, W=640, H=800,
                  bg=(0.05, 0.06, 0.09), exposure=1.4, bloom=0.0, az0=0.6, hero_png=None):
    import imageio.v2 as imageio
    os.makedirs(frames_dir, exist_ok=True)
    for k in range(n):
        az = az0 + 2 * np.pi * k / n
        cam = _orbit_cam(center, radius, az, W, H)
        hdr, _ = gr.render(cloud["means"], cloud["cov"], cloud["color"], cloud["opacity"], cam, bg=bg)
        img = (np.clip(gr.tonemap(hdr, exposure=exposure, bloom=bloom), 0, 1) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(frames_dir, f"f{k:04d}.png"), img)
        if hero_png is not None and k == n // 6:
            imageio.imwrite(hero_png, img)
    _ffmpeg(frames_dir, out_mp4)


def make_demos(out_dir="demo_output", seed=7, W=640, H=800, n_orbit=48):
    os.makedirs(out_dir, exist_ok=True)
    fr = os.path.join(out_dir, "_frames")
    print("=== running the Tier-3 burning scene (physics) ===")
    r = run_burning_scene(SceneConfig(seed=seed), out_dir=os.path.join(out_dir, "omniverse_scene"),
                          verbose=False)
    tree, can, fuel, sc = r.tree, r.canopy, r.fuel, r.fire
    z0, z1 = tree.pos[:, 2].min(), tree.pos[:, 2].max(); H_t = z1 - z0
    center = np.array([tree.pos[:, 0].mean(), tree.pos[:, 1].mean(), 0.5 * (z0 + z1)])
    radius = 2.2 * max(H_t, 1.0)

    # ---- Act 1: the grown tree (morphology) ----
    tsp = sp.tree_splats(tree); csp = sp.canopy_splats(can)
    grown = sp.merge(tsp, csp)
    print(f"[demo] Act 1 grown tree: {len(grown['means'])} splats")
    _render_orbit(grown, center, radius, os.path.join(out_dir, "physics_1_grown.mp4"),
                  os.path.join(fr, "grown"), n=n_orbit, W=W, H=H,
                  hero_png=os.path.join(out_dir, "physics_grown.png"))

    # ---- Act 2: the burning tree (hot flame + char + crown flash + smoke) ----
    verts = tsp["means"]; Tv = _sample(sc["T"], r.fire_origin, r.fire_spacing, verts)
    zfrac = np.clip((verts[:, 2] - z0) / (0.5 * H_t), 0, 1)
    chi_v = np.clip((Tv - 600) / 600, 0, 1) * (1 - 0.5 * zfrac) + 0.6 * np.exp(-3 * zfrac)
    tsp_burn = sp.tree_splats(tree, chi_grid=(sc["char"] if "char" in sc else None))
    # recolour the tree splats by char + ember (derived appearance)
    from ..geometry import appearance as ap
    surf = ap.surface_appearance(tsp["color"], T=Tv, chi=np.clip(chi_v, 0, 1))
    tsp_burn = dict(tsp); tsp_burn["color"] = surf["albedo"] + surf["emission"]
    csp_burn = sp.canopy_splats(can, fuel=fuel)
    flame = sp.flame_splats(sc, r.fire_origin, r.fire_spacing, T_hot=750.0)
    burning = sp.merge(tsp_burn, csp_burn, flame)
    print(f"[demo] Act 2 burning: {len(burning['means'])} splats (flame {0 if flame is None else len(flame['means'])})")
    _render_orbit(burning, center, radius, os.path.join(out_dir, "physics_2_burning.mp4"),
                  os.path.join(fr, "burning"), n=n_orbit, W=W, H=H, bg=(0.04, 0.04, 0.06),
                  exposure=1.5, bloom=0.6, hero_png=os.path.join(out_dir, "physics_burning.png"))

    # ---- Act 3: show-the-physics (recolour by the raw temperature field) ----
    Tcanopy = np.full(can.n, 300.0)
    phys_tree = sp.show_physics(tsp, Tv, cmap="inferno", vmin=300, vmax=1400)
    phys = sp.merge(phys_tree, sp.show_physics(csp, Tcanopy, cmap="inferno", vmin=300, vmax=1400), flame)
    print(f"[demo] Act 3 show-the-physics (temperature field)")
    _render_orbit(phys, center, radius, os.path.join(out_dir, "physics_3_temperature.mp4"),
                  os.path.join(fr, "phys"), n=n_orbit, W=W, H=H, bg=(0.0, 0.0, 0.0),
                  hero_png=os.path.join(out_dir, "physics_temperature.png"))

    print("\nphysics demos written to", out_dir,
          "(physics_1_grown.mp4, physics_2_burning.mp4, physics_3_temperature.mp4 + hero PNGs)")
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    make_demos(a.out, a.seed)
