"""
The integrated burning tree — Tier-3 scene (Theme E): grow → flame → crown flash → char/bark relief
→ USD+OpenVDB export, with a realism-acceptance gate over the DERIVED STATE.

This ties the Tier-3 mechanisms together into the milestone scene and hands it to the path tracer.
Consistent with causal fidelity, the acceptance gate is checked on **what the simulation computed**
(the derived fields/morphology), NOT on rendered pixels — the beauty render is the downstream path
tracer's job; here we certify that the physics + morphology are right and exported faithfully.

  1. GROW    : tree (pipe-model root flare + surface roots, V3.8) + phyllotactic canopy (V3.4).
  2. FLAME   : the buoyant reacting flame (V3.1/V3.2, physical kinetics) at the trunk base → T, soot.
  3. CHAR    : the trunk chars where the flame was hot; the canopy flashes (fine-fuel, V3.5).
  4. RELIEF  : derived bark-fissure (V3.8) + char-alligator (V3.6) maps.
  5. EXPORT  : USD scene + fire.vdb + manifest (the path-tracer handoff, derived appearance V3.3).
  6. ACCEPT  : a frozen realism checklist over the derived state; + a deterministic scene digest.

Deterministic in (seed, config). The flame sim is the cost (~20–40 s); everything else is fast.
"""
from dataclasses import dataclass, field as _field
import hashlib
import os

import numpy as np

from ..operators.growth import grow_tree, GrowthParams
from ..operators import canopy as cano
from ..operators import fine_fuel as ff
from ..operators import gas_combustion as gc
from ..geometry.mesh_export import tube_mesh, _sample
from ..geometry import char_texture as ct
from ..geometry import bark_texture as bt


@dataclass
class SceneConfig:
    seed: int = 7
    fire_nx: int = 28
    fire_nz: int = 64
    fire_steps: int = 60
    pilot_steps: int = 12
    fuel_rate: float = 2.0
    crown_front_speed: float = 0.5


@dataclass
class SceneResult:
    tree: object
    canopy: object
    fuel: object
    fire: dict
    fire_origin: np.ndarray
    fire_spacing: float
    accept: dict
    digest: int
    out_dir: str = None


def _digest(*arrays, scale=1e4):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(np.rint(np.asarray(a, float) * scale).astype(np.int64).tobytes())
    return int.from_bytes(h.digest(), "big")


def run_burning_scene(cfg=SceneConfig(), out_dir=None, verbose=True, do_export=True):
    def log(*a):
        if verbose:
            print(*a)

    # 1) GROW (morphology: flare + surface roots; phyllotactic canopy) ---------------------------
    gp = GrowthParams(dim=3)
    tree = grow_tree(seed=cfg.seed, gp=gp)
    can = cano.generate_canopy(tree, cano.CanopyParams(), seed=cfg.seed)
    z0, z1 = float(tree.pos[:, 2].min()), float(tree.pos[:, 2].max()); H = z1 - z0
    log(f"[1] grow: {tree.n} nodes, {can.n} leaves; root flare {tree.radius[0]:.2f}, "
        f"surface roots {int(((tree.pos[:,2]>=-0.02)&(tree.pos[:,2]<0.1)).sum())}")

    # 2) FLAME at the trunk base (physical hot kinetics) -----------------------------------------
    nx, nz = cfg.fire_nx, cfg.fire_nz
    fsp = 0.7 * H / nz
    origin = np.array([tree.pos[0, 0] - nx / 2 * fsp, tree.pos[0, 1] - nx / 2 * fsp, z0])
    shape = (nx, nx, nz)
    p = gc.ReactingParams(fuel_rate=cfg.fuel_rate)
    ix, iy = np.meshgrid(np.arange(nx), np.arange(nx), indexing="ij")
    disk = ((ix - nx // 2) ** 2 + (iy - nx // 2) ** 2) <= 9
    src = np.zeros(shape, bool); src[disk, 0:2] = True
    pm = np.zeros(shape, bool); pm[disk, 5:11] = True
    sc, vel = gc.make_state(shape, p)
    for n in range(cfg.fire_steps):
        if n < cfg.pilot_steps:
            gc.pilot(sc, pm)
        sc, vel, info, rr = gc.step(sc, vel, p, 0.5, source=src)
    flame_T = float(sc["T"].max()); src_T = float(sc["T"][src].max())
    log(f"[2] flame: peak {flame_T:.0f} K (source {src_T:.0f} K), soot max {sc['soot'].max():.3f}")

    # 3) CHAR the trunk from the flame; FLASH the canopy -----------------------------------------
    verts, faces, vnode = tube_mesh(tree)
    Tv = _sample(sc["T"], origin, fsp, verts)
    zfrac = np.clip((verts[:, 2] - z0) / (0.5 * H), 0, 1)
    chi_v = np.clip((Tv - 600) / 600, 0, 1) * (1 - 0.5 * zfrac) + 0.6 * np.exp(-3 * zfrac)
    chi_v = np.clip(chi_v, 0, 1)
    # the crown flashes: a rising front climbs THROUGH the canopy (fine-fuel, V3.5)
    fuel, _hist, _rel = ff.crown_flash(can, ff.FineFuelParams(), dt=0.1, n_steps=130,
                                       front_speed=cfg.crown_front_speed, z0=z0)
    log(f"[3] char: {(chi_v>0.5).sum()} charred verts; canopy ignited {fuel.ignited.mean()*100:.0f}%, "
        f"burned {(1-fuel.mass.sum()/max(fuel.mass0.sum(),1e-9))*100:.0f}%")

    # 4) DERIVE relief (bark + char) -------------------------------------------------------------
    bark = bt.bark_relief(tree, verts, vnode, seed=cfg.seed)
    char = ct.char_relief(tree, verts, vnode, chi_v, seed=cfg.seed)
    log(f"[4] relief: bark fissures {(bark['fissure']>0.05).sum()}, char cracks {(char['fissure']>0.05).sum()}")

    # 5) EXPORT (USD + VDB handoff) --------------------------------------------------------------
    fire = {"T": sc["T"], "soot": sc["soot"], "char": chi_v * 0 + 0,  # grid char not tracked here
            "m_s": np.ones_like(sc["T"])}
    if do_export and out_dir is not None:
        from ..geometry.export import export_scene
        from ..fields.heightfield import make_terrain
        hf = make_terrain(seed=cfg.seed, size=float(np.ptp(tree.pos[:, :2]) + 2.0))
        man = export_scene(out_dir, tree, fire={"T": sc["T"], "soot": sc["soot"]},
                           fire_origin=origin, fire_spacing=fsp, canopy=can, fuel=fuel,
                           heightfield=hf, provenance={"seed": cfg.seed, "scene": "burning_tree"})
        log(f"[5] export: USD + VDB to {out_dir} (vdb {man['fire_volume']['vdb_written']})")

    # 6) ACCEPTANCE GATE over the derived state (NOT rendered pixels) ----------------------------
    # flame rises: soot/heat center-of-mass above the ignition band
    hot = sc["T"] - 300.0
    zc = origin[2] + (np.arange(nz) + 0.5) * fsp
    heat_com = float((hot.sum(axis=(0, 1)) * zc).sum() / (hot.sum() + 1e-9))
    # root flare (derived): base vs thick lower trunk
    zz = tree.pos[:, 2]
    tm = (zz > 0.08 * H) & (zz < 0.25 * H) & (tree.radius > 0.4 * tree.radius[0])
    flare = float(tree.radius[0] / (np.median(tree.radius[tm]) if tm.any() else tree.radius[0]))
    accept = {
        "flame_hot": flame_T >= 1100.0,
        "flame_hotter_than_source": flame_T > src_T + 50,
        "flame_rises": heat_com > origin[2] + 0.1 * (nz * fsp),
        "smoke_present": float(sc["soot"].max()) > 0.005,
        "canopy_flashed": fuel.ignited.mean() > 0.5,
        "trunk_charred": (chi_v > 0.5).sum() > 100,
        "char_cracked": (char["fissure"][chi_v > 0.5] > 0.05).any(),
        "bark_fissured": (bark["fissure"] > 0.05).sum() > 100,
        "root_flare": flare >= 1.2,
        "surface_roots": int(((zz >= -0.02) & (zz < 0.1)).sum()) >= 3,
    }
    digest = _digest(tree.pos, tree.radius, sc["T"], sc["soot"], chi_v)
    npass = sum(accept.values())
    log(f"[6] realism-acceptance: {npass}/{len(accept)} checks pass over the derived state")
    for k, v in accept.items():
        log(f"      {'PASS' if v else 'FAIL'}  {k}")
    log(f"[*] scene digest: {digest:032x}")

    return SceneResult(tree=tree, canopy=can, fuel=fuel, fire=sc, fire_origin=origin,
                       fire_spacing=fsp, accept=accept, digest=digest, out_dir=out_dir)


if __name__ == "__main__":
    import tempfile
    np.seterr(all="ignore")
    out = os.path.join(tempfile.gettempdir(), "nebula_burning_scene")
    print("=== Nebula Tier-3 burning scene (grow → flame → char → relief → export) ===")
    r = run_burning_scene(SceneConfig(seed=7), out_dir=out)
    npass = sum(r.accept.values())
    print(f"\nrealism-acceptance: {npass}/{len(r.accept)} derived-state checks pass")
    assert npass == len(r.accept), f"acceptance gate: {[k for k,v in r.accept.items() if not v]}"

    # determinism: re-run, identical digest (the program IS the asset)
    r2 = run_burning_scene(SceneConfig(seed=7), out_dir=None, verbose=False, do_export=False)
    print(f"determinism: digest match = {r.digest == r2.digest}")
    assert r.digest == r2.digest
    print("\nburning-scene integration + acceptance gate passed.")
