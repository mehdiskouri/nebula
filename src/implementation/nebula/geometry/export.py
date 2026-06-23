"""
Derived-state export — Nebula's handoff to the path tracer (ARCHITECTURE Part I; Decision #2).

Nebula's job is to LAND the physics + morphology and hand the DERIVED state to a path tracer
(Omniverse). The beauty render is downstream; Nebula must NOT beautify — it exports exactly what
the simulation computed, with a manifest stating how to render it physically. This writes a USD
scene (Omniverse-native) carrying:

  - the grown tree mesh (`tube_mesh` — the morphology), with **derived** per-vertex material
    primvars: albedo, roughness, emissiveColor (from `geometry.appearance`, i.e. wood layer + char
    + blackbody ember — all simulation outputs, not painted);
  - the canopy as a point cloud with derived leaf colors;
  - the FIRE as a `UsdVolVolume` referencing an **OpenVDB** `.vdb` (named grids `temperature` [K]
    and `density` [soot]) so the path tracer renders blackbody emission from T + Beer–Lambert
    absorption from soot — the physically-correct flame, not splat pixels;
  - a `manifest.json` "causal contract": every field, its units, the physical render recipe, and
    provenance (seed / params / scene digest) so the asset is reproducible.

USD via `usd-core` (this venv). The `.vdb` is written by `_vdb_convert.py` under the `vdb` conda
env (OpenVDB has no py3.13 wheel) via a subprocess; the dense grids are also kept as `.npz`.
"""
import json
import os
import subprocess

import numpy as np

from .mesh_export import tube_mesh, segment_field, COL_BARK, COL_SAPWOOD, COL_HEARTWOOD, ground_mesh
from . import appearance as ap

VDB_PY = "/venv/vdb/bin/python"           # conda env with pyopenvdb (see _vdb_convert.py)


def _derived_surface(tree, verts, vert_node, chi=None, Tv=None):
    """Per-vertex DERIVED material: albedo / roughness / emission from layer + char + ember."""
    rho, _, rb_at, rh_at, _ = segment_field(verts, tree)
    base = np.tile(COL_SAPWOOD / 255.0, (len(verts), 1))
    base[rho > rb_at] = COL_BARK / 255.0
    base[rho < rh_at] = COL_HEARTWOOD / 255.0
    s = ap.surface_appearance(base, T=Tv, chi=chi)
    return s["albedo"].astype(np.float32), np.broadcast_to(s["roughness"], (len(verts),)).astype(np.float32), \
        s["emission"].astype(np.float32)


def export_scene(out_dir, tree, fire=None, fire_origin=None, fire_spacing=None,
                 canopy=None, fuel=None, heightfield=None, provenance=None, write_vdb=True):
    """Write the USD scene + (optional) fire VDB + manifest into `out_dir`. Returns the manifest dict."""
    from pxr import Usd, UsdGeom, UsdShade, UsdVol, Sdf, Gf, Vt
    os.makedirs(out_dir, exist_ok=True)
    usd_path = os.path.join(out_dir, "scene.usda")
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # --- tree mesh + derived material primvars ---
    verts, faces, vnode = tube_mesh(tree)
    chi = Tv = None
    if fire is not None and fire_origin is not None:
        from .mesh_export import _sample
        chi = np.clip(_sample(fire["char"] / (fire["char"] + fire["m_s"] + 1e-12)
                              if "m_s" in fire else fire.get("char", np.zeros(1)),
                              fire_origin, fire_spacing, verts), 0, 1) if "char" in fire else None
        Tv = _sample(fire["T"], fire_origin, fire_spacing, verts) if "T" in fire else None
    albedo, rough, emis = _derived_surface(tree, verts, vnode, chi, Tv)

    # DERIVED bark-fissure relief (V3.8): displacement + perturbed normal + AO-darkened albedo —
    # a texture that is a simulation output (radial-growth tension), exported as primvars/displaced
    # geometry the path tracer reads. Not an authored bark image.
    from .bark_texture import bark_relief, apply_fissure_albedo
    rel = bark_relief(tree, verts, vnode, seed=int(getattr(tree, "seed", 0)))
    verts = rel["verts_relief"]                       # fissure-displaced surface
    albedo = apply_fissure_albedo(albedo, rel["fissure"])

    mesh = UsdGeom.Mesh.Define(stage, "/World/Tree")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.astype(np.int32).flatten().tolist()))
    pv = UsdGeom.PrimvarsAPI(mesh)
    pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                     UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(albedo))
    pv.CreatePrimvar("emissiveColor", Sdf.ValueTypeNames.Color3fArray,
                     UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(emis))
    pv.CreatePrimvar("roughness", Sdf.ValueTypeNames.FloatArray,
                     UsdGeom.Tokens.vertex).Set(Vt.FloatArray(rough.tolist()))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(rel["normal"].astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    pv.CreatePrimvar("barkFissure", Sdf.ValueTypeNames.FloatArray,
                     UsdGeom.Tokens.vertex).Set(Vt.FloatArray(rel["displacement"].astype(float).tolist()))

    # a UsdPreviewSurface material (vertex displayColor drives diffuse in Omniverse)
    mat = UsdShade.Material.Define(stage, "/World/Materials/Wood")
    sh = UsdShade.Shader.Define(stage, "/World/Materials/Wood/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(mat)

    # --- canopy as a point cloud with derived leaf colors ---
    n_leaves = 0
    if canopy is not None and canopy.n > 0:
        from ..render.splat import canopy_splats
        cs = canopy_splats(canopy, fuel=fuel)
        n_leaves = len(cs["means"])
        pts = UsdGeom.Points.Define(stage, "/World/Canopy")
        pts.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(cs["means"].astype(np.float32)))
        w = np.sqrt(np.clip(canopy.area[:n_leaves] if len(canopy.area) >= n_leaves else 0.01, 1e-5, None))
        pts.CreateWidthsAttr(Vt.FloatArray((0.06 * np.ones(n_leaves)).tolist()))
        UsdGeom.PrimvarsAPI(pts).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
            UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.clip(cs["color"], 0, 1).astype(np.float32)))

    # --- ground ---
    if heightfield is not None:
        gm = ground_mesh(heightfield)
        g = UsdGeom.Mesh.Define(stage, "/World/Ground")
        g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(gm.vertices.astype(np.float32)))
        g.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(gm.faces)))
        g.CreateFaceVertexIndicesAttr(Vt.IntArray(gm.faces.astype(np.int32).flatten().tolist()))

    # --- fire volume: dense grids -> .npz -> .vdb -> UsdVolVolume ---
    vdb_written = False
    if fire is not None and fire_origin is not None and "T" in fire:
        soot = fire.get("soot", np.zeros_like(fire["T"]))
        npz = os.path.join(out_dir, "fire_fields.npz")
        np.savez(npz, temperature=np.asarray(fire["T"], np.float32),
                 density=np.asarray(soot, np.float32), voxel_size=float(fire_spacing),
                 origin=np.asarray(fire_origin, float))
        vdb_path = os.path.join(out_dir, "fire.vdb")
        if write_vdb and os.path.exists(VDB_PY):
            conv = os.path.join(os.path.dirname(__file__), "..", "render", "_vdb_convert.py")
            r = subprocess.run([VDB_PY, os.path.abspath(conv), npz, vdb_path],
                               capture_output=True, text=True)
            vdb_written = os.path.exists(vdb_path)
            if not vdb_written:
                print("VDB conversion failed:", r.stderr[-300:])
        vol = UsdVol.Volume.Define(stage, "/World/Fire")
        for fld in ("density", "temperature"):
            asset = UsdVol.OpenVDBAsset.Define(stage, f"/World/Fire/{fld}")
            asset.CreateFilePathAttr("./fire.vdb")
            asset.CreateFieldNameAttr(fld)
            vol.CreateFieldRelationship(fld, asset.GetPath())

    stage.GetRootLayer().Save()

    # --- manifest: the causal contract ---
    manifest = {
        "generator": "nebula", "format": "usd+openvdb",
        "scene": os.path.basename(usd_path),
        "morphology": {"tree_vertices": int(len(verts)), "canopy_leaves": int(n_leaves)},
        "derived_material": {
            "note": "per-vertex primvars are SIMULATION OUTPUTS, not authored textures",
            "displayColor": "albedo: wood layer (bark/sapwood/heartwood) + char darkening (chi) + bark-fissure AO",
            "emissiveColor": "blackbody ember emission ∝ T^4 (Planckian locus) where charring",
            "roughness": "derived from char/moisture state",
            "normals + barkFissure": "DERIVED bark-fissure relief (radial-growth tension; depth ∝ radius/growth) — geometry is fissure-displaced, normals perturbed; not an authored texture"},
        "fire_volume": ({
            "asset": "fire.vdb", "fields": {"temperature": "Kelvin", "density": "soot fraction"},
            "render_recipe": "blackbody emission from temperature (Planck) + Beer-Lambert absorption from density",
            "dense_source": "fire_fields.npz", "vdb_written": bool(vdb_written)}
            if fire is not None and fire_origin is not None and "T" in fire else None),
        "provenance": provenance or {},
        "discipline": "Nebula exports what it computed; the path tracer renders it faithfully (no beautification)."}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    import tempfile
    from ..operators.growth import grow_tree, GrowthParams
    from ..operators import canopy as cano
    from ..fields.heightfield import make_terrain
    np.seterr(all="ignore")

    tree = grow_tree(seed=7, gp=GrowthParams(dim=3))
    can = cano.generate_canopy(tree, cano.CanopyParams(), seed=7)
    hf = make_terrain(seed=7, size=float(np.ptp(tree.pos[:, :2]) + 2.0))
    # a small synthetic hot fire field at the base (exercise the VDB + emission export)
    lo = tree.pos.min(0) - 0.2
    fsp = 0.08
    shp = tuple(int(np.ceil((np.ptp(tree.pos[:, d]) + 0.4) / fsp)) + 1 for d in range(3))
    T = np.full(shp, 300.0); soot = np.zeros(shp)
    T[:, :, : shp[2] // 4] = 1300.0; soot[:, :, : shp[2] // 3] = 0.4
    fire = {"T": T, "soot": soot, "char": np.zeros(shp), "m_s": np.ones(shp)}

    out = os.path.join(tempfile.gettempdir(), "nebula_export")
    man = export_scene(out, tree, fire=fire, fire_origin=lo, fire_spacing=fsp, canopy=can,
                       heightfield=hf, provenance={"seed": 7, "scene_digest": "demo"})
    print("1) wrote USD scene + manifest to", out)
    print("   morphology:", man["morphology"])
    print("   fire_volume:", {k: man["fire_volume"][k] for k in ("asset", "vdb_written")})
    assert os.path.exists(os.path.join(out, "scene.usda"))
    assert man["fire_volume"]["vdb_written"], "VDB not written"
    assert os.path.exists(os.path.join(out, "fire.vdb"))

    # reload the USD and check the derived primvars + volume are present
    from pxr import Usd, UsdGeom, UsdVol
    stage = Usd.Stage.Open(os.path.join(out, "scene.usda"))
    m = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Tree"))
    cols = UsdGeom.PrimvarsAPI(m.GetPrim()).GetPrimvar("emissiveColor").Get()
    vol = stage.GetPrimAtPath("/World/Fire")
    print(f"2) reloaded: tree has emissiveColor primvar ({len(cols)} verts); Fire volume valid: {vol.IsValid()}")
    assert cols is not None and len(cols) > 1000 and vol.IsValid()
    print("\nexport (USD + OpenVDB derived-state handoff) self-checks passed.")
