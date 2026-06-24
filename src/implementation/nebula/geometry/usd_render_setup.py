"""
Render-ready USD composition for Omniverse RTX (the path-traced beauty render).

Keeps the export (`geometry/export`) as the pure derived-state handoff (USD geometry + primvars +
fire.vdb + manifest) and composes a SEPARATE render layer over it that the path tracer needs:
  - PBR materials that honour the DERIVED primvars (UsdPreviewSurface + UsdPrimvarReader: diffuse
    from `displayColor`, emission from `emissiveColor` — so char/ember show);
  - the FIRE as emissive geometry sampled from the temperature field (blackbody colour from
    `geometry.appearance`, HDR emissive intensity ∝ T⁴) — in path tracing this is an *area light*,
    so the fire actually illuminates the bark/canopy (the earned light transport);
  - the smoke VDB volume (density) for absorption;
  - a dim dome + key light and an orbiting camera.

The materials/lights are authored with `usd-core` (no GPU); the Kit script `tools/omni_render.py`
opens the result, sets RTX Path Tracing, and captures frames.
"""
import os

import numpy as np

from . import appearance as ap


def compose_render_usd(scene_dir, out_usd=None, flame_T_hot=620.0, max_flame=12000,
                       emissive_scale=22.0):
    from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf, Vt
    scene_usda = os.path.join(scene_dir, "scene.usda")
    fields_npz = os.path.join(scene_dir, "fire_fields.npz")
    out_usd = out_usd or os.path.join(scene_dir, "scene_render.usd")

    stage = Usd.Stage.CreateNew(out_usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    # reference the exported derived-state scene (geometry + primvars + fire volume)
    stage.GetRootLayer().subLayerPaths.append("./scene.usda")

    # --- PBR material honouring the derived primvars (diffuse=displayColor, emission=emissiveColor) ---
    looks = UsdGeom.Scope.Define(stage, "/World/Looks")
    mat = UsdShade.Material.Define(stage, "/World/Looks/Wood")
    pbr = UsdShade.Shader.Define(stage, "/World/Looks/Wood/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    rd = UsdShade.Shader.Define(stage, "/World/Looks/Wood/DiffuseReader")
    rd.CreateIdAttr("UsdPrimvarReader_float3")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    rd.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
    re = UsdShade.Shader.Define(stage, "/World/Looks/Wood/EmissReader")
    re.CreateIdAttr("UsdPrimvarReader_float3")
    re.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("emissiveColor")
    re.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(re.ConnectableAPI(), "result")
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")
    over = stage.OverridePrim("/World/Tree")
    UsdShade.MaterialBindingAPI(over).Bind(mat)

    # --- FIRE as emissive geometry (a natural blackbody TONGUE) — a real area light in path tracing ---
    flame_center = None
    if os.path.exists(fields_npz):
        from ..render.splat import flame_particles
        d = np.load(fields_npz)
        T = d["temperature"]; vs = float(d["voxel_size"]); org = d["origin"].astype(float)
        pos, Tv, fsize = flame_particles({"T": T}, org, vs, n=max_flame, T_hot=flame_T_hot, seed=7)
        if len(pos):
            inten = ap.emission_intensity(Tv, T_on=600.0, T_full=1500.0)
            col = ap.blackbody_rgb(Tv) * inten[:, None] * emissive_scale
            flame_center = pos.mean(0)
            pts = UsdGeom.Points.Define(stage, "/World/Fire/Flame")
            pts.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pos.astype(np.float32)))
            pts.CreateWidthsAttr(Vt.FloatArray((1.4 * fsize).astype(float).tolist()))
            UsdGeom.PrimvarsAPI(pts).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.clip(col, 0, 50).astype(np.float32)))
            # emissive material driven by the per-point blackbody colour
            fm = UsdShade.Material.Define(stage, "/World/Looks/Flame")
            fs = UsdShade.Shader.Define(stage, "/World/Looks/Flame/PBR")
            fs.CreateIdAttr("UsdPreviewSurface")
            fr = UsdShade.Shader.Define(stage, "/World/Looks/Flame/Reader")
            fr.CreateIdAttr("UsdPrimvarReader_float3")
            fr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
            fr.CreateOutput("result", Sdf.ValueTypeNames.Float3)
            fs.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(fr.ConnectableAPI(), "result")
            fs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
            fm.CreateSurfaceOutput().ConnectToSource(fs.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI(pts).Bind(fm)

    # --- lighting: a dim dome (ambient) + a soft key; the fire is the real light source ---
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(180.0); dome.CreateColorAttr(Gf.Vec3f(0.35, 0.45, 0.7))   # cool sky
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(900.0); key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9)); key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-55, 20, 0))

    # --- camera (the Kit script re-poses it per orbit frame) ---
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(28.0); cam.CreateHorizontalApertureAttr(36.0)

    stage.GetRootLayer().Save()
    return {"usd": out_usd, "flame_center": None if flame_center is None else flame_center.tolist(),
            "n_flame": 0 if flame_center is None else int(len(pos))}


if __name__ == "__main__":
    import sys
    sd = sys.argv[1] if len(sys.argv) > 1 else "/workspace/nebula/demo_output/omniverse_scene"
    info = compose_render_usd(sd)
    print("composed render USD:", info["usd"], "| flame points:", info["n_flame"])
    from pxr import Usd
    assert Usd.Stage.Open(info["usd"]) is not None
    print("render USD opens OK")
