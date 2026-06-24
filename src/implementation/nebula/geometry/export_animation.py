"""
Time-sampled USD of the burning-tree ANIMATION for Omniverse RTX (the path-traced beauty render).

Authors ONE USD whose char colours, charring canopy, and flame evolve over the timeline — the fire
SPREADING through the tree, not a camera orbit. Driven by `operators.tree_fire`. The flame is the Tg
VOLUME sampled into time-sampled EMISSIVE geometry (blackbody) — a real area light in path tracing
that lights the bark as the fire climbs (reliable in RTX without a custom volume material). A Kit
script renders the timeline (`tools/omni/render_anim.py`).
"""
import os

import numpy as np

from ..operators.growth import grow_tree, GrowthParams
from ..operators import canopy as cano
from ..operators import tree_fire as tfire
from ..geometry import mesh_export as me
from ..geometry import appearance as ap
from ..geometry.bark_texture import bark_relief, apply_fissure_albedo


def export_burn_animation(out_dir, seed=7, n_frames=56, fps=18, flame_T_hot=560.0, max_flame=14000,
                          emissive_scale=20.0):
    from pxr import Usd, UsdGeom, UsdShade, UsdLux, Sdf, Gf, Vt
    os.makedirs(out_dir, exist_ok=True)
    tree = grow_tree(seed=seed, gp=GrowthParams(dim=3))
    can = cano.generate_canopy(tree, cano.CanopyParams(), seed=seed)
    tf, times, hist = tfire.simulate(tree, can, n_frames=n_frames)
    verts, faces, vnode = me.tube_mesh(tree)
    # bark relief (static) baked into the geometry + base albedo
    rel = bark_relief(tree, verts, vnode, seed=seed); verts = rel["verts_relief"]
    rho, _, rb, rh, _ = me.segment_field(verts, tree)
    base = np.tile(me.COL_SAPWOOD / 255.0, (len(verts), 1))
    base[rho > rb] = me.COL_BARK / 255.0; base[rho < rh] = me.COL_HEARTWOOD / 255.0
    base = apply_fissure_albedo(base, rel["fissure"])
    nbf = tf.n_branch

    usd = os.path.join(out_dir, "burn_anim.usdc")          # binary USD (per-vertex colour samples are heavy)
    stage = Usd.Stage.CreateNew(usd)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0); stage.SetEndTimeCode(n_frames - 1)
    stage.SetTimeCodesPerSecond(fps)

    # --- tree mesh (static geometry, time-sampled colours) ---
    mesh = UsdGeom.Mesh.Define(stage, "/World/Tree")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.astype(np.int32).flatten().tolist()))
    pv = UsdGeom.PrimvarsAPI(mesh)
    dcol = pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    ecol = pv.CreatePrimvar("emissiveColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
    # canopy points (static positions, time-sampled colour + widths→0 when consumed)
    pts_l = UsdGeom.Points.Define(stage, "/World/Canopy")
    from ..render.splat import canopy_splats
    cs = canopy_splats(can)
    pts_l.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(cs["means"].astype(np.float32)))
    lcol = UsdGeom.PrimvarsAPI(pts_l).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                    UsdGeom.Tokens.vertex)
    lwid = pts_l.CreateWidthsAttr()
    leaf_base = cs["color"]; leaf_w0 = np.full(len(cs["means"]), 0.05)
    # flame emissive points (time-sampled)
    flame = UsdGeom.Points.Define(stage, "/World/Fire")
    fpts = flame.CreatePointsAttr(); fwid = flame.CreateWidthsAttr()
    fcol = UsdGeom.PrimvarsAPI(flame).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                    UsdGeom.Tokens.vertex)

    org, fsp = tf.origin, tf.spacing
    for k, fs in enumerate(hist):
        t = Usd.TimeCode(k)
        # tree char + ember
        cn = np.zeros(tree.n); cn[tf.branch_node] = fs["char"][:nbf]
        bn = np.zeros(tree.n, bool); bn[tf.branch_node] = fs["burning"][:nbf]
        cv = cn[vnode]; bv = bn[vnode]
        TgV = me._sample(fs["Tg"], org, fsp, verts)
        surf = ap.surface_appearance(base, T=np.where(bv, np.clip(TgV, 850, 1350), 0.0), chi=cv)
        dcol.Set(Vt.Vec3fArray.FromNumpy(np.clip(surf["albedo"], 0, 1).astype(np.float32)), t)
        ecol.Set(Vt.Vec3fArray.FromNumpy(np.clip(surf["emission"], 0, 30).astype(np.float32)), t)
        # canopy charring
        lchar = fs["char"][nbf:]; lburn = fs["burning"][nbf:]
        c = np.clip(lchar, 0, 1)[:, None]
        lc = (1 - np.clip(c * 1.6, 0, 1)) * leaf_base + np.clip(c * 1.6, 0, 1) * np.array([0.28, 0.15, 0.04])
        lc = (1 - np.clip((c - 0.5) * 2, 0, 1)) * lc + np.clip((c - 0.5) * 2, 0, 1) * np.array([0.05, 0.045, 0.04])
        if lburn.any():
            lc[lburn] = lc[lburn] + ap.ember_emission(np.full(int(lburn.sum()), 1150.0), chi=np.ones(int(lburn.sum())))
        lcol.Set(Vt.Vec3fArray.FromNumpy(np.clip(lc, 0, 30).astype(np.float32)), t)
        lwid.Set(Vt.FloatArray((leaf_w0 * (lchar < 0.92)).astype(float).tolist()), t)
        # flame from the Tg volume (emissive blackbody points)
        hot = np.argwhere(fs["Tg"] > flame_T_hot)
        if len(hot):
            Tv = fs["Tg"][hot[:, 0], hot[:, 1], hot[:, 2]]
            if len(hot) > max_flame:
                keep = np.argsort(-Tv)[:max_flame]; hot, Tv = hot[keep], Tv[keep]
            fp = org[None, :] + (hot + 0.5) * fsp
            fcl = ap.blackbody_rgb(Tv) * ap.emission_intensity(Tv, T_on=500, T_full=1400)[:, None] * emissive_scale
            fpts.Set(Vt.Vec3fArray.FromNumpy(fp.astype(np.float32)), t)
            fwid.Set(Vt.FloatArray([float(0.95 * fsp)] * len(fp)), t)
            fcol.Set(Vt.Vec3fArray.FromNumpy(np.clip(fcl, 0, 60).astype(np.float32)), t)
        else:
            fpts.Set(Vt.Vec3fArray.FromNumpy(np.zeros((1, 3), np.float32)), t)
            fwid.Set(Vt.FloatArray([0.0]), t); fcol.Set(Vt.Vec3fArray.FromNumpy(np.zeros((1, 3), np.float32)), t)

    # --- materials (emissive from displayColor), lights, camera ---
    def emissive_mat(path, prim, diffuse=True):
        m = UsdShade.Material.Define(stage, path); sh = UsdShade.Shader.Define(stage, path + "/S")
        sh.CreateIdAttr("UsdPreviewSurface")
        r = UsdShade.Shader.Define(stage, path + "/E"); r.CreateIdAttr("UsdPrimvarReader_float3")
        r.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("emissiveColor" if not diffuse else "displayColor")
        r.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(r.ConnectableAPI(), "result")
        if diffuse:
            rd = UsdShade.Shader.Define(stage, path + "/D"); rd.CreateIdAttr("UsdPrimvarReader_float3")
            rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
            rd.CreateOutput("result", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
            er = UsdShade.Shader.Define(stage, path + "/EE"); er.CreateIdAttr("UsdPrimvarReader_float3")
            er.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("emissiveColor")
            er.CreateOutput("result", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(er.ConnectableAPI(), "result")
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        else:
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(m)
    emissive_mat("/World/Looks/Wood", mesh, diffuse=True)
    emissive_mat("/World/Looks/Leaf", pts_l, diffuse=True)
    emissive_mat("/World/Looks/Flame", flame, diffuse=False)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(140.0); dome.CreateColorAttr(Gf.Vec3f(0.35, 0.45, 0.7))
    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(700.0); key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9)); key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-55, 20, 0))
    UsdGeom.Camera.Define(stage, "/World/Camera").CreateFocalLengthAttr(30.0)
    stage.GetRootLayer().Save()
    return {"usd": usd, "n_frames": n_frames, "fps": fps}


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/workspace/nebula/demo_output/omniverse_anim"
    info = export_burn_animation(out, n_frames=int(sys.argv[2]) if len(sys.argv) > 2 else 56)
    print("wrote time-sampled burn USD:", info["usd"], "frames", info["n_frames"])
    from pxr import Usd
    assert Usd.Stage.Open(info["usd"])
    print("opens OK")
