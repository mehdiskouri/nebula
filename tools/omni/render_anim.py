"""
Headless Omniverse RTX render of the TIME-SAMPLED burn animation (run inside Kit via --exec).

Opens the time-sampled USD (animated char colours + charring canopy + flame), sets RTX Path Tracing,
a FIXED camera framing the tree, and captures every frame of the timeline (the fire spreading). A
separate step muxes the frames into an MP4. Env: NEBULA_USD, NEBULA_OUT, NEBULA_W, NEBULA_H, NEBULA_ACC.
"""
import os

import carb
import omni.kit.app
import omni.usd
import omni.timeline
from pxr import Usd, UsdGeom, Gf

USD = os.environ.get("NEBULA_USD")
OUTDIR = os.environ.get("NEBULA_OUT")
W = int(os.environ.get("NEBULA_W", "800"))
H = int(os.environ.get("NEBULA_H", "1000"))
ACC = int(os.environ.get("NEBULA_ACC", "26"))
AZ = float(os.environ.get("NEBULA_AZ", "0.6"))
ELEV = float(os.environ.get("NEBULA_ELEV", "0.18"))

app = omni.kit.app.get_app()


def pump(n):
    for _ in range(n):
        app.update()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ctx = omni.usd.get_context()
    ctx.open_stage(USD)
    pump(150)
    stage = ctx.get_stage()
    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/totalSpp", 192)
    s.set("/rtx/pathtracing/spp", 1)
    s.set("/rtx/pathtracing/maxBounces", 5)
    s.set("/persistent/app/viewport/displayOptions", 0)
    s.set("/app/viewport/grid/enabled", False)

    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    vp = get_active_viewport()
    vp.resolution = (W, H)
    vp.camera_path = "/World/Camera"

    # fixed camera framing the above-ground tree
    cache = UsdGeom.BBoxCache(Usd.TimeCode(0), ["default", "render", "proxy"])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Tree")).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    top = float(mx[2])
    center = Gf.Vec3d((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, 0.5 * top)
    radius = 1.65 * max(top, 1.0)
    import math
    eye = center + Gf.Vec3d(radius * math.cos(AZ), radius * math.sin(AZ), ELEV * radius)
    cam = UsdGeom.Xformable(UsdGeom.Camera.Get(stage, "/World/Camera"))
    cam.ClearXformOpOrder()
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 0, 1)).GetInverse())

    tl = omni.timeline.get_timeline_interface()
    fps = stage.GetTimeCodesPerSecond() or 18.0
    nf = int(stage.GetEndTimeCode())
    print(f"[nebula] timeline {nf+1} frames @ {fps}fps; res {W}x{H}")
    for k in range(nf + 1):
        tl.set_current_time(k / fps)
        pump(ACC)
        capture_viewport_to_file(vp, os.path.join(OUTDIR, f"f{k:04d}.png"))
        pump(12)
        print(f"[nebula] frame {k+1}/{nf+1}")
    pump(40)
    print("[nebula] DONE")
    os._exit(0)


main()
