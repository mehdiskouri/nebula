"""
Headless Omniverse RTX render of the Nebula burning-tree USD (run inside Kit via --exec).

Opens the render-ready USD (geometry + derived materials + emissive fire + lights), sets RTX Path
Tracing, orbits a camera around the tree, and captures each frame to PNG. A separate step muxes the
frames into an MP4. Configured by env vars:
  NEBULA_USD, NEBULA_OUT (frame dir), NEBULA_W, NEBULA_H, NEBULA_N (orbit frames), NEBULA_ACC (PT
  accumulation updates per frame), NEBULA_ELEV (camera elevation fraction).
"""
import math
import os
import time

import carb
import omni.kit.app
import omni.usd
from pxr import Usd, UsdGeom, Gf

USD = os.environ.get("NEBULA_USD", "/workspace/nebula/demo_output/omniverse_scene/scene_render.usd")
OUTDIR = os.environ.get("NEBULA_OUT", "/workspace/nebula/demo_output/_omni_frames")
W = int(os.environ.get("NEBULA_W", "900"))
H = int(os.environ.get("NEBULA_H", "1150"))
N = int(os.environ.get("NEBULA_N", "36"))
ACC = int(os.environ.get("NEBULA_ACC", "70"))
ELEV = float(os.environ.get("NEBULA_ELEV", "0.28"))

app = omni.kit.app.get_app()


def pump(n):
    for _ in range(n):
        app.update()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    ctx = omni.usd.get_context()
    print("[nebula] opening", USD)
    ctx.open_stage(USD)
    pump(150)                                    # let assets / materials load
    stage = ctx.get_stage()

    s = carb.settings.get_settings()
    s.set("/rtx/rendermode", "PathTracing")
    s.set("/rtx/pathtracing/totalSpp", 256)
    s.set("/rtx/pathtracing/spp", 1)
    s.set("/rtx/pathtracing/maxBounces", 6)
    s.set("/rtx/pathtracing/maxSamplesPerLaunch", 1000000)
    s.set("/app/captureFrame/setAlphaTo1", True)
    # clean frame: no grid / world-axis gizmo / selection outline
    s.set("/persistent/app/viewport/displayOptions", 0)
    s.set("/app/viewport/grid/enabled", False)
    s.set("/app/viewport/show/camera", False)
    s.set("/app/viewport/show/lights", False)

    # camera + viewport
    cam_path = "/World/Camera"
    cam = UsdGeom.Camera.Get(stage, cam_path)
    camx = UsdGeom.Xformable(cam)
    camx.ClearXformOpOrder()
    op = camx.AddTransformOp()

    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    vp = get_active_viewport()
    if vp is None:
        print("[nebula] no active viewport; creating one")
        from omni.kit.viewport.utility import create_viewport_window
        win = create_viewport_window("nebula", width=W, height=H)
        vp = win.viewport_api
    try:
        vp.resolution = (W, H)
    except Exception as e:
        print("[nebula] set resolution failed:", e)
    vp.camera_path = cam_path

    # scene bounds -> orbit center + radius
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Tree")).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    # frame the ABOVE-GROUND tree (deep roots otherwise inflate the bbox); aim at the trunk/canopy
    top = float(mx[2]); height = max(top, 1.0)
    center = Gf.Vec3d((mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, 0.5 * top)
    radius = 1.55 * height
    print(f"[nebula] center {center} radius {radius:.2f} top {top:.2f} res {W}x{H} frames {N}")

    az0 = 0.6
    for k in range(N):
        t0 = time.time()
        az = az0 + 2 * math.pi * k / N
        eye = center + Gf.Vec3d(radius * math.cos(az), radius * math.sin(az), ELEV * radius)
        view = Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 0, 1))
        op.Set(view.GetInverse())
        pump(ACC)                                # accumulate path-traced samples
        path = os.path.join(OUTDIR, f"f{k:04d}.png")
        capture_viewport_to_file(vp, path)
        pump(15)                                 # let the capture flush to disk
        print(f"[nebula] frame {k+1}/{N} -> {path}  ({time.time()-t0:.1f}s)")

    pump(45)                                     # flush the last async captures to disk
    print("[nebula] DONE")
    # Kit's headless shutdown hangs; the frames are on disk, so force a clean exit.
    os._exit(0)


main()
