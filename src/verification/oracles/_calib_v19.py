"""V1.9 calibration scratch (UNCOMMITTED). Confirms thresholds robust across reduction ratios,
modes, severities; times the full-scale GPU path toward ~1M render points. Not a test."""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import dualcloud as dc

bp = dc.BeamParams()
print("backend:", "GPU (Warp)" if dc._HAS_WARP else "CPU only")

print("\n=== reduction sweep (bend 90deg, twist 180deg): LBS stays low, foil grows ===")
for ds in (2, 3, 5, 8, 12):
    rb = dc.run_case(bp, dense_scale=ds, mode="bend", severity=np.deg2rad(90))
    rt = dc.run_case(bp, dense_scale=ds, mode="twist", severity=np.deg2rad(180))
    print("  scale=%2d red=%6.1fx (Nd=%d): bend LBS=%.4f foil=%.4f (%.1fx) | twist LBS=%.4f foil=%.4f (%.1fx)"
          % (ds, rb["reduction"], rb["Nd"], rb["lbs_mean"], rb["tr_mean"], rb["tr_mean"]/rb["lbs_mean"],
             rt["lbs_mean"], rt["tr_mean"], rt["tr_mean"]/rt["lbs_mean"]))

print("\n=== graceful sweeps (max step-to-step jump in LBS mean error) ===")
for mode, hi in (("bend", 90), ("twist", 180)):
    errs = [dc.run_case(bp, dense_scale=4, mode=mode, severity=np.deg2rad(a))["lbs_mean"]
            for a in np.linspace(0, hi, 10)]
    print("  %-6s LBS errs %s  maxjump=%.4f" % (mode, [round(e, 4) for e in errs], np.diff(errs).max()))

if dc._HAS_WARP:
    print("\n=== full-scale GPU skinning (toward ~1M render points) ===")
    for ds in (12, 18, 24):
        Xc, _, sc = dc.make_lattice(bp, scale=1)
        xc = dc.deform_field(Xc, bp, "twist", np.deg2rad(180))
        Rc = dc.node_rotations(Xc, xc, dc.neighbor_lists(Xc, sc, bp.neighbor_r))
        Xd, _, _ = dc.make_lattice(bp, scale=ds)
        truth = dc.deform_field(Xd, bp, "twist", np.deg2rad(180))
        idx, w = dc.bind_weights(Xd, Xc, k=4)
        t = time.time(); p = dc.skin_lbs_gpu(Xd, idx, w, Xc, xc, Rc); gt = time.time() - t
        err = np.linalg.norm(p - truth, axis=1).mean() / bp.L
        print("  scale=%2d  Nd=%8d  reduction=%8.0fx  GPU skin=%.3fs  LBS mean err=%.4f"
              % (ds, len(Xd), len(Xd) / len(Xc), gt, err))
print("DONE")
