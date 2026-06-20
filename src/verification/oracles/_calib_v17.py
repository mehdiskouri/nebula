"""V1.7 calibration scratch (UNCOMMITTED). Finds per-domain setpoints k and prints the
numbers used to FREEZE the notebook's pass thresholds (with margin). Not a test.

Compliance ratio = CONTINUOUS compliance at matched CONTINUOUS volume (mean ρ): both Wolff
and SIMP fields scored with the same penalized stiffness, SIMP optimized at volfrac=mean(ρ_w).
This is the fair, standard comparison (SIMP is the minimizer ⇒ ratio ≥ 1) and avoids the
binarization-disconnection artifact. Connectivity is a SEPARATE check on the ρ>0.5 solid set.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
import topology_opt as to
import wolff as wf

fp = to.FEParams()


def compare(dims, dom, supp, wp, simp_iter=60, label=""):
    t = time.time()
    res = wf.run_wolff(dims, dom, wp=wp, fp=fp)
    Vc = float(res["rho"].mean())                  # continuous volume for matching
    c_w = to.compliance_of(dims, res["rho"], dom, fp)
    rho_s, _ = to.simp_optimize(dims, dom, volfrac=Vc, params=fp, n_iter=simp_iter)
    c_s = to.compliance_of(dims, rho_s, dom, fp)
    con = to.connectivity((res["rho"] > 0.5).astype(float), supp)
    dt = time.time() - t
    print("[%s] k=%.4g conv=%s its=%d  Vcont=%.3f solidfrac=%.3f  ratio=%.3f (cw=%.3e cs=%.3e) "
          "conn=%s frac=%.3f ncomp=%d (%.1fs)"
          % (label, wp.k, res["converged"], res["n_iter"], Vc, res["solid_frac"],
             c_w / c_s, c_w, c_s, con["connected"], con["frac_in_largest"],
             con["n_components"], dt), flush=True)
    return res, c_w / c_s, con


print("=== CANTILEVER (calibration) dims=(32,12,6) ===", flush=True)
dims = (32, 12, 6)
dom, supp = to.build_cantilever(dims, load=1.0)
for k in [0.08, 0.10, 0.12]:
    compare(dims, dom, supp, wf.WolffParams(k=k), label="cant")

print("\n=== CREATURE earth-g dims=(24,8,20) ===", flush=True)
dimc = (24, 8, 20)
dom_e, supp_c = to.build_creature(dimc, g=1.0)
for k in [100.0, 300.0, 600.0, 1000.0]:
    compare(dimc, dom_e, supp_c, wf.WolffParams(k=k), label="creat")

print("\n=== CREATURE gravity sweep (fixed k=KC) + seraph ===", flush=True)
KC = 400.0   # adjust after the earth-g block
for g in [1.0, 0.4, 0.15]:
    dom_g, supp_g = to.build_creature(dimc, g=g)
    res = wf.run_wolff(dimc, dom_g, wp=wf.WolffParams(k=KC), fp=fp)
    con = to.connectivity((res["rho"] > 0.5).astype(float), supp_g)
    print("  g=%.2f  solidfrac=%.3f Vcont=%.3f conv=%s conn=%s frac=%.3f"
          % (g, res["solid_frac"], res["rho"].mean(), res["converged"],
             con["connected"], con["frac_in_largest"]), flush=True)
dom_s, _ = to.build_creature(dimc, g=1.0, support_alpha=1.0)
res = wf.run_wolff(dimc, dom_s, wp=wf.WolffParams(k=KC), fp=fp)
print("  seraph(alpha=1) solidfrac=%.4f Vcont=%.4f conv=%s"
      % (res["solid_frac"], res["rho"].mean(), res["converged"]), flush=True)
print("DONE", flush=True)
