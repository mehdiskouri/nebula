"""SCRATCH (uncommitted) calibration for V2.2 — measures margins to freeze the notebook thresholds.

Covers the original three metrics AND the graded-fix additions:
  (1) gap blindness   : gap(seam) == gap(control) exactly; seam-vs-control stiffness ratio.
  (2) off-axis blind  : best-axis orthotropic-proxy error vs DNS (worst at 45 deg).
  (3) trigger         : percolates(seam) / percolates(control).
  (4) GRADED informativeness : Spearman(g_perc, true DNS weakness) over the battery; AND the
                               per-pair separation g_perc(seam) > g_perc(control) (identical fractions).
  (5) THIN/diagonal robustness : 6-conn vs 26-conn detection on thickness-1/2 diagonal seams + control FP.
  (6) COST            : conductance-proxy wall-time vs the elastic DNS RVE on the same cell.
  (7) SINGLE-CURRENCY : connectivity-aware envelope score(seam) > score(control) per pair, where the
                        fraction-only envelope gives IDENTICAL scores (zero discrimination).
Builds + caches the DNS solves to verification_notebooks/phase2/cache/v22_*.npz.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from scipy.stats import spearmanr

import percolation as pc
import violent_cells as vc
from surrogate_gnn import EnvelopeDetector
from dns_elasticity_3d import effective_stiffness

N = 24
THICK = 3
ANGLES = [0, 15, 30, 45, 60, 75, 90]
CONTRASTS = [60.0, 100.0]
CACHE = pathlib.Path(__file__).parents[3] / "verification_notebooks" / "phase2" / "cache"
CACHE.mkdir(exist_ok=True)


def solve_cached(cell, tag):
    f = CACHE / f"v22_{tag}.npz"
    if f.exists():
        return np.load(f)["C"]
    C = effective_stiffness(cell.grid, cell.materials)
    np.savez(f, C=C)
    return C


t0 = time.time()
print(f"N={N} thick={THICK}  angles={ANGLES}  contrasts={CONTRASTS}")

# ---- the seam/control battery (reuses the cached DNS) + per-cell graded quantities -------------
gp_max, weak, kinds = [], [], []          # for metric 4 (correlation across all cells)
pair_gp_sep = []                          # g_perc(seam) - g_perc(control) per pair
for contrast in CONTRASTS:
    print(f"\n=== contrast {contrast:g} ===")
    print(" th | perc s/c | gap== | Eseam/ctrl ratio | proxyErr | g_perc s/c | sf26 s/c")
    for th in ANGLES:
        seam = pc.seam_cell_at(N, th, thickness=THICK, contrast=contrast)
        ctrl = pc.shuffled_control(seam, seed=1000 + th)
        C_s = solve_cached(seam, f"c{int(contrast)}_a{th}_seam")
        C_c = solve_cached(ctrl, f"c{int(contrast)}_a{th}_ctrl")
        g_s, g_c = pc.gap_vector(seam), pc.gap_vector(ctrl)
        nrm = pc.seam_normal(th)
        En_s, En_c = pc.uniaxial_modulus(C_s, nrm), pc.uniaxial_modulus(C_c, nrm)
        perr, _ = pc.best_axis_proxy_error(seam, C_s)
        gps = pc.connectivity_residual(seam.grid, seam.materials).max()
        gpc = pc.connectivity_residual(ctrl.grid, ctrl.materials).max()
        sfs = pc.spanning_fraction_loadplane(seam.grid, seam.materials)
        sfc = pc.spanning_fraction_loadplane(ctrl.grid, ctrl.materials)
        pair_gp_sep.append(gps - gpc)
        for cell, C, gp, kd in ((seam, C_s, gps, "seam"), (ctrl, C_c, gpc, "ctrl")):
            gp_max.append(gp); weak.append(-pc.min_modulus_xz(C)); kinds.append(kd)
        print(f" {th:2d} |  {pc.percolates_xz(seam):d}/{pc.percolates_xz(ctrl):d}    | "
              f"{str(np.array_equal(g_s,g_c)):5s} | {En_s:6.3f}/{En_c:6.3f} {En_s/En_c:.2f} | "
              f"{perr:.2f}     | {gps:.3f}/{gpc:.3f} | {sfs:.2f}/{sfc:.2f}")

# ---- metric 4: graded informativeness ----------------------------------------------------------
rho, _ = spearmanr(gp_max, weak)
print(f"\n(4) GRADED informativeness:")
print(f"    Spearman(g_perc, true DNS weakness) over {len(gp_max)} cells = {rho:.3f}")
print(f"    per-pair g_perc(seam)-g_perc(control): min={min(pair_gp_sep):.3f} "
      f"(all > 0: {all(s > 0 for s in pair_gp_sep)})")

# ---- metric 5: thin/diagonal robustness (split by thickness) -----------------------------------
print(f"\n(5) THIN/diagonal robustness (6-conn vs 26-conn detect; control FP; g_perc separation):")
for thick in (1, 2):
    old, new, fp, gp_sep = [], [], [], []
    for th in (30, 45, 60):
        for contrast in CONTRASTS:
            s = pc.seam_cell_at(N, th, thickness=thick, contrast=contrast)
            c = pc.shuffled_control(s, seed=5000 + th + thick)
            old.append(pc.percolates_xz(s)); new.append(pc.percolates_xz_hard(s))
            fp.append(pc.percolates_xz_hard(c))
            gp_sep.append(pc.connectivity_residual(s.grid, s.materials).max()
                          - pc.connectivity_residual(c.grid, c.materials).max())
    print(f"    thickness {thick}: 6-conn {np.mean(old)*100:.0f}% -> 26-conn {np.mean(new)*100:.0f}% "
          f"detect; control 26-conn FP {np.mean(fp)*100:.0f}%; g_perc(seam-ctrl) min {min(gp_sep):+.3f} "
          f"(all>0 {all(g > 0 for g in gp_sep)})")

# ---- metric 6: cost ----------------------------------------------------------------------------
probe = pc.seam_cell_at(N, 45, thickness=THICK, contrast=100.0)
_ = pc.directional_conductance(probe.grid, probe.materials)   # warm up
t = time.time(); pc.directional_conductance(probe.grid, probe.materials); t_cond = time.time() - t
t = time.time(); effective_stiffness(probe.grid, probe.materials); t_dns = time.time() - t
print(f"\n(6) COST: conductance {t_cond*1e3:.1f} ms vs elastic DNS {t_dns*1e3:.1f} ms  "
      f"-> ratio {t_cond/t_dns:.3f}")

# ---- metric 7: single-currency restoration (fraction-controlled per-pair discrimination) -------
# The connectivity signal now lives IN the descriptor (channels 20:23), so the trust-scalar
# coordinate discriminates a matched pair that the fraction coordinate (0:20) provably cannot.
# Airtight, threshold-free statement: the appended connectivity channels differ (seam more connected)
# while the original 20-vec is byte-identical.
seams = [pc.seam_cell_at(N, th, thickness=THICK, contrast=c) for c in CONTRASTS for th in ANGLES]
ctrls = [pc.shuffled_control(s, seed=1000 + i) for i, s in enumerate(seams)]
Ds = np.stack([vc.descriptor(s.grid, s.materials, connectivity=True) for s in seams])
Dc = np.stack([vc.descriptor(s.grid, s.materials, connectivity=True) for s in ctrls])
frac_identical = np.array_equal(Ds[:, :20], Dc[:, :20])       # fraction coordinate: zero discrimination
conn_sep = Ds[:, 20:].max(1) - Dc[:, 20:].max(1)             # connectivity coordinate: seam - control
print(f"\n(7) SINGLE-CURRENCY (fraction-controlled per-pair discrimination):")
print(f"    fraction sub-descriptor (0:20) byte-identical seam-vs-control: {frac_identical} "
      f"(ZERO discrimination — the V-R blind spot)")
print(f"    connectivity sub-descriptor (20:): max(g_perc) seam > control for "
      f"{np.mean(conn_sep > 0)*100:.0f}% of pairs; min margin {conn_sep.min():+.3f}")

print(f"\n({time.time()-t0:.0f}s)")
