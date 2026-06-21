"""Scratch calibrator for V2.3 — measures the margins behind the FROZEN notebook thresholds.

Mirrors the V2.3 notebook metrics WITHOUT the markdown/figure, so the thresholds in `_build_v23_nb.py`
are set *below* what is actually measured here (repo practice, protocol §1). Uncommitted/auxiliary.
Run: .venv/bin/python src/verification/oracles/_calib_v23.py
"""
import time
import numpy as np
from scipy.stats import spearmanr

import cells
import spectral_lod as sl
import percolation as pc
from dns_elasticity_3d import effective_stiffness, _HAS_GPU

np.set_printoptions(precision=4, suppress=True)
print(f"DNS backend: {'GPU (cupy CG)' if _HAS_GPU else 'CPU (sparse LU)'}")
t0 = time.time()
N = 24


def _seam(ang, thickness=2, contrast=60.0):
    s = pc.seam_cell_at(N, ang, thickness=thickness, contrast=contrast)
    return cells.Cell(grid=s.grid, materials=s.materials, kind="seam", contrast=contrast, layer_axis=None)


# ---------------------------------------------------------- (1) misalignment + (2) co-designed basis
print("\n=== (1) misalignment + (2) co-designed basis: thin char layer, homogenization limit k=1 ===")
print(" contrast | series true | stiff err | comp err | ratio")
mis, codes, ratios = [], [], []
for contrast in (10.0, 30.0, 60.0, 100.0):
    cell = sl.thin_char_layer_cell(n=N, thickness_vox=1, axis=2, contrast=contrast)
    Ct = effective_stiffness(cell.grid, cell.materials)
    s = 2
    es = sl.directional_modulus_error(
        sl.effective_tensor_of_field(sl.reconstruct_field(cell, "stiffness", lambda f: sl.lowpass_axis(f, 1, s))), Ct)[0][s]
    ec = sl.directional_modulus_error(
        sl.effective_tensor_of_field(sl.reconstruct_field(cell, "compliance", lambda f: sl.lowpass_axis(f, 1, s))), Ct)[0][s]
    mis.append(es); codes.append(ec); ratios.append(ec / max(es, 1e-9))
    print(f"   {contrast:6.0f} | {Ct[s, s]:10.3f} | {es:9.3f} | {ec:8.4f} | {ec/max(es,1e-9):6.4f}")
print(f" -> misalignment max stiff err {max(mis):.2f} (>=0.50) ; co-designed max comp err {max(codes):.4f} ; worst ratio {max(ratios):.4f}")

# --------------------------------------------------------------------------- (3) off-axis residual
print("\n=== (3) off-axis residual: BOTH global bases fail at coarse budget (-> refine) ===")
print(" angle | stiff frob | comp frob | min-over-bases | phys-wtd sel | lod_trust")
resid, sel_ratio = [], []
for ang in (30, 45, 60):
    cell = _seam(ang)
    Ct = effective_stiffness(cell.grid, cell.materials)
    K = 32
    geo = sl.directional_modulus_error(sl.effective_tensor_of_field(
        sl.reconstruct_field(cell, "stiffness", lambda f: sl.lowpass_nd(f, K))), Ct)[2]
    cmp = sl.directional_modulus_error(sl.effective_tensor_of_field(
        sl.reconstruct_field(cell, "compliance", lambda f: sl.lowpass_nd(f, K))), Ct)[2]
    g, m = sl.physics_weighted_select(cell, K, rep="compliance")
    sel = sl.directional_modulus_error(effective_stiffness(g, m), Ct)[2]
    resid.append(min(geo, cmp)); sel_ratio.append(sel / max(geo, 1e-9))
    print(f"   {ang:3d} | {geo:10.3f} | {cmp:9.3f} | {min(geo,cmp):14.3f} | {sel:12.3f} | {float(np.max(sl.lod_trust(cell))):8.3f}")
print(f" -> off-axis residual (min over bases) >= {min(resid):.3f} for all (cannot truncate); sel best ratio {min(sel_ratio):.3f}")

# --------------------------------------------- (4) lod_trust correlation + necessity (vs geom energy)
print("\n=== (4) lod_trust vs geometric energy: CONTROLLED layer battery (contrast x thickness) ===")
te, tr, en = [], [], []
for contrast in (5.0, 10.0, 20.0, 40.0, 80.0):
    for thk in (1, 2, 3):
        cell = sl.thin_char_layer_cell(n=N, thickness_vox=thk, axis=2, contrast=contrast)
        Ct = effective_stiffness(cell.grid, cell.materials)
        err = sl.directional_modulus_error(sl.effective_tensor_of_field(
            sl.reconstruct_field(cell, "stiffness", lambda f: sl.lowpass_axis(f, 1, 2))), Ct)[0][2]
        te.append(err); tr.append(float(np.max(sl.lod_trust(cell))))
        en.append(sl.discarded_energy_fraction(sl.to_field(cell, "stiffness"), 1))
rho_t = float(spearmanr(tr, te).correlation)
rho_e = float(spearmanr(en, te).correlation)
print(f" cells {len(te)}: rho(lod_trust, err) = {rho_t:+.3f} ; rho(geom energy, err) = {rho_e:+.3f} ; advantage {rho_t-rho_e:+.3f}")
print(f" geometric discarded-energy range {min(en):.3f}..{max(en):.3f} (contrast-blind)")

# necessity sub-check: seam vs matched control (identical gap) — only g_perc inside lod_trust separates
seam = pc.seam_cell_at(N, 45, thickness=2, contrast=60.0)
ctrl = pc.shuffled_control(seam, seed=7)
sc = cells.Cell(grid=seam.grid, materials=seam.materials, kind="seam", contrast=60.0, layer_axis=None)
cc = cells.Cell(grid=ctrl.grid, materials=ctrl.materials, kind="ctrl", contrast=60.0, layer_axis=None)
print(f" seam vs matched control (identical V-R gap): lod_trust {float(np.max(sl.lod_trust(sc))):.3f} "
      f"vs {float(np.max(sl.lod_trust(cc))):.3f}  (g_perc separates)")

print(f"\ntotal calib time {time.time()-t0:.1f}s")
print("\n--- suggested FROZEN thresholds (set below these) ---")
print(f"  MIS_MIN          = 0.50   (measured max {max(mis):.2f})")
print(f"  CODES_MAX        = 0.02   (measured max {max(codes):.4f})")
print(f"  CODES_RATIO_MAX  = 0.20   (measured worst {max(ratios):.4f})")
print(f"  RESID_MIN        = 0.40   (measured min {min(resid):.3f})")
print(f"  TRUST_RHO_MIN    = 0.85   (measured {rho_t:.3f})")
print(f"  TRUST_ADV_MIN    = 0.15   (measured {rho_t-rho_e:+.3f})")
