"""
Spectral LOD vs physical fidelity — the V2.3 danger case (Decision #24 caveat; Risk: geo/physical).

The architecture's unifying claim is that ONE operation — spectral truncation — is LOD =
homogenization = compilation ("coarse = low-frequency, detail = high-frequency"). The load-bearing
caveat (ARCHITECTURE Part VII; §III.8): *geometric smoothness != physical smoothness*. A char layer
is GEOMETRICALLY THIN (tiny volume fraction, high spatial frequency) but PHYSICALLY DOMINANT (it is
the series bottleneck — the harmonic/Reuss mean is controlled by the softest layer). A low-frequency
GEOMETRIC truncation can therefore silently discard a feature with large physical impact.

The exact mechanism (why this is a theorem, not a vibe): a low-pass keeps the DC component, so
**truncating a field preserves that field's mean.** *Which* mean depends on which field you spectrally
represent:
  - truncate the STIFFNESS field E(x)      -> preserves <E>   = ARITHMETIC / Voigt  mean
        (right for the in-plane / parallel directions; WRONG, over-stiff, for the cross-layer series)
  - truncate the COMPLIANCE field S=1/E    -> preserves <S>   = HARMONIC   / Reuss  mean
        (right for the cross-layer series direction; wrong for the in-plane parallel directions)
A thin char layer is a tiny dip in E (vanishes under a stiffness low-pass -> series modulus jumps to
~uncharred = huge error) but it dominates <S>. So the geometric basis silently chooses the WRONG
average; the physics-co-designed per-channel representation (compliance for series, stiffness for
parallel == `homogenization.directional_estimate`, proven exact for layered media in V0.1) is the cure.

For a regular voxel grid the geometric spectral basis (separable DCT-II) IS the grid graph-Laplacian
eigenbasis — i.e. literally the "keep the lowest graph frequencies" operator `coupling_pipeline.truncate`
applies to the skeleton, now applied to the cell field. So V2.3 tests the *physical faithfulness* of the
very operator V1.4 ships for form.

This module supplies the V2.3 battery + truncation machinery, reusing — WITHOUT editing — `cells`
(`layered_cell`, `char_wedge_cell`, `Cell`), `homogenization` (`voigt_bound`/`reuss_bound`/
`relative_gap`/`directional_estimate`/`isotropic_stiffness`), the DNS oracle
`dns_elasticity_3d.effective_stiffness`, and `percolation` (`seam_cell_at`, `directional_conductance`,
`connectivity_residual`). Pure numpy + scipy.fft (+ the GPU DNS/conduction paths).

Voigt convention [11,22,33,23,13,12], engineering shear — matches homogenization.py.
"""
from dataclasses import dataclass, field as _dc_field

import numpy as np
from scipy.fft import dctn, idctn, dct, idct

import cells
import percolation as pc
from dns_elasticity_3d import effective_stiffness
from homogenization import (isotropic_stiffness, voigt_bound, reuss_bound, relative_gap,
                            directional_estimate)

E_WOOD = 10.0
NU = 0.3
DIR_LABELS = ["11", "22", "33", "23", "13", "12"]


# ============================================================================================
# Cells — the thin-but-physically-dominant features.
# ============================================================================================
def thin_char_layer_cell(n=32, thickness_vox=1, axis=2, contrast=60.0, E_wood=E_WOOD, nu=NU):
    """A THIN char layer (phase 1) of exactly `thickness_vox` voxels, normal to `axis`, centred.

    The canonical V2.3 cell: geometrically thin (fraction ~thickness/n) yet physically dominant in the
    cross-`axis` (series) direction. Returns a `cells.Cell` (so .grid/.materials/.fractions/.C_phases
    all work). `contrast` sets the char softness (E_char = E_wood / contrast).
    """
    materials = [(E_wood, nu), (E_wood / contrast, nu)]
    grid = np.zeros((n, n, n), dtype=np.int64)
    c = n // 2
    lo = c - thickness_vox // 2
    hi = lo + thickness_vox
    idx = [slice(None)] * 3
    idx[axis] = slice(lo, hi)
    grid[tuple(idx)] = 1
    return cells.Cell(grid=grid, materials=materials, kind="thin_char_layer",
                      contrast=float(contrast), layer_axis=axis,
                      meta={"thickness_vox": thickness_vox, "char_fraction": float(grid.mean())})


# ============================================================================================
# Per-voxel modulus field <-> phase grid (the basis co-design knob is WHICH field you truncate).
# ============================================================================================
def to_field(cell, rep="stiffness"):
    """Per-voxel scalar field for spectral truncation. rep in {stiffness, compliance, logE}.

    `stiffness`  -> E(x)        (the naive geometric representation)
    `compliance` -> 1/E(x)      (the physics-co-designed representation for series channels)
    `logE`       -> ln E(x)     (the geometric-mean-preserving representation; a middle ground)
    """
    Es = np.array([E for (E, _) in cell.materials], float)
    E = Es[np.asarray(cell.grid)]
    if rep == "stiffness":
        return E
    if rep == "compliance":
        return 1.0 / E
    if rep == "logE":
        return np.log(E)
    raise ValueError(rep)


def from_field(fld, rep="stiffness"):
    """Invert `to_field`: a (clipped) per-voxel modulus field E(x) from the truncated representation."""
    if rep == "stiffness":
        return fld
    if rep == "compliance":
        return 1.0 / fld
    if rep == "logE":
        return np.exp(fld)
    raise ValueError(rep)


def field_to_phases(Efield, nu=NU, levels=64, E_floor=1e-4):
    """Bin a continuous modulus field into <=`levels` phases so the EXISTING DNS oracle can solve it.

    Each phase's modulus is the MEAN of the field values assigned to its bin (a data-driven center,
    not the bin midpoint) — exact for a few-valued field and near-exact for a continuous one, which
    matters because the harmonic/series response is dominated by the soft end that uniform bins
    under-resolve. Bin EDGES are log-spaced so the soft and stiff ends are resolved proportionally.
    Re-quantization is the only approximation V2.3 introduces; the notebook checks (§A) it is <<2% —
    negligible against the misalignment effect (>50%). Returns (phase_grid:int, materials:list[(E,nu)]).
    """
    E = np.clip(np.asarray(Efield, float), E_floor, None)
    lo, hi = float(E.min()), float(E.max())
    if hi - lo <= 1e-12 * max(abs(hi), 1.0):
        return np.zeros(E.shape, np.int64), [(0.5 * (lo + hi), nu)]
    edges = np.exp(np.linspace(np.log(lo), np.log(hi), levels + 1))
    edges[-1] *= 1.0 + 1e-9                                   # keep the max inside the last bin
    idx = np.clip(np.digitize(E.ravel(), edges) - 1, 0, levels - 1)
    used = np.unique(idx)
    remap = np.full(levels, -1, np.int64)
    remap[used] = np.arange(used.size)
    grid = remap[idx].reshape(E.shape).astype(np.int64)
    # data-driven center: mean modulus of the voxels in each used bin
    sums = np.bincount(idx, weights=E.ravel(), minlength=levels)
    cnts = np.bincount(idx, minlength=levels)
    materials = [(float(sums[u] / cnts[u]), nu) for u in used]
    return grid, materials


# ============================================================================================
# The geometric basis: separable DCT-II == grid graph-Laplacian eigenbasis ("low graph frequency").
# ============================================================================================
def lowpass_axis(fld, k_keep, axis):
    """1-D DCT low-pass along `axis`: keep the `k_keep` lowest frequencies, zero the rest."""
    coeff = dct(fld, axis=axis, norm="ortho")
    sl = [slice(None)] * fld.ndim
    sl[axis] = slice(k_keep, None)
    coeff[tuple(sl)] = 0.0
    return idct(coeff, axis=axis, norm="ortho")


def _freq_order(shape):
    """Indices of DCT coefficients in ascending radial frequency (stable) + the squared-freq map."""
    grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    fmag2 = sum(g.astype(float) ** 2 for g in grids)
    return np.argsort(fmag2.ravel(), kind="stable"), fmag2


def lowpass_nd(fld, k_keep):
    """N-D DCT low-pass: keep the `k_keep` lowest-radial-frequency coefficients, zero the rest."""
    coeff = dctn(fld, norm="ortho")
    order, _ = _freq_order(fld.shape)
    keep = np.zeros(fld.size, bool)
    keep[order[:k_keep]] = True
    return idctn(np.where(keep.reshape(fld.shape), coeff, 0.0), norm="ortho")


def _reconstruct_mask(coeff, keep_mask):
    return idctn(np.where(keep_mask, coeff, 0.0), norm="ortho")


def reconstruct_field(cell, rep, lowpass):
    """Truncate `to_field(cell, rep)` with `lowpass` (a callable field->field), invert to a per-voxel
    modulus field, and CLAMP it to the constituents' physical range [E_min, E_max].

    The clamp suppresses Gibbs over/undershoot of the spectral reconstruction — which would otherwise
    create moduli outside the material range (and, in the compliance domain, negative S that inverts to
    garbage). Genuine constituent values never leave the range; only ringing does. Returns E(x)."""
    Es = np.array([E for (E, _) in cell.materials], float)
    E = from_field(lowpass(to_field(cell, rep)), rep)
    return np.clip(E, float(Es.min()), float(Es.max()))


def discarded_energy_fraction(fld, k_keep):
    """Pure GEOMETRIC truncation-danger signal: fraction of the field's spectral energy dropped by an
    nd low-pass at budget `k_keep`. Geometry-only (no physics) — the metric-4 competitor `lod_trust`
    must beat at predicting the true PHYSICAL error."""
    coeff = dctn(fld, norm="ortho")
    order, _ = _freq_order(fld.shape)
    e = (coeff.ravel() ** 2)
    tot = e.sum()
    return float(e[order[k_keep:]].sum() / (tot + 1e-300))


# ============================================================================================
# Effective-property probes (elastic DNS oracle + cheap conduction proxy) on a modulus field.
# ============================================================================================
def effective_tensor_of_field(Efield, nu=NU, levels=64, use_gpu=True):
    """DNS effective 6x6 tensor of a continuous modulus field (re-quantized to phases)."""
    grid, materials = field_to_phases(Efield, nu=nu, levels=levels)
    return effective_stiffness(grid, materials, use_gpu=use_gpu)


def directional_modulus_error(C_trunc, C_true):
    """Per-Voigt-channel relative error of the diagonal (the directional stiffnesses), length-6, plus
    the worst channel and the whole-tensor Frobenius error. The series channel is where a stiffness-
    domain low-pass fails."""
    dt, dr = np.diag(C_trunc), np.diag(C_true)
    per = np.abs(dt - dr) / np.abs(dr)
    frob = np.linalg.norm(C_trunc - C_true) / np.linalg.norm(C_true)
    return per, float(per.max()), float(frob)


def conduction_min(grid, materials, use_gpu=True):
    """Cheap physical scalar: the minimum directional effective conductance (the series/weak axis).
    Used as the fast scoring function for the physics-weighted mode selection."""
    return float(pc.directional_conductance(grid, materials, use_gpu=use_gpu).min())


# ============================================================================================
# Physics-weighted mode selection — the protocol's "physics-weighted error metric" (off-axis case).
#
# At an EQUAL coefficient budget K, instead of the K lowest-frequency modes, keep the K modes whose
# presence most changes a cheap PHYSICAL property (the conduction proxy). The char feature's modes
# carry little spectral energy but large physical impact, so a physics-weighted score retains them
# where a frequency score discards them. Reduces but (off-axis) does not zero the error -> CONSTRAIN.
# ============================================================================================
def physics_weighted_select(cell, budget, rep="compliance", pool_mult=4, levels=48, use_gpu=True):
    """Keep `budget` DCT modes of `to_field(cell, rep)` chosen by PHYSICAL impact (conduction proxy).

    Candidate pool = the `pool_mult * budget` lowest-frequency modes (so it competes head-to-head with
    the geometric low-pass at the same budget, just choosing *within* a slightly larger low-frequency
    pool by physics). The DC mode is always kept. Importance of a pooled mode = |Delta conduction-min|
    when it is removed from the full-pool reconstruction. Returns the re-quantized (grid, materials)
    of the selected-K reconstruction.
    """
    fld = to_field(cell, rep)
    shape = fld.shape
    coeff = dctn(fld, norm="ortho")
    order, _ = _freq_order(shape)
    P = min(pool_mult * budget, fld.size)
    pool = order[:P]
    pool_mask = np.zeros(fld.size, bool)
    pool_mask[pool] = True
    pool_mask = pool_mask.reshape(shape)

    def prop(mask):
        g, m = field_to_phases(from_field(_reconstruct_mask(coeff, mask), rep), levels=levels)
        return conduction_min(g, m, use_gpu=use_gpu)

    base = prop(pool_mask)
    flatcoeff = coeff.ravel()
    importance = np.zeros(P)
    for i, m in enumerate(pool):
        mask = pool_mask.copy().ravel()
        mask[m] = False
        importance[i] = abs(prop(mask.reshape(shape)) - base)
    # DC (lowest freq, pool[0]) is always kept; pick the top (budget-1) others by physical impact.
    keep_local = {0}
    rank = np.argsort(-importance)
    for r in rank:
        if len(keep_local) >= budget:
            break
        keep_local.add(int(r))
    keep = np.zeros(fld.size, bool)
    keep[pool[list(keep_local)]] = True
    rec = _reconstruct_mask(coeff, keep.reshape(shape))
    return field_to_phases(from_field(rec, rep), levels=levels)


# ============================================================================================
# The adopted fix: the physics-weighted LOD-trust signal == OLD machinery (V0.1 gap + V2.2 g_perc).
# ============================================================================================
def lod_trust(cell, use_gpu=True):
    """Per-axis physical-truncation-danger signal — the metric a geometric coarsener should read.

    It is built from machinery Nebula already has (the "new capability is old machinery" payoff):
      - the V0.1 directional Voigt-Reuss relative gap `relative_gap` (fraction-only) — large in the
        series direction for high-contrast layers, i.e. exactly where a stiffness-domain low-pass is
        unsafe; and
      - the V2.2 directional connectivity residual `connectivity_residual` — the off-axis / connected
        residual the gap cannot see.
    Returns a length-3 per-axis danger = max(axial V-R gap, connectivity residual). Higher = a
    geometric low-pass will incur larger physical error along that axis.
    """
    Cv = voigt_bound(cell.fractions, cell.C_phases)
    Cr = reuss_bound(cell.fractions, cell.C_phases)
    gap_axial = relative_gap(Cv, Cr)[:3]
    gperc = pc.connectivity_residual(cell.grid, cell.materials, use_gpu=use_gpu)
    return np.maximum(gap_axial, gperc)


# ============================================================================================
# Self-check: the module is self-validating like the other oracles.
# ============================================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    from dns_elasticity_3d import _HAS_GPU
    print(f"DNS backend: {'GPU (cupy CG)' if _HAS_GPU else 'CPU (sparse LU)'}\n")

    n = 24
    contrast = 60.0
    layer = thin_char_layer_cell(n=n, thickness_vox=1, axis=2, contrast=contrast)
    C_true = effective_stiffness(layer.grid, layer.materials)
    print(f"thin char layer n={n}, char fraction {layer.meta['char_fraction']:.3f}, contrast {contrast:g}")
    print(f"  true DNS diag C = {np.diag(C_true)}")

    # (i) re-quantization fidelity: DNS of the re-quantized FULL field ~ DNS of the true phases.
    Cq = effective_tensor_of_field(to_field(layer, "stiffness"), levels=64)
    requant_err = np.linalg.norm(Cq - C_true) / np.linalg.norm(C_true)
    print(f"\n(i) re-quantization fidelity: rel err = {requant_err:.4f}  (must be << misalignment)")
    assert requant_err < 0.03, "re-quantization too lossy — raise `levels`."

    # The series (cross-layer) channel is where the thin layer dominates and the misalignment lives.
    # (V0.1: directional_estimate is Reuss-EXACT on this channel; the in-plane normals carry the
    # Backus correction and are only bracketed — not the point here.)
    series = layer.layer_axis
    C_codes = directional_estimate(layer.fractions, layer.C_phases, series)
    codes_series = abs(C_codes[series, series] - C_true[series, series]) / C_true[series, series]
    print(f"\n(ii) co-designed directional estimate, SERIES channel [{series}] vs DNS: "
          f"err {codes_series:.4f}  (V0.1 Reuss-exactness on the series channel)")
    assert codes_series < 0.02, "co-designed (Reuss/series) channel not exact — check directional split."

    # (iii) stiffness-domain GEOMETRIC low-pass loses the layer -> big SERIES error; the physics-
    #       co-designed COMPLIANCE-domain low-pass at the SAME budget keeps it (preserves <1/E>).
    print("\n(iii) geometric low-pass along the layer axis (series-channel relative error):")
    stiff_series_err, comp_series_err = {}, {}
    for k in (1, 2, 4, 8):
        E_s = reconstruct_field(layer, "stiffness", lambda f: lowpass_axis(f, k, series))
        per_s, _, _ = directional_modulus_error(effective_tensor_of_field(E_s), C_true)
        E_c = reconstruct_field(layer, "compliance", lambda f: lowpass_axis(f, k, series))
        per_c, _, _ = directional_modulus_error(effective_tensor_of_field(E_c), C_true)
        stiff_series_err[k], comp_series_err[k] = per_s[series], per_c[series]
        print(f"   k={k:2d}:  stiffness-domain {per_s[series]:.3f}   |   compliance-domain {per_c[series]:.3f}")
    assert stiff_series_err[1] > 0.5, "expected a large series-channel error from stiffness-domain truncation."
    assert comp_series_err[1] < 0.02, "co-designed (compliance) basis should be exact at the homogenization limit."
    print("\n   -> at the homogenization limit (k=1) the compliance (co-designed) basis is EXACT while the")
    print("      stiffness basis over-stiffens the series direction by >200%. (A truncated cosine basis")
    print("      still spreads a 1-voxel feature at intermediate k — the same coupling, one level deeper.)")

    # (iv) physics-weighted mode SELECTION beats frequency-ordering at an equal intermediate budget,
    #      on the off-axis char WEDGE (no clean principal split).
    wedge = cells.char_wedge_cell(n=n, depth=0.6, contrast=contrast)
    Cw_true = effective_stiffness(wedge.grid, wedge.materials)
    K = 24
    E_geo = reconstruct_field(wedge, "stiffness", lambda f: lowpass_nd(f, K))
    _, geo_worst, geo_frob = directional_modulus_error(effective_tensor_of_field(E_geo), Cw_true)
    g_sel, m_sel = physics_weighted_select(wedge, K, rep="compliance", pool_mult=4)
    _, sel_worst, sel_frob = directional_modulus_error(effective_stiffness(g_sel, m_sel), Cw_true)
    print(f"\n(iv) char wedge, equal budget K={K}: geometric frob {geo_frob:.3f} (worst {geo_worst:.3f}) "
          f"-> physics-weighted frob {sel_frob:.3f} (worst {sel_worst:.3f})")
    assert sel_frob <= geo_frob, "physics-weighted selection should not be worse than geometric."

    # (iv) lod_trust flags the dangerous (series) axis; off-axis seam raises the connectivity term.
    print("\n(iv) lod_trust (V0.1 gap + V2.2 g_perc), per-axis danger:")
    print(f"   thin layer  : {lod_trust(layer)}  (series axis {series} highest)")
    seam = pc.seam_cell_at(n, 45, thickness=2, contrast=contrast)
    seam_cell = cells.Cell(grid=seam.grid, materials=seam.materials, kind="seam", contrast=contrast)
    print(f"   45deg seam  : {lod_trust(seam_cell)}  (connectivity term lifts off-axis danger)")
    print("\nspectral_lod self-check PASSED.")
