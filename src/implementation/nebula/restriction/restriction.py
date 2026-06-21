"""
The restriction operator, assembled (ARCHITECTURE §III.4; "ONE SCALAR, FOUR JOBS").

Collapses a heterogeneous cell into one effective element + the single trust scalar every
other subsystem reads. It composes the three verified error channels into one currency:

  - Voigt-Reuss directional gap            (constitutive RESPONSES error)        V0.1
  - sub-cell variance epsilon              (nonlinear-RATE error, the Jensen term) V1.3
  - connectivity residual g_perc           (the CONNECTIVITY the gap is blind to)  V2.2
  - lod_trust = gap_axial * (1 + g_perc)   (the physics-weighted refine-vs-truncate gate) V2.3

The refinement rule honors report §3.2 exactly: refine when the V-R gap exceeds T_hi OR the
variance eps exceeds eps* (~0.5) OR connectivity is high; and ALWAYS refine on a hard
26-connectivity span of the soft phase (the off-axis thin-connected percolation tail is a
standing always-refine ceiling, report §5.1). The same scalar gates refinement, conservation
tolerance, surrogate trust (later phases), and LOD.
"""
from dataclasses import dataclass, field as _field

import numpy as np

from .homogenization import (isotropic_stiffness, voigt_bound, reuss_bound, relative_gap,
                             containment, directional_estimate, DIR_LABELS)
from .jensen import variance_error_scalar
from . import percolation as pc

# default thresholds (frozen by the verification: V0.1/V0.2 gap T_hi=0.30; V1.3 eps*=0.5)
T_HI = 0.30
EPS_STAR = 0.5
GPERC_HI = 0.5
TA_DEFAULT = 9000.0          # pyrolysis activation temperature (operators.fire.FireParams.Ta_py)


@dataclass
class RestrictionResult:
    """The restriction of one cell: the effective tensor + the single trust scalar's components."""
    fractions: np.ndarray
    C_voigt: np.ndarray
    C_reuss: np.ndarray
    C_est: np.ndarray
    gap: np.ndarray            # (6,) per-direction Voigt-Reuss relative gap
    gap_axial: np.ndarray      # (3,) the axial channels
    eps: float                 # the Jensen sub-cell-variance scalar
    g_perc: np.ndarray         # (3,) directional connectivity residual
    lod_trust: np.ndarray      # (3,) gap_axial * (1 + g_perc)
    trust: float               # THE single scalar (max lod_trust; higher = less trustworthy)
    hard_percolation: bool     # 26-conn soft-phase span across the load plane
    contained: bool            # C_est within [Reuss, Voigt] (theorem sanity)
    refine: bool
    reasons: list = _field(default_factory=list)


def restrict_cell(grid, materials, Tfield=None, Ta=TA_DEFAULT, layer_axis=None, damage_phase=None,
                  T_hi=T_HI, eps_star=EPS_STAR, gperc_hi=GPERC_HI, use_gpu=True):
    """Restrict one heterogeneous cell -> RestrictionResult (effective tensor + trust + refine).

    grid: (n,n,n) int phase ids; materials: list of (E, nu) per phase; Tfield: optional sub-cell
    temperature field for the variance term; layer_axis: the ring/series axis (default: the axis
    of largest gap, the natural layering direction); damage_phase: the defect (char/crack) phase id
    the hard connectivity backstop tracks (default: the weakest phase) -- pass it so an authored
    soft LAYER like bark is not mistaken for a percolating crack.
    """
    grid = np.asarray(grid)
    P = len(materials)
    fractions = np.bincount(grid.ravel(), minlength=P) / grid.size
    C_phases = [isotropic_stiffness(E, nu) for (E, nu) in materials]

    Cv = voigt_bound(fractions, C_phases)
    Cr = reuss_bound(fractions, C_phases)
    gap = relative_gap(Cv, Cr)
    gap_axial = gap[:3]
    if layer_axis is None:
        layer_axis = int(np.argmax(gap_axial))
    C_est = directional_estimate(fractions, C_phases, layer_axis)

    # connectivity: g_perc is the directional conductance residual over the whole field (it is ~1
    # for ANY clean layered cell -- a perfect series path -- so it is folded WITH the gap, never used
    # as a standalone trigger: lod_trust = gap*(1+g_perc) is the V2.3 gate). The hard 26-conn backstop
    # fires only on a DECLARED defect phase (char/crack), so authored soft layers do not trip it.
    present = int((fractions > 0).sum())
    if present > 1:
        g_perc = pc.connectivity_residual(grid, materials, use_gpu=use_gpu)
    else:
        g_perc = np.zeros(3)
    hard_perc = (damage_phase is not None and int(damage_phase) < P and fractions[int(damage_phase)] > 0
                 and pc.percolates_load_plane(grid, materials, phase=int(damage_phase)))

    lod = gap_axial * (1.0 + g_perc)        # V2.3 physics-weighted refine gate (the one currency)

    eps = 0.0
    if Tfield is not None:
        T = np.asarray(Tfield, float)
        eps = float(variance_error_scalar(T.mean(), T.var(), Ta))

    trust = float(lod.max())
    reasons = []
    if trust > T_hi:                        # proxy-error term: gap folded with connectivity
        reasons.append(f"lod_trust {trust:.2f}>{T_hi} (gap {gap_axial.max():.2f}, g_perc {g_perc.max():.2f})")
    if eps > eps_star:                      # nonlinear-rate term (Jensen)
        reasons.append(f"eps {eps:.2f}>{eps_star}")
    if hard_perc:                           # the always-refine ceiling (off-axis thin-connected tail)
        reasons.append("26-conn defect span (always-refine ceiling)")
    refine = len(reasons) > 0

    # per-direction containment: the shipped orthotropic proxy picks each diagonal from a bound
    # (Voigt parallel / Reuss series), so its diagonals lie within [Reuss, Voigt] by construction.
    # (Full-tensor PSD bracketing is a theorem about the TRUE DNS tensor, V0.1 -- not the proxy.)
    contained = bool(containment(C_est, Cv, Cr)["per_dir_ok"].all())

    return RestrictionResult(
        fractions=fractions, C_voigt=Cv, C_reuss=Cr, C_est=C_est, gap=gap, gap_axial=gap_axial,
        eps=eps, g_perc=g_perc, lod_trust=lod, trust=trust, hard_percolation=hard_perc,
        contained=contained, refine=refine, reasons=reasons)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    n = 16
    WOOD, NU = 10.0, 0.3

    # 1) quiescent LOW-CONTRAST layered cell (sapwood/heartwood, contrast ~1.3) -> tight gap, no refine.
    mats2 = [(9.0, 0.35), (12.0, 0.35)]
    g = np.zeros((n, n, n), np.int64)
    g[n // 2:] = 1
    r = restrict_cell(g, mats2, use_gpu=True)
    print(f"1) quiescent wood: gap_axial {r.gap_axial}  trust={r.trust:.3f}  refine={r.refine} {r.reasons}")
    assert not r.refine and r.contained

    # 2) char wedge (soft char invading; char = phase 1) -> gap blows open -> refine (V0.1/V0.2).
    matsC = [(WOOD, NU), (WOOD / 60.0, NU)]
    gc = np.zeros((n, n, n), np.int64)
    xi = (np.arange(n) + 0.5) / n; zi = (np.arange(n) + 0.5) / n
    mask = xi[:, None] < (0.6 * zi)[None, :]
    gc[mask[:, None, :].repeat(n, axis=1)] = 1
    rc = restrict_cell(gc, matsC, damage_phase=1, use_gpu=True)
    print(f"2) char wedge: gap_axial {rc.gap_axial}  g_perc {rc.g_perc}  trust={rc.trust:.3f}  "
          f"refine={rc.refine} {rc.reasons}")
    assert rc.refine

    # 3) connected char seam (thin) -> hard 26-conn percolation (always-refine ceiling) + high g_perc.
    gs = np.zeros((n, n, n), np.int64); gs[n // 2:n // 2 + 1, :, :] = 1
    rs = restrict_cell(gs, matsC, damage_phase=1, use_gpu=True)
    print(f"3) char seam: g_perc {rs.g_perc}  hard_perc={rs.hard_percolation}  refine={rs.refine} {rs.reasons}")
    assert rs.hard_percolation and rs.refine

    # 4) steep sub-cell T gradient -> Jensen eps high -> refine (V1.3).
    Tfield = np.broadcast_to((350.0 + 700.0 * (np.arange(n) + 0.5) / n)[:, None, None], (n, n, n))
    rt = restrict_cell(g, mats2, Tfield=Tfield, use_gpu=True)
    print(f"4) steep T-gradient layered cell: eps={rt.eps:.2f}  refine={rt.refine} {rt.reasons}")
    assert rt.eps > EPS_STAR and rt.refine
    print("\nrestriction operator self-checks passed.")
