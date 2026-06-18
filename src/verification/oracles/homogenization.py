"""
Homogenization proxy under test (Decision #15; ARCHITECTURE.md §III.4).

Claim being supported: a heterogeneous cell can be collapsed to one effective
6x6 stiffness tensor whose true value is *bracketed* by the Voigt and Reuss
bounds in every direction, and the width of that bracket (the Voigt-Reuss gap)
is the per-cell trust scalar that drives refinement / LOD / surrogate trust.

- Voigt  bound = volume-weighted ARITHMETIC mean of stiffness  (uniform-strain).
- Reuss  bound = volume-weighted HARMONIC   mean of stiffness  (uniform-stress).
- For any macroscopic strain e:   e:C_Reuss:e  <=  e:C_eff:e  <=  e:C_Voigt:e.

This module is pure numpy. The independent ground truth it is checked against
lives in dns_elasticity_3d.py (a direct fine-scale solve).

Voigt notation throughout: stress/strain are length-6 vectors ordered
[11, 22, 33, 23, 13, 12]; shear strains are engineering (gamma = 2*eps).
"""
import numpy as np

# Voigt-index labels for the 3 axial ("principal direction") + 3 shear channels.
DIR_LABELS = ["11", "22", "33", "23", "13", "12"]
AXIAL = (0, 1, 2)
SHEAR = (3, 4, 5)


def isotropic_stiffness(E, nu):
    """6x6 isotropic elastic stiffness (Voigt, engineering shear)."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu
    return C


def voigt_bound(fractions, C_phases):
    """Sum_i f_i C_i  — arithmetic mean of stiffness (uniform-strain upper bound)."""
    fractions = np.asarray(fractions, float)
    return np.tensordot(fractions, np.asarray(C_phases, float), axes=(0, 0))


def reuss_bound(fractions, C_phases):
    """(Sum_i f_i C_i^-1)^-1 — harmonic mean of stiffness (uniform-stress lower bound)."""
    fractions = np.asarray(fractions, float)
    S = np.stack([np.linalg.inv(C) for C in C_phases])  # compliances
    S_eff = np.tensordot(fractions, S, axes=(0, 0))
    return np.linalg.inv(S_eff)


def relative_gap(C_voigt, C_reuss):
    """Per-direction (Voigt-Reuss)/mean for the 6 Voigt channels — the trust scalar.

    Returns a length-6 array aligned with DIR_LABELS.
    """
    dv = np.diag(C_voigt)
    dr = np.diag(C_reuss)
    mean = 0.5 * (dv + dr)
    return (dv - dr) / mean


def containment(C_eff, C_voigt, C_reuss, tol=1e-9):
    """Check the bracketing theorem C_Reuss <= C_eff <= C_Voigt.

    Returns a dict with:
      - signed[d] in [0,1]: position of C_eff[d,d] inside [Reuss, Voigt] per channel.
        (0 == on Reuss, 1 == on Voigt; outside [0,1] == violation.)
      - per_dir_ok : bool array, signed within [-tol, 1+tol] per channel.
      - psd_lower / psd_upper : min eigenvalue of (C_eff - C_R) and (C_V - C_eff);
        the rigorous full-tensor statement (both must be >= -tol).
      - ok : overall containment (per-direction AND positive-semidefinite).
    """
    dv = np.diag(C_voigt)
    dr = np.diag(C_reuss)
    de = np.diag(C_eff)
    gap = dv - dr
    signed = np.where(gap > 0, (de - dr) / np.where(gap == 0, 1.0, gap), 0.0)
    per_dir_ok = (signed >= -tol) & (signed <= 1.0 + tol)

    psd_lower = float(np.linalg.eigvalsh(C_eff - C_reuss).min())
    psd_upper = float(np.linalg.eigvalsh(C_voigt - C_eff).min())
    psd_ok = (psd_lower >= -tol) and (psd_upper >= -tol)

    return {
        "signed": signed,
        "per_dir_ok": per_dir_ok,
        "psd_lower": psd_lower,
        "psd_upper": psd_upper,
        "psd_ok": psd_ok,
        "ok": bool(per_dir_ok.all() and psd_ok),
    }


def directional_estimate(fractions, C_phases, layer_axis):
    """The orthotropic proxy Nebula actually ships for a LAYERED cell.

    Layers stacked along `layer_axis` (0,1,2). The rule is "series vs parallel":
    deformation modes that load the layers *in series* (stress continuous across
    the stack) follow Reuss; modes loading them *in parallel* (strain continuous)
    follow Voigt.
      - Reuss/series : the across-layer normal AND the two out-of-plane shears
                       (the shears that involve the layer-normal axis).
      - Voigt/parallel: the two in-plane normals and the in-plane shear.
    For clean layered media these coincide EXACTLY with the bounds in the
    principal directions (ARCHITECTURE.md §III.4); the Voigt-Reuss gap is the
    error bar on the in-plane normals (which carry the Backus correction).
    """
    C_v = voigt_bound(fractions, C_phases)
    C_r = reuss_bound(fractions, C_phases)
    C_est = C_v.copy()
    in_plane_shear = {0: 3, 1: 4, 2: 5}[layer_axis]
    out_of_plane_shear = [s for s in SHEAR if s != in_plane_shear]
    for d in [layer_axis] + out_of_plane_shear:        # series -> Reuss
        C_est[d, d] = C_r[d, d]
    return C_est


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # Smoke test: a single phase must have Voigt == Reuss == itself, zero gap.
    C = isotropic_stiffness(10.0, 0.3)
    Cv = voigt_bound([1.0], [C])
    Cr = reuss_bound([1.0], [C])
    print("1) single-phase identity:")
    print("   ||Voigt - C|| =", np.linalg.norm(Cv - C))
    print("   ||Reuss - C|| =", np.linalg.norm(Cr - C))
    print("   relative gap  =", relative_gap(Cv, Cr), "(==0 expected)")

    # Two-phase: Voigt >= Reuss channel-wise, and C must sit between them.
    C1 = isotropic_stiffness(10.0, 0.3)
    C2 = isotropic_stiffness(0.2, 0.3)
    f = [0.6, 0.4]
    Cv = voigt_bound(f, [C1, C2])
    Cr = reuss_bound(f, [C1, C2])
    print("\n2) two-phase high-contrast bounds:")
    print("   diag Voigt =", np.diag(Cv))
    print("   diag Reuss =", np.diag(Cr))
    print("   relative gap =", relative_gap(Cv, Cr))
    mid = 0.5 * (Cv + Cr)
    print("   midpoint contained:", containment(mid, Cv, Cr)["ok"])
