"""
Homogenization proxy (Decision #15; ARCHITECTURE §III.4). Verified by V0.1 (PASS):
DNS within the gap for 100% of cells; layered principal-direction residual < 1%; gap < 30%
for >80% of low-contrast cells.

A heterogeneous cell collapses to one effective 6x6 stiffness tensor whose true value is
BRACKETED by the Voigt and Reuss bounds in every direction; the width of that bracket (the
Voigt-Reuss gap) is the per-cell trust scalar driving refinement / LOD / surrogate trust.
  - Voigt bound = volume-weighted ARITHMETIC mean of stiffness  (uniform-strain upper bound).
  - Reuss bound = volume-weighted HARMONIC   mean of stiffness  (uniform-stress lower bound).
  - For any macroscopic strain e:  e:C_Reuss:e <= e:C_eff:e <= e:C_Voigt:e.

Ported verbatim from the frozen oracle src/verification/oracles/homogenization.py. Voigt
notation throughout: stress/strain are length-6 vectors [11,22,33,23,13,12], engineering shear.
"""
import numpy as np

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
    """Sum_i f_i C_i — arithmetic mean of stiffness (uniform-strain upper bound)."""
    fractions = np.asarray(fractions, float)
    return np.tensordot(fractions, np.asarray(C_phases, float), axes=(0, 0))


def reuss_bound(fractions, C_phases):
    """(Sum_i f_i C_i^-1)^-1 — harmonic mean of stiffness (uniform-stress lower bound)."""
    fractions = np.asarray(fractions, float)
    S = np.stack([np.linalg.inv(C) for C in C_phases])
    S_eff = np.tensordot(fractions, S, axes=(0, 0))
    return np.linalg.inv(S_eff)


def relative_gap(C_voigt, C_reuss):
    """Per-direction (Voigt-Reuss)/mean for the 6 Voigt channels — the trust scalar (length 6)."""
    dv = np.diag(C_voigt)
    dr = np.diag(C_reuss)
    mean = 0.5 * (dv + dr)
    return (dv - dr) / mean


def containment(C_eff, C_voigt, C_reuss, tol=1e-9):
    """Check the bracketing theorem C_Reuss <= C_eff <= C_Voigt (per-direction + PSD)."""
    dv = np.diag(C_voigt); dr = np.diag(C_reuss); de = np.diag(C_eff)
    gap = dv - dr
    signed = np.where(gap > 0, (de - dr) / np.where(gap == 0, 1.0, gap), 0.0)
    per_dir_ok = (signed >= -tol) & (signed <= 1.0 + tol)
    psd_lower = float(np.linalg.eigvalsh(C_eff - C_reuss).min())
    psd_upper = float(np.linalg.eigvalsh(C_voigt - C_eff).min())
    psd_ok = (psd_lower >= -tol) and (psd_upper >= -tol)
    return {"signed": signed, "per_dir_ok": per_dir_ok, "psd_lower": psd_lower,
            "psd_upper": psd_upper, "psd_ok": psd_ok,
            "ok": bool(per_dir_ok.all() and psd_ok)}


def directional_estimate(fractions, C_phases, layer_axis):
    """The orthotropic proxy Nebula ships for a LAYERED cell (series vs parallel).

    Modes loading the layers in SERIES (the across-layer normal + the two out-of-plane shears)
    follow Reuss; modes loading them in PARALLEL (the two in-plane normals + the in-plane shear)
    follow Voigt. For clean layered media these coincide EXACTLY with the bounds in the principal
    directions (V0.1 layered-exactness; the gap is the error bar on the in-plane Backus correction).
    """
    C_v = voigt_bound(fractions, C_phases)
    C_r = reuss_bound(fractions, C_phases)
    C_est = C_v.copy()
    in_plane_shear = {0: 3, 1: 4, 2: 5}[layer_axis]
    out_of_plane_shear = [s for s in SHEAR if s != in_plane_shear]
    for d in [layer_axis] + out_of_plane_shear:
        C_est[d, d] = C_r[d, d]
    return C_est


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    C = isotropic_stiffness(10.0, 0.3)
    Cv = voigt_bound([1.0], [C]); Cr = reuss_bound([1.0], [C])
    print("1) single-phase identity: ||V-C||", np.linalg.norm(Cv - C),
          "gap", relative_gap(Cv, Cr))
    C1 = isotropic_stiffness(10.0, 0.3); C2 = isotropic_stiffness(0.2, 0.3)
    Cv = voigt_bound([0.6, 0.4], [C1, C2]); Cr = reuss_bound([0.6, 0.4], [C1, C2])
    print("2) two-phase gap", relative_gap(Cv, Cr),
          " midpoint contained:", containment(0.5 * (Cv + Cr), Cv, Cr)["ok"])
