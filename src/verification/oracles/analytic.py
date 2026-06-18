"""
Closed-form laminate effective stiffness (protocol §7 analytic library).

Two jobs:
  1. Oracle-of-the-oracle: the DNS micro-solver (dns_elasticity_3d.py) is trusted
     only after it reproduces this closed form on clean layered cells.
  2. Layered-exactness reference for V0.1: a stack of isotropic layers is exactly
     transversely isotropic, and its effective tensor is given in closed form by
     Backus (1962) averaging. The across-layer normal and the shear channels each
     coincide *exactly* with a Voigt or Reuss bound (ARCHITECTURE.md §III.4).

Backus averaging for isotropic layers stacked along a special axis, with per-layer
Lame constants (lambda, mu) and volume fractions f (angle brackets = f-weighted mean,
den = lambda + 2 mu):

    C33 = 1 / <1/den>                          (across-layer normal  = Reuss-exact)
    C13 = <lambda/den> / <1/den>
    C11 = <4 mu (lambda+mu)/den> + <lambda/den>^2 / <1/den>   (in-plane normal)
    C12 = C11 - 2 C66
    C44 = 1 / <1/mu>                           (out-of-plane shear   = Reuss-exact)
    C66 = <mu>                                 (in-plane shear       = Voigt-exact)

Voigt notation [11,22,33,23,13,12], engineering shear; matches homogenization.py.
"""
import numpy as np


def _lame(E, nu):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def _shear_indices(axis):
    """(in_plane_shear_voigt_index, [out_of_plane_shear_indices]) for a special axis."""
    # Voigt shear indices: 3<->(y,z), 4<->(x,z), 5<->(x,y).
    in_plane = {0: 3, 1: 4, 2: 5}[axis]          # shear in the plane perpendicular to axis
    out_of_plane = [i for i in (3, 4, 5) if i != in_plane]
    return in_plane, out_of_plane


def layered_exact_channels(axis):
    """Voigt channels whose bound is EXACTLY achieved by a layered cell.

    Returns (reuss_exact, voigt_exact) lists of Voigt indices. The across-layer
    normal and both out-of-plane shears coincide with the Reuss bound; the
    in-plane shear coincides with the Voigt bound. (The two in-plane normals are
    NOT bound-exact — they carry the Backus correction — so they are excluded
    from the exactness check and covered by the containment theorem instead.)
    """
    s_in, s_out = _shear_indices(axis)
    reuss_exact = [axis] + list(s_out)
    voigt_exact = [s_in]
    return reuss_exact, voigt_exact


def laminate_stiffness(fractions, moduli, nus, axis=2):
    """Exact effective 6x6 stiffness of isotropic layers stacked along `axis`."""
    f = np.asarray(fractions, float)
    f = f / f.sum()
    lam = np.array([_lame(E, nu)[0] for E, nu in zip(moduli, nus)])
    mu = np.array([_lame(E, nu)[1] for E, nu in zip(moduli, nus)])
    den = lam + 2.0 * mu

    inv_den = np.sum(f / den)
    lam_den = np.sum(f * lam / den)
    C33 = 1.0 / inv_den
    C13 = lam_den / inv_den
    C11 = np.sum(f * 4.0 * mu * (lam + mu) / den) + lam_den ** 2 / inv_den
    C44 = 1.0 / np.sum(f / mu)
    C66 = np.sum(f * mu)
    C12 = C11 - 2.0 * C66

    # in-plane axes (the two that are not `axis`)
    p, q = [i for i in (0, 1, 2) if i != axis]
    C = np.zeros((6, 6))
    C[axis, axis] = C33
    C[p, p] = C[q, q] = C11
    C[p, q] = C[q, p] = C12
    C[axis, p] = C[p, axis] = C[axis, q] = C[q, axis] = C13
    s_in, s_out = _shear_indices(axis)
    C[s_in, s_in] = C66
    for s in s_out:
        C[s, s] = C44
    return C


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # A two-layer stack: across-layer normal must be Reuss-exact, in-plane shear Voigt-exact.
    from homogenization import voigt_bound, reuss_bound, isotropic_stiffness

    f = [0.5, 0.5]
    moduli = [10.0, 1.0]
    nus = [0.3, 0.3]
    axis = 2
    C = laminate_stiffness(f, moduli, nus, axis)
    Cp = [isotropic_stiffness(E, nu) for E, nu in zip(moduli, nus)]
    Cv = voigt_bound(f, Cp)
    Cr = reuss_bound(f, Cp)
    print("1) Backus laminate (axis=2):")
    print(C)
    print("\n2) across-layer normal C33: laminate=%.4f  Reuss=%.4f  (should match)"
          % (C[2, 2], Cr[2, 2]))
    print("   in-plane shear  C[5,5]: laminate=%.4f  Voigt=%.4f  (should match)"
          % (C[5, 5], Cv[5, 5]))
    print("   out-plane shear C[3,3]: laminate=%.4f  Reuss=%.4f  (should match)"
          % (C[3, 3], Cr[3, 3]))
    print("   bracketing of in-plane normal C11: Reuss=%.3f <= %.3f <= Voigt=%.3f"
          % (Cr[0, 0], C[0, 0], Cv[0, 0]))
