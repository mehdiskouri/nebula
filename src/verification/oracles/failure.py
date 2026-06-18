"""
Distance-to-failure oracle for V0.2 (the criticality coincidence; protocol §7).

Given a solved cell's `Localization` (from dns_elasticity_3d.effective_stiffness with
return_localization=True), reconstruct the local stress field under ANY applied macro
load and measure how close the load-bearing material is to failure. Two measures are
provided (per the V0.2 decision: track both, threshold on stress):
  - peak von-Mises stress in the load-bearing phase vs a cohesive strength  -> the
    "distance-to-failure" that the criticality-onset threshold is read from;
  - total stored elastic strain energy density -> supporting evidence.

The local stress comes for free from the 6 unit-strain solves already done by the DNS
oracle: local strain = eps_loc @ macro_strain, local stress = C_phase : local strain.
Pure numpy; Voigt convention [11,22,33,23,13,12], engineering shear.
"""
import numpy as np


def local_strain(loc, macro_strain):
    """Per-element centroid strain (n^3, 6) under a macroscopic Voigt strain vector."""
    return loc.eps_loc @ np.asarray(macro_strain, float)


def local_stress(loc, macro_strain):
    """Per-element Voigt stress (n^3, 6) under a macroscopic Voigt strain vector."""
    eps = local_strain(loc, macro_strain)                      # (n^3, 6)
    C_stack = np.stack(loc.C_phases)[loc.phases]               # (n^3, 6, 6)
    return np.einsum("eij,ej->ei", C_stack, eps)


def von_mises(sigma):
    """von-Mises stress from Voigt stress vectors (..., 6), engineering-shear aware."""
    s = np.asarray(sigma, float)
    s11, s22, s33, s23, s13, s12 = (s[..., i] for i in range(6))
    return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
                   + 3.0 * (s23 ** 2 + s13 ** 2 + s12 ** 2))


def peak_stress(loc, macro_strain, phase=0):
    """Max von-Mises stress over elements of the load-bearing `phase` (default wood)."""
    vm = von_mises(local_stress(loc, macro_strain))            # (n^3,)
    mask = loc.phases == phase
    return float(vm[mask].max()) if mask.any() else float(vm.max())


def stored_energy(loc, macro_strain):
    """Total elastic strain-energy density (1/2V) sum_e eps_e . sigma_e."""
    eps = local_strain(loc, macro_strain)
    sig = local_stress(loc, macro_strain)
    n_el = loc.eps_loc.shape[0]
    return float(0.5 * np.einsum("ei,ei->", eps, sig) / n_el)


def distance_to_failure(peak, strength):
    """Normalized distance to first-failure: 1 - peak/strength (<=0 == critical)."""
    return 1.0 - peak / strength


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import numpy as np
    from dns_elasticity_3d import effective_stiffness
    from homogenization import isotropic_stiffness
    import cells
    np.set_printoptions(precision=4, suppress=True)

    # 1) homogeneous cell under uniaxial strain -> uniform, analytic von-Mises.
    n, E, nu, eps = 12, 10.0, 0.3, 1e-3
    grid = np.zeros((n, n, n), dtype=np.int64)
    _, loc = effective_stiffness(grid, [(E, nu)], return_localization=True)
    e = np.array([eps, 0, 0, 0, 0, 0.0])
    sig = isotropic_stiffness(E, nu) @ e
    vm_true = von_mises(sig)
    vm_dns = peak_stress(loc, e, phase=0)
    print("1) homogeneous uniaxial strain:")
    print(f"   peak von-Mises  dns={vm_dns:.6f}  analytic={vm_true:.6f}  "
          f"rel err={abs(vm_dns-vm_true)/vm_true:.2e}")
    U_true = 0.5 * e @ sig
    print(f"   stored energy   dns={stored_energy(loc, e):.6e}  analytic={U_true:.6e}")

    # 2) char wedge: peak wood stress rises as the wedge deepens.
    print("2) char wedge — peak wood stress vs depth (uniaxial across wedge):")
    for depth in (0.0, 0.3, 0.6, 0.9):
        c = cells.char_wedge_cell(n=16, depth=depth, contrast=60.0)
        _, loc = effective_stiffness(c.grid, c.materials, return_localization=True)
        p = peak_stress(loc, np.array([1e-3, 0, 0, 0, 0, 0.0]), phase=0)
        print(f"   depth={depth:.1f}  char_frac={c.meta['char_fraction']:.3f}  "
              f"peak_wood_vm={p:.5f}")
