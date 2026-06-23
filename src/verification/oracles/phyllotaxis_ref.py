"""
Phyllotaxis oracle for V3.4 (Tier 3) — the golden-angle reference for canopy generation.

Phase-0 has NO foliage: the tree is a bare skeleton, so a "tree on fire" has nothing to
flash. The canopy operator (operators/canopy.py) deposits leaves on terminal twigs. This
oracle is the independent reference for the *arrangement* claim: real leaves around a shoot
are placed by spiral phyllotaxis at the golden angle ψ = 360°·(2 − φ) = 137.507…°, where
φ = (1+√5)/2. Vogel's model places primordium n at azimuth n·ψ and radius c·√n, giving the
even, non-overlapping packing seen in sunflowers/conifers. Deviating from ψ clumps the
leaves (poor light capture and an unrealistic canopy); matching it is what reads as foliage.

The oracle provides: the exact golden angle, the Vogel spiral, a packing-uniformity score
(nearest-neighbour distance regularity vs. a non-golden control), and a divergence-angle
estimator for a generated leaf set. No tree code here — obtained a different way.
"""
import numpy as np

PHI = (1.0 + np.sqrt(5.0)) / 2.0
GOLDEN_ANGLE_DEG = 360.0 * (2.0 - PHI)           # 137.50776…°
GOLDEN_ANGLE_RAD = np.deg2rad(GOLDEN_ANGLE_DEG)


def vogel_spiral(n, c=1.0, angle_rad=GOLDEN_ANGLE_RAD):
    """n points by Vogel's model: azimuth k·angle, radius c·√k. Returns (n,2) xy."""
    k = np.arange(n)
    theta = k * angle_rad
    r = c * np.sqrt(k)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def packing_uniformity(pts):
    """Coefficient of variation of nearest-neighbour distances (lower = more even packing).

    The golden angle minimises this — primordia are maximally evenly spread. A rational-
    fraction angle lines points into spokes with large gaps → high CV.
    """
    from scipy.spatial import cKDTree
    if len(pts) < 3:
        return np.nan
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    return float(nn.std() / (nn.mean() + 1e-30))


def divergence_angles(azimuths_rad):
    """Consecutive azimuthal increments (mod 2π → principal value in (−π,π]) along a shoot.

    For spiral phyllotaxis these cluster at the golden angle (≡ −222.49°, principal value
    +137.5° taken as |Δ| folded into [0,180]).
    """
    a = np.asarray(azimuths_rad, float)
    d = np.diff(a)
    d = (d + np.pi) % (2 * np.pi) - np.pi        # wrap to (−π, π]
    return np.abs(np.rad2deg(d))


def folded_to_180(angle_deg):
    """Fold an angle to [0,180] (a divergence of 137.5° and 222.5° are the same spacing)."""
    a = np.asarray(angle_deg, float) % 360.0
    return np.minimum(a, 360.0 - a)


def leaf_area_index(leaf_area_total, crown_footprint_area):
    """LAI = one-sided leaf area per unit ground the crown covers (broadleaf canopies ~2–8)."""
    return float(leaf_area_total / max(crown_footprint_area, 1e-12))


if __name__ == "__main__":
    print(f"golden angle = {GOLDEN_ANGLE_DEG:.5f}°  (φ = {PHI:.6f})")

    # 1) the golden angle packs more evenly than nearby non-golden controls.
    g = packing_uniformity(vogel_spiral(600, angle_rad=GOLDEN_ANGLE_RAD))
    controls = [packing_uniformity(vogel_spiral(600, angle_rad=np.deg2rad(a)))
                for a in (130.0, 137.0, 138.0, 145.0, 120.0)]
    print(f"1) packing CV: golden {g:.3f}  vs controls {[round(c,3) for c in controls]}")
    assert g < min(controls), "golden angle should pack most evenly"

    # 2) a Vogel shoot recovers the golden divergence angle from its azimuth sequence.
    k = np.arange(40)
    div = divergence_angles(k * GOLDEN_ANGLE_RAD)
    print(f"2) recovered divergence angle: median {np.median(folded_to_180(div)):.3f}°  "
          f"(target {folded_to_180(GOLDEN_ANGLE_DEG):.3f}°)")
    assert abs(np.median(folded_to_180(div)) - folded_to_180(GOLDEN_ANGLE_DEG)) < 1e-6

    # 3) LAI sanity.
    lai = leaf_area_index(leaf_area_total=120.0, crown_footprint_area=30.0)
    print(f"3) LAI example = {lai:.2f}  (broadleaf 2–8)")
    assert 2.0 <= lai <= 8.0
    print("\nphyllotaxis oracle self-checks passed.")
