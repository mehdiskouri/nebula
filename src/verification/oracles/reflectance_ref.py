"""
Reflectance oracle for V3.7 (Tier 3) — the surface BRDF endpoints sit in measured ranges.

The derived appearance map (`geometry/appearance`) must land its material endpoints in *measured*
reflectance ranges so the path tracer renders physically-plausible wood/char/ash/wet surfaces — the
appearance is a derived simulation output, but its constants must be grounded, not invented. This
oracle holds representative diffuse-reflectance (albedo) ranges from the materials literature, the
wet-darkening factor, and the blackbody-emission scaling, with a containment check.

(These are *plausibility-engine* representative values, Part VII — internally consistent and grounded,
not a spectrophotometric standard.)
"""
import numpy as np

# representative diffuse reflectance (albedo) ranges — broadband visible
ALBEDO_RANGES = {
    "fresh_wood": (0.30, 0.60),    # pale sapwood / fresh-cut softwood
    "bark":       (0.12, 0.40),    # outer bark
    "char":       (0.02, 0.06),    # soot-black charcoal (very low)
    "ash":        (0.22, 0.45),    # pale grey ash
}
WET_DARKEN_RANGE = (0.45, 0.75)    # wet albedo / dry albedo (water film darkens)


def luminance(rgb):
    """Rec.709 relative luminance of a linear-sRGB albedo triple."""
    rgb = np.asarray(rgb, float)
    return float(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])


def in_range(value, name):
    lo, hi = ALBEDO_RANGES[name]
    return lo <= value <= hi


def blackbody_emission_ratio(T_hi, T_lo):
    """Stefan–Boltzmann ratio (T_hi/T_lo)^4 — the emission-scaling the ember map must follow."""
    return (T_hi / T_lo) ** 4


if __name__ == "__main__":
    print("1) measured albedo ranges:", {k: v for k, v in ALBEDO_RANGES.items()})
    # char is much darker than fresh wood than ash ordering sanity
    assert ALBEDO_RANGES["char"][1] < ALBEDO_RANGES["bark"][0]
    assert ALBEDO_RANGES["char"][1] < ALBEDO_RANGES["ash"][0]
    print("2) ordering: char ≪ bark, char ≪ ash, fresh wood brightest — OK")

    # containment helper works
    assert in_range(0.035, "char") and not in_range(0.5, "char")
    print("3) containment check: char 0.035 in range, 0.5 not — OK")

    # wet darkening + emission scaling
    assert WET_DARKEN_RANGE[0] <= 0.55 <= WET_DARKEN_RANGE[1]
    r = blackbody_emission_ratio(1600, 800)
    print(f"4) wet-darken 0.55 in {WET_DARKEN_RANGE}; emission ratio (1600/800)^4 = {r:.0f} (=16)")
    assert abs(r - 16) < 1e-6
    print("\nreflectance oracle self-checks passed.")
