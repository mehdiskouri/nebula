"""
Bark & root-morphology oracle for V3.8 (Tier 3) — "a tree being a tree", derived not authored.

Two derived-from-mechanism morphology claims, with their independent scaling laws:

1. ROOT FLARE / BUTTRESS. The basal trunk–root junction carries the entire above-ground load into
   the roots, so it must be the thickest cross-section — a *derived* consequence of the pipe model
   (da Vinci / Murray: r_parent^p = Σ r_child^p) plus Wolff's law (stress-driven thickening, the
   V1.7 mechanism). The flare is therefore not authored: the base is enlarged because the section
   modulus must carry the bending moment of the crown. Reference: the base radius exceeds the mid-
   trunk radius (pipe-model summation over trunk+roots), and the load-bearing section grows with the
   supported mass.

2. BARK FISSURE RELIEF. Secondary (radial) growth stretches the rigid outer bark over an expanding
   circumference; the bark fails in tension into vertical fissures whose spacing/depth scale with the
   trunk diameter and growth increment (a stretching-instability / drying-crack thickness law:
   characteristic crack spacing ∝ layer thickness; deeper/wider fissures on older, faster-grown
   trunks). Reference: fissure depth increases monotonically with trunk radius and with the radial
   growth increment, and the fissures are oriented along the grain (vertical).

These are the falsifiable scalings the implementation (`space_colonization` flare, `bark_texture`
relief) must reproduce — both DERIVED from the growth/load state, then exported as maps.
"""
import numpy as np


def pipe_model_radius(child_radii, p=2.3):
    """Pipe-model parent radius from children: r = (Σ r_child^p)^(1/p) (p≈2.0–2.5)."""
    child_radii = np.asarray(child_radii, float)
    return float((np.sum(child_radii ** p)) ** (1.0 / p))


def basal_flare_ratio(trunk_radius, root_radii, p=2.3):
    """Base/trunk radius ratio: the basal node supports BOTH the trunk and all major roots, so by the
    pipe model its radius = (r_trunk^p + Σ r_root^p)^{1/p} > r_trunk. Returns that ratio (>1)."""
    base = (trunk_radius ** p + np.sum(np.asarray(root_radii, float) ** p)) ** (1.0 / p)
    return base / trunk_radius


def section_modulus_demand(crown_mass, height, lever):
    """Bending moment the base must carry (∝ crown_mass·g·lever); the basal section grows to meet it.
    Returns a relative demand (monotone in crown mass and lever) — the Wolff driver for the flare."""
    return float(crown_mass * 9.81 * lever)


def fissure_depth(trunk_radius, growth_increment, bark_thickness, c_r=3.0, c_g=50.0):
    """Bark-fissure depth from radial-growth tension: deeper on larger (more accumulated hoop strain)
    and faster-grown trunks. Driven by `c_r·r + c_g·Δr`, saturating at the bark thickness — so depth
    grows monotonically with both trunk radius and the radial growth increment."""
    drive = c_r * trunk_radius + c_g * growth_increment
    return float(np.clip(bark_thickness * (1.0 - np.exp(-drive)), 0, bark_thickness))


def fissure_spacing(bark_thickness, contrast=1.0):
    """Characteristic fissure spacing ∝ layer thickness (the mud-crack / stretching thickness law)."""
    return float(2.0 * bark_thickness / max(contrast, 1e-6))


if __name__ == "__main__":
    # 1) root flare: a trunk supported by several major roots has an enlarged base (pipe model).
    ratio = basal_flare_ratio(0.20, [0.14, 0.13, 0.11, 0.10], p=2.3)
    print(f"1) basal flare ratio (base/trunk) = {ratio:.3f}  (>1 ⇒ derived flare)")
    assert ratio > 1.2

    # 2) the flare grows with supported load (section-modulus demand monotone in crown mass).
    d = [section_modulus_demand(m, 8.0, 1.5) for m in (50, 100, 200, 400)]
    print(f"2) basal load demand vs crown mass: {[round(x) for x in d]} (increasing)")
    assert np.all(np.diff(d) > 0)

    # 3) bark fissure depth increases with trunk radius and with growth increment (monotone).
    dr = [fissure_depth(r, 0.01, 0.02) for r in (0.05, 0.1, 0.2, 0.4)]
    dg = [fissure_depth(0.2, g, 0.02) for g in (0.002, 0.006, 0.012, 0.02)]
    print(f"3) fissure depth vs trunk radius {np.round(dr,4)}; vs growth incr {np.round(dg,4)} (both ↑)")
    assert np.all(np.diff(dr) >= 0) and np.all(np.diff(dg) >= 0) and dr[-1] > dr[0] and dg[-1] > dg[0]

    # 4) fissure spacing scales with bark thickness (the thickness law).
    s = [fissure_spacing(b) for b in (0.005, 0.01, 0.02, 0.04)]
    print(f"4) fissure spacing vs bark thickness: {np.round(s,3)} (∝ thickness)")
    assert np.allclose(np.diff(s) / np.diff([0.005, 0.01, 0.02, 0.04]), 2.0)
    print("\nbark/root-morphology oracle self-checks passed.")
