"""
Derived appearance — texture/color as a SIMULATION OUTPUT (ARCHITECTURE Part I; V3.3/V3.7).

Phase-0 faked appearance with three hardcoded lerps: a constant emissive `[1.0,0.55,0.12]`,
a linear `(T-650)/450` red→yellow flame ramp, and a flat char blend (blockers #2, #6). This
module derives appearance from physics instead:

  - EMISSION (the flame / embers) is BLACKBODY incandescence: color follows the Planckian locus
    (Wien hue shift red→orange→yellow→white) and intensity scales as T^4 (Stefan–Boltzmann). The
    color table is ported from the V3.3 oracle (`blackbody.py`) — the V3.3 notebook checks parity.
  - SMOKE opacity is Beer–Lambert in the soot column.
  - SURFACE reflectance is derived from state: char (chi) darkens albedo toward soot-black and
    roughens it; moisture darkens (wetting); fresh wood keeps its layer color. Outputs PBR
    (albedo, roughness, emission) for a real renderer, not a flat vertex tint.

Determinism: a fixed blackbody LUT (no per-call CIE integral), pure numpy.
"""
import numpy as np

# physical constants (match blackbody.py)
_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                     [-0.9689, 1.8758, 0.0415],
                     [0.0557, -0.2040, 1.0570]])

# reference reflectances (sRGB 0..1) — char/soot and ash, the burnt-surface endpoints
CHAR_ALBEDO = np.array([0.04, 0.035, 0.03])     # soot-black (measured char ~0.03–0.05)
ASH_ALBEDO = np.array([0.32, 0.30, 0.28])       # pale grey ash
WET_DARKEN = 0.55                                # albedo multiplier when fully wet


def _planck(lam_m, T):
    return (2.0 * _H * _C * _C) / (lam_m ** 5) / np.expm1(_H * _C / (lam_m * _KB * T))


def _gp(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def _cmf(lam_nm):
    x = (1.056 * _gp(lam_nm, 599.8, 37.9, 31.0) + 0.362 * _gp(lam_nm, 442.0, 16.0, 26.7)
         - 0.065 * _gp(lam_nm, 501.1, 20.4, 26.2))
    y = 0.821 * _gp(lam_nm, 568.8, 46.9, 40.5) + 0.286 * _gp(lam_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _gp(lam_nm, 437.0, 11.8, 36.0) + 0.681 * _gp(lam_nm, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


def _build_lut(Tlo=600.0, Thi=6000.0, n=128):
    """Blackbody sRGB chromaticity LUT over T (luminance-normalized, gamma-encoded)."""
    lam = np.linspace(360.0, 830.0, 300)
    bar = _cmf(lam)
    Ts = np.linspace(Tlo, Thi, n)
    cols = np.zeros((n, 3))
    for i, T in enumerate(Ts):
        B = _planck(lam * 1e-9, T)
        xyz = np.trapezoid(B[:, None] * bar, lam, axis=0)
        xyz = xyz / (xyz[1] + 1e-30)
        rgb = np.clip(_XYZ2RGB @ xyz, 0.0, None)
        rgb = rgb / (rgb.max() + 1e-30)
        cols[i] = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return Ts, np.clip(cols, 0.0, 1.0)


_LUT_T, _LUT_RGB = _build_lut()


def blackbody_rgb(T):
    """sRGB (0..1) chromaticity of a blackbody at temperature(s) T via the LUT (Planckian locus)."""
    T = np.atleast_1d(np.asarray(T, float))
    out = np.empty(T.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(T, _LUT_T, _LUT_RGB[:, c])
    return out


def emission_intensity(T, T_on=800.0, T_full=1600.0):
    """Normalized radiant emission strength in [0,1]: ∝ (T^4 − T_on^4), ramped to 1 at T_full."""
    T = np.asarray(T, float)
    num = np.clip(T, 0, None) ** 4 - T_on ** 4
    den = T_full ** 4 - T_on ** 4
    return np.clip(num / den, 0.0, 1.0)


def ember_emission(T, chi=None):
    """Emission RGB (HDR, may exceed 1) of glowing/charred matter: blackbody color × T^4 intensity,
    gated by char fraction chi (only charring matter glows; fresh wood doesn't emit)."""
    col = blackbody_rgb(T)
    inten = emission_intensity(T)
    if chi is not None:
        inten = inten * np.clip(np.asarray(chi, float), 0.0, 1.0)
    return col * inten[..., None]


def surface_appearance(base_rgb, T=None, chi=None, soot=None, moisture=None):
    """Derive PBR (albedo, roughness, emission) from the wood base color + simulation state.

    base_rgb: (...,3) wood-layer color in 0..1. Returns dict of (...,3)/(...,) arrays.
    """
    base = np.asarray(base_rgb, float)
    alb = base.copy()
    rough = np.full(base.shape[:-1], 0.7)         # bark is fairly matte
    if moisture is not None:
        w = np.clip(np.asarray(moisture), 0, 1)[..., None]
        alb = alb * (1.0 - w * (1.0 - WET_DARKEN))   # wetting darkens
        rough = rough * (1.0 - 0.5 * w[..., 0])      # wet → smoother/specular
    if chi is not None:
        c = np.clip(np.asarray(chi), 0, 1)[..., None]
        alb = (1.0 - c) * alb + c * CHAR_ALBEDO       # char darkens to soot-black
        rough = np.maximum(rough, 0.6 + 0.4 * c[..., 0])  # char is rough/matte
    if soot is not None:
        d = np.clip(np.asarray(soot), 0, 1)[..., None]
        alb = (1.0 - d) * alb + d * CHAR_ALBEDO       # soot deposit
    emis = np.zeros_like(alb)
    if T is not None:
        emis = ember_emission(T, chi)
    return {"albedo": np.clip(alb, 0, 1), "roughness": np.clip(rough, 0, 1), "emission": emis}


def smoke_alpha(soot_column, kappa=1.0):
    """Smoke opacity (1 − transmittance) from the integrated soot column (Beer–Lambert)."""
    return 1.0 - np.exp(-kappa * np.clip(np.asarray(soot_column, float), 0, None))


if __name__ == "__main__":
    np.seterr(all="ignore")
    # 1) blackbody flame colors along the Planckian locus (the real fire palette).
    print("1) blackbody flame color (LUT):")
    for T in (900.0, 1200.0, 1600.0, 2200.0):
        print(f"   T={T:.0f}K -> sRGB {np.round(blackbody_rgb(T)[0], 3)}  "
              f"emission×{emission_intensity(T):.2f}")
    assert blackbody_rgb(900.0)[0, 0] > blackbody_rgb(900.0)[0, 2]      # 900K red
    assert blackbody_rgb(2200.0)[0, 2] > blackbody_rgb(1200.0)[0, 2]    # hotter is bluer

    # 2) emission grows as T^4 above threshold; cold wood does not emit.
    assert emission_intensity(700.0) == 0.0 and emission_intensity(1600.0) == 1.0
    r = (emission_intensity(1500.0, T_on=0, T_full=3000) /
         emission_intensity(750.0, T_on=0, T_full=3000))
    print(f"2) emission(1500)/emission(750) = {r:.1f} (T^4 → 16)"); assert abs(r - 16) < 1e-6

    # 3) char darkens & roughens; wetting darkens; only hot char emits.
    bark = np.array([0.40, 0.26, 0.13])
    fresh = surface_appearance(bark)
    charred = surface_appearance(bark, T=1400.0, chi=0.9)
    wet = surface_appearance(bark, moisture=1.0)
    print(f"3) bark albedo {np.round(fresh['albedo'],3)} -> charred {np.round(charred['albedo'],3)} "
          f"(emission {np.round(charred['emission'],2)}); wet {np.round(wet['albedo'],3)}")
    assert charred["albedo"].sum() < fresh["albedo"].sum()       # char darker
    assert charred["roughness"] > fresh["roughness"]              # char rougher
    assert wet["albedo"].sum() < fresh["albedo"].sum()           # wet darker
    assert charred["emission"].sum() > 0 and fresh["emission"].sum() == 0

    # 4) smoke opacity grows with soot column (Beer–Lambert).
    a = smoke_alpha([0.0, 0.5, 2.0, 5.0])
    print(f"4) smoke opacity vs soot column: {np.round(a,3)}")
    assert a[0] == 0 and np.all(np.diff(a) > 0) and a[-1] < 1
    print("\nappearance self-checks passed.")
