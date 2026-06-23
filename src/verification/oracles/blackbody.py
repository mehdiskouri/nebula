"""
Blackbody / soot-emission oracle for V3.3 (Tier 3) — the physically-correct flame color.

Phase-0 rendered fire with a hardcoded emissive `[1.0,0.55,0.12]` and a linear `(T-650)/450`
red→yellow lerp (blocker #2): not blackbody, no Wien hue shift, no T^4 intensity, no smoke.
A real flame glows by incandescence — soot particles radiate as a blackbody at the local
temperature — so its color follows the Planckian locus (dull red ~1000 K → orange ~1500 K →
yellow-white ~2000 K → white ~2500 K+), its radiant power scales as T^4 (Stefan–Boltzmann),
and the smoke's opacity follows Beer–Lambert in the soot column. This oracle supplies those
laws (and the CIE color via the compact Wyman-2013 analytic color-matching approximations, so
no data table is needed); the implementation `geometry/appearance.py` must match it.
"""
import numpy as np

H = 6.62607015e-34       # Planck [J·s]
C = 2.99792458e8         # speed of light [m/s]
KB = 1.380649e-23        # Boltzmann [J/K]
WIEN_B = 2.897771955e-3  # Wien displacement [m·K]
SIGMA_SB = 5.670374419e-8

# sRGB (linear) from CIE XYZ (D65)
_XYZ2RGB = np.array([[3.2406, -1.5372, -0.4986],
                     [-0.9689, 1.8758, 0.0415],
                     [0.0557, -0.2040, 1.0570]])


def planck_radiance(lam_m, T):
    """Spectral radiance B(λ,T) [W·sr^-1·m^-3] (Planck's law). λ in metres."""
    lam = np.asarray(lam_m, float)
    return (2.0 * H * C * C) / (lam ** 5) / (np.expm1(H * C / (lam * KB * T)))


def wien_peak_wavelength(T):
    """Peak-emission wavelength λ_max = b/T [m]."""
    return WIEN_B / T


def stefan_boltzmann(T):
    """Total radiant exitance ∝ T^4 [W/m^2]."""
    return SIGMA_SB * np.asarray(T, float) ** 4


def _gauss_piecewise(x, mu, s1, s2):
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def cie_xyz_bar(lam_nm):
    """CIE 1931 2° color-matching functions (Wyman et al. 2013 multi-lobe analytic fit). λ in nm."""
    x = (1.056 * _gauss_piecewise(lam_nm, 599.8, 37.9, 31.0)
         + 0.362 * _gauss_piecewise(lam_nm, 442.0, 16.0, 26.7)
         - 0.065 * _gauss_piecewise(lam_nm, 501.1, 20.4, 26.2))
    y = (0.821 * _gauss_piecewise(lam_nm, 568.8, 46.9, 40.5)
         + 0.286 * _gauss_piecewise(lam_nm, 530.9, 16.3, 31.1))
    z = (1.217 * _gauss_piecewise(lam_nm, 437.0, 11.8, 36.0)
         + 0.681 * _gauss_piecewise(lam_nm, 459.0, 26.0, 13.8))
    return np.stack([x, y, z], axis=-1)


def blackbody_xyz(T, lam_nm=None):
    """Unnormalized CIE XYZ of a Planck emitter at temperature T (integrate B·CMF over visible)."""
    if lam_nm is None:
        lam_nm = np.linspace(360.0, 830.0, 471)
    B = planck_radiance(lam_nm * 1e-9, T)
    bar = cie_xyz_bar(lam_nm)
    return np.trapezoid(B[:, None] * bar, lam_nm, axis=0)


def blackbody_srgb(T, normalize="luminance"):
    """sRGB (0..1, gamma-encoded) chromaticity of a blackbody at T — the Planckian locus color."""
    xyz = blackbody_xyz(T)
    if normalize == "luminance":
        xyz = xyz / (xyz[1] + 1e-30)            # normalize Y → hue/chroma only (intensity via SB)
    rgb = _XYZ2RGB @ xyz
    rgb = np.clip(rgb, 0.0, None)
    if rgb.max() > 0:
        rgb = rgb / rgb.max()
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.clip(srgb, 0.0, 1.0)


def beer_lambert_transmittance(soot_column, kappa=1.0):
    """Smoke transmittance through a soot column: exp(−κ · ∫soot dl) ∈ (0,1]."""
    return np.exp(-kappa * np.asarray(soot_column, float))


if __name__ == "__main__":
    # 1) Wien: numeric peak of Planck matches b/T across the flame range.
    lam = np.linspace(2e-7, 1.2e-5, 40000)
    print("1) Wien peak wavelength (numeric vs b/T):")
    for T in (1000.0, 1500.0, 2000.0):
        num = lam[np.argmax(planck_radiance(lam, T))]
        print(f"   T={T:.0f}K  numeric {num*1e9:7.1f} nm  b/T {wien_peak_wavelength(T)*1e9:7.1f} nm")
        assert abs(num - wien_peak_wavelength(T)) / wien_peak_wavelength(T) < 0.01

    # 2) Stefan–Boltzmann: doubling T multiplies radiant power by 16 (2^4).
    r = stefan_boltzmann(2000.0) / stefan_boltzmann(1000.0)
    print(f"2) Stefan–Boltzmann power ratio (2000K/1000K) = {r:.2f}  (target 16)")
    assert abs(r - 16.0) < 1e-6

    # 3) Planckian locus: hot is bluer than cold (B/R rises with T); flame colors plausible.
    print("3) blackbody sRGB along the Planckian locus:")
    prev_br = -1
    for T in (1000.0, 1500.0, 2000.0, 3000.0, 6500.0):
        rgb = blackbody_srgb(T)
        br = (rgb[2] + 1e-6) / (rgb[0] + 1e-6)
        tag = "red" if T <= 1100 else ("orange" if T < 1800 else ("yellow-white" if T < 2600 else "white"))
        print(f"   T={T:5.0f}K -> sRGB {np.round(rgb,3)}  ({tag})  B/R={br:.3f}")
        assert br > prev_br - 1e-9; prev_br = br
        assert rgb[0] >= rgb[2] - 1e-6 or T >= 6000   # red-dominant until near-white
    assert blackbody_srgb(1000.0)[0] > blackbody_srgb(1000.0)[2]    # 1000K is red, not blue

    # 4) Beer–Lambert: opacity grows with soot column, transmittance in (0,1].
    t = beer_lambert_transmittance([0.0, 0.5, 2.0, 5.0], kappa=1.0)
    print(f"4) smoke transmittance vs soot column: {np.round(t,3)}")
    assert t[0] == 1.0 and np.all(np.diff(t) < 0) and t[-1] > 0
    print("\nblackbody oracle self-checks passed.")
