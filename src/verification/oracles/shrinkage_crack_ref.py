"""
Shrinkage-crack oracle for V3.6 (Tier 3) — the char "alligator" crackle, derived not authored.

When wood chars, the char layer loses mass and desiccates: it shrinks in-plane while bonded to the
unburnt wood beneath, which loads it in tension until it fractures into the characteristic polygonal
"alligator skin" network. The governing scaling is the **thickness law** shared by mud cracks, paint
crackle, and thermal-char cracking: the characteristic crack SPACING (polygon cell size) is set by the
shrinking layer's THICKNESS — thicker char → larger scales — `λ ≈ c · h_char` (the stress-relaxation
length over which the substrate re-anchors the layer). So the crack network's cell size is a DERIVED
function of the local char depth, and the crack depth grows with the char fraction χ.

This module gives those falsifiable scalings (and a cell-size estimator from a crack field) that the
implementation (`geometry/char_texture`) must reproduce — then the pattern is exported as a map.
"""
import numpy as np


def crack_spacing(char_depth, c=6.0):
    """Characteristic alligator-cell spacing ∝ char-layer thickness (the mud-crack thickness law)."""
    return c * np.asarray(char_depth, float)


def crack_cell_area(char_depth, c=6.0):
    """Polygon cell area ∝ spacing² ∝ char_depth² (so crack density ∝ 1/depth²)."""
    return crack_spacing(char_depth, c) ** 2


def crack_depth(chi, char_thickness, k=1.0):
    """Crack depth grows with char fraction χ (more charred → deeper, fully-developed cracks)."""
    return float(np.clip(k * char_thickness * np.clip(chi, 0, 1), 0, char_thickness))


def measure_cell_size(crack_field, spacing_px=1.0):
    """Dominant cell size of a 2-D crack field via its radial power spectrum peak (independent of the
    generator). Returns the wavelength [world units] of the strongest periodicity — the polygon size."""
    f = np.asarray(crack_field, float)
    f = f - f.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    ny, nx = f.shape
    yy, xx = np.mgrid[:ny, :nx]
    r = np.hypot(yy - ny / 2, xx - nx / 2).astype(int)
    radial = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    radial[0] = 0.0
    k_peak = max(int(np.argmax(radial[1:len(radial) // 2])) + 1, 1)
    return (nx * spacing_px) / k_peak          # wavelength = domain / peak frequency


if __name__ == "__main__":
    # 1) crack spacing scales linearly with char depth (the thickness law).
    h = np.array([0.002, 0.004, 0.008, 0.016])
    s = crack_spacing(h)
    print(f"1) crack spacing vs char depth: {np.round(s,4)}  (∝ depth, slope {np.polyfit(h,s,1)[0]:.1f})")
    assert np.allclose(s / h, s[0] / h[0]) and np.all(np.diff(s) > 0)

    # 2) cell area ∝ depth² (crack density falls as 1/depth²).
    a = crack_cell_area(h)
    print(f"2) cell area vs depth: {np.round(a,5)}  (exponent {np.polyfit(np.log(h),np.log(a),1)[0]:.2f} ≈ 2)")
    assert abs(np.polyfit(np.log(h), np.log(a), 1)[0] - 2.0) < 1e-9

    # 3) crack depth grows with χ.
    cd = [crack_depth(c, 0.01) for c in (0.0, 0.3, 0.6, 1.0)]
    print(f"3) crack depth vs χ: {np.round(cd,4)} (increasing)")
    assert np.all(np.diff(cd) >= 0) and cd[-1] > cd[0]

    # 4) the cell-size estimator recovers a known periodic pattern's wavelength.
    n = 128; period = 16.0
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    patt = (np.sin(2 * np.pi * xx / period) + np.sin(2 * np.pi * yy / period))
    est = measure_cell_size(patt, spacing_px=1.0)
    print(f"4) cell-size estimator on a period-{period:.0f} grid -> {est:.1f}")
    assert abs(est - period) < 2.0
    print("\nshrinkage-crack oracle self-checks passed.")
