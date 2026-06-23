"""
Fine-fuel combustion oracle for V3.5 (Tier 3) — why the canopy flashes.

A leaf and a branch are the same material but burn utterly differently because of their SIZE:
a fuel particle's thermal response and burnout are governed by its surface-area-to-volume ratio
σ = (surface/volume) ∝ 1/d. Fine fuel (leaf, d~0.3 mm) has huge σ → it heats through, dries,
ignites, and burns out in seconds; a branch (d~cm) takes far longer. So when the trunk fire's
plume preheats the crown, the leaves ignite in a fast wave and the canopy "flashes" while the
wood is still warming. This oracle gives the size-scaling laws (no tree/flow code):

  - thermal response time τ_resp ∝ d²/α  (conduction through the particle — the d²-law),
  - dry-out + ignition time grows with d and with moisture (water must boil off first),
  - burnout time grows with d (a thick particle holds more fuel per unit heated area),
  - Rothermel-style spread: rate increases with σ (fine fuels carry fire fast),
  - crown-ignition lag = preheat distance / spread rate.

These are the falsifiable predictions the `fine_fuel` mechanism must reproduce.
"""
import numpy as np


def thermal_response_time(d, alpha=1.2e-7):
    """Conduction time to heat through a particle of thickness d [m]: τ ∝ d²/α (the d²-law)."""
    return np.asarray(d, float) ** 2 / alpha


def surface_area_to_volume(d):
    """σ = surface/volume for a slab/cylinder of characteristic size d: ∝ 1/d (here 4/d, a cylinder)."""
    return 4.0 / np.asarray(d, float)


def ignition_time(d, moisture=0.0, flux=1.0, alpha=1.2e-7, L_vap=1.0):
    """Time to dry (boil off moisture) + heat to ignition under a heat flux. Grows with d and moisture."""
    d = np.asarray(d, float)
    dry = L_vap * np.asarray(moisture, float) * d / flux       # latent load ∝ moisture·thickness
    heat = thermal_response_time(d, alpha) / max(flux, 1e-9)
    return dry + heat


def burnout_time(d, k=1.0, n=2.0):
    """Burnout time of a fuel particle ∝ d^n (the d²-law, n≈2): fine ≪ coarse."""
    return k * np.asarray(d, float) ** n


def rothermel_spread_rate(sigma, base=0.02, gain=2.0e-4):
    """Flame-spread rate increasing with the SAV ratio σ (fine fuels spread fire fast)."""
    return base + gain * np.asarray(sigma, float)


def crown_ignition_lag(preheat_distance, spread_rate):
    """Time for the fire front to climb a preheat distance to the crown (≥ 0)."""
    return np.asarray(preheat_distance, float) / np.maximum(spread_rate, 1e-9)


if __name__ == "__main__":
    d_leaf, d_branch = 3.0e-4, 2.0e-2     # 0.3 mm leaf vs 2 cm branch

    # 1) burnout & response: fine fuel is orders faster (the d²-law).
    rb = burnout_time(d_leaf) / burnout_time(d_branch)
    rr = thermal_response_time(d_leaf) / thermal_response_time(d_branch)
    print(f"1) leaf/branch burnout ratio {rb:.2e}, response ratio {rr:.2e} (both ≪ 1, d² scaling)")
    assert rb < 0.01 and rr < 0.01
    assert abs(np.log(rb) / np.log(d_leaf / d_branch) - 2.0) < 1e-9     # exponent is 2

    # 2) σ: the leaf has a far larger surface-to-volume ratio (why it heats fast).
    s_leaf, s_branch = surface_area_to_volume(d_leaf), surface_area_to_volume(d_branch)
    print(f"2) SAV σ: leaf {s_leaf:.0f} ≫ branch {s_branch:.0f}  (ratio {s_leaf/s_branch:.0f}×)")
    assert s_leaf > 50 * s_branch

    # 3) ignition time grows with moisture and with thickness (monotone).
    ts = [ignition_time(d_leaf, moisture=m) for m in (0.0, 0.3, 0.6, 0.9)]
    print(f"3) leaf ignition time vs moisture: {[round(t,4) for t in ts]} (increasing)")
    assert np.all(np.diff(ts) > 0)
    assert ignition_time(d_branch) > ignition_time(d_leaf)

    # 4) spread rate rises with σ; crown lags the base by a positive preheat time.
    R = rothermel_spread_rate(np.array([s_branch, s_leaf]))
    lag = crown_ignition_lag(3.0, R[1])
    print(f"4) spread rate branch {R[0]:.3f} < leaf {R[1]:.3f}; crown ignition lag {lag:.1f}s (>0)")
    assert R[1] > R[0] and lag > 0
    print("\nfinefuel oracle self-checks passed.")
