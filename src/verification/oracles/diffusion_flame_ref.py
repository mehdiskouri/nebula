"""
Diffusion-flame oracle for V3.2 (Tier 3) — Burke–Schumann flame-sheet theory + flame-height
correlations. The independent reference for "the flame stands OFF the fuel."

Phase-0's combustion burned IN PLACE inside the wood voxels — there was no flame, just hot
wood. A real diffusion flame is where pyrolysis gas (fuel) meets entrained oxidizer in
stoichiometric proportion, which — because the gas is carried up by buoyancy (V3.1) — is a
sheet ABOVE the fuel source. Classical theory (Burke & Schumann 1928): with fast chemistry,
fuel and oxidizer cannot coexist; the flame is the iso-surface of the conserved MIXTURE
FRACTION Z at its stoichiometric value Z_st, where the temperature peaks. This module gives
that flame-sheet structure and the flame-HEIGHT scaling laws — Roper (laminar, H∝fuel flow)
and Heskestad (buoyant/turbulent, H∝Q^{2/5}) — with no shared code with the reacting solver.

Mixture fraction Z (conserved scalar): Z=1 in the fuel stream, Z=0 in the oxidizer stream.
"""
import numpy as np


def stoich_mixture_fraction(s_o2=1.0, Y_fuel_stream=1.0, Y_O2_oxidizer=0.23):
    """Z_st = Y_O2,ox / (s·Y_fuel,stream + Y_O2,ox): the mixture fraction at which fuel and
    oxidizer are in stoichiometric proportion (s = mass O2 per unit fuel)."""
    return Y_O2_oxidizer / (s_o2 * Y_fuel_stream + Y_O2_oxidizer)


def burke_schumann_profiles(Z, Z_st, T_ox=300.0, T_peak=2000.0, Y_fuel_stream=1.0,
                            Y_O2_oxidizer=0.23):
    """Flame-sheet (fast-chemistry) flamelet relations vs mixture fraction Z (array).

    Fuel side (Z>Z_st): no oxidizer; Y_fuel grows linearly to Y_fuel_stream at Z=1.
    Oxidizer side (Z<Z_st): no fuel; Y_O2 grows linearly to Y_O2,ox at Z=0.
    Temperature is a tent peaking at Z_st (adiabatic flame temperature), linear to the
    endpoints — the canonical Burke–Schumann piecewise-linear structure.
    """
    Z = np.asarray(Z, float)
    Yf = np.where(Z > Z_st, Y_fuel_stream * (Z - Z_st) / (1 - Z_st), 0.0)
    Yo = np.where(Z < Z_st, Y_O2_oxidizer * (1 - Z / Z_st), 0.0)
    T = np.where(Z <= Z_st, T_ox + (T_peak - T_ox) * (Z / Z_st),
                 T_ox + (T_peak - T_ox) * (1 - Z) / (1 - Z_st))
    return {"Y_fuel": Yf, "Y_O2": Yo, "T": T}


def roper_height_laminar(Q_fuel, D=1.0, c=1.0):
    """Laminar (Roper) flame height ∝ volumetric fuel flow / diffusivity: H = c·Q_fuel/D."""
    return c * np.asarray(Q_fuel, float) / D


HESKESTAD_EXPONENT = 2.0 / 5.0


def heskestad_height(Qdot, D, c=0.235, offset=1.02):
    """Heskestad mean flame height for a buoyant fire: L = c·Qdot^{2/5} − offset·D (Qdot in kW)."""
    return c * np.asarray(Qdot, float) ** HESKESTAD_EXPONENT - offset * D


def flame_height_from_field(Z_centerline, z, Z_st):
    """The flame TIP: highest z where the centerline mixture fraction crosses Z_st (fuel runs
    out into stoichiometric). The reaction sheet lies between the source and this height."""
    Z_centerline = np.asarray(Z_centerline, float); z = np.asarray(z, float)
    above = Z_centerline >= Z_st
    if not above.any():
        return 0.0
    return float(z[above].max())


if __name__ == "__main__":
    Z_st = stoich_mixture_fraction(s_o2=1.0, Y_fuel_stream=1.0, Y_O2_oxidizer=0.23)
    print(f"Z_st (s=1, oxidizer 0.23 O2) = {Z_st:.4f}")
    assert abs(Z_st - 0.23 / 1.23) < 1e-9

    # 1) Burke–Schumann: temperature peaks exactly at Z_st; fuel & oxidizer are mutually exclusive.
    Z = np.linspace(0, 1, 201)
    prof = burke_schumann_profiles(Z, Z_st)
    zpeak = Z[np.argmax(prof["T"])]
    overlap = float((prof["Y_fuel"] * prof["Y_O2"]).max())
    print(f"1) T peaks at Z={zpeak:.3f} (Z_st={Z_st:.3f}); fuel·O2 overlap max {overlap:.2e} (≈0)")
    assert abs(zpeak - Z_st) < 0.01 and overlap < 1e-12

    # 2) flame height increases with fuel supply (both correlations monotone & positive-exponent).
    Q = np.array([1.0, 2.0, 4.0, 8.0])
    Hl = roper_height_laminar(Q); Hb = heskestad_height(Q * 100, D=0.3)
    sl = np.polyfit(np.log(Q), np.log(Hl), 1)[0]
    print(f"2) Roper H ∝ Q^{sl:.2f} (laminar→1); Heskestad exponent {HESKESTAD_EXPONENT}")
    assert abs(sl - 1.0) < 1e-9 and np.all(np.diff(Hb) > 0)

    # 3) flame tip from a centerline mixture-fraction profile decaying with height.
    z = np.linspace(0, 20, 200); Zc = np.exp(-z / 6.0)
    H = flame_height_from_field(Zc, z, Z_st)
    print(f"3) flame tip (Z=Z_st crossing) at z={H:.2f}")
    assert H > 0
    print("\ndiffusion_flame oracle self-checks passed.")
