"""Builds V3_3_emission_appearance.ipynb via nbformat (Tier 3, the appearance physics).

Covers blocker #2 (fake flame color) and #6 (no derived texture): the implementation appearance
map (`nebula.geometry.appearance`) must match the blackbody oracle (`blackbody.py`) on the
Planckian locus + T^4 + Beer–Lambert, and derive a physically-grounded surface BRDF from state
(char/wet/soot) — texture as a simulation output, not three hardcoded lerps. (Also exercises the
V3.7 surface-reflectance content: char/ash endpoints, wet darkening, ember = blackbody emission.)

Run: .venv/bin/python verification_notebooks/phase3/_build_v33_nb.py
"""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
C = []
C.append(new_markdown_cell(
    "# V3.3 — Blackbody emission, smoke & derived appearance  **TIER 3 / the fire's look**\n\n"
    "**Claim (pre-registered).** The implementation appearance map (`nebula.geometry.appearance`) "
    "derives the fire's look from physics: flame/ember EMISSION is blackbody incandescence (Planckian-"
    "locus color + T^4 intensity), SMOKE opacity is Beer–Lambert in the soot column, and SURFACE "
    "reflectance is derived from state (char darkens toward soot-black & roughens; moisture darkens). "
    "It must match the independent blackbody oracle within LUT tolerance.\n\n"
    "**Why load-bearing.** Blocker #2: Phase-0's flame was a constant emissive `[1.0,0.55,0.12]` + a "
    "linear `(T-650)/450` ramp — no Wien hue shift, no T^4, no smoke. Blocker #6: surface was flat "
    "vertex tints. Real fire color IS blackbody; getting it physical is what stops it reading as CG.\n\n"
    "**Independent oracle.** `blackbody.py` — Planck's law, Wien displacement, Stefan–Boltzmann, the "
    "CIE Planckian-locus color (Wyman-2013 analytic CMFs), and Beer–Lambert. The implementation ports "
    "this into a fast LUT; this notebook checks PARITY (the parity-test discipline).\n\n"
    "**Pre-registered pass criteria (frozen below measured margins):**\n\n"
    "| # | Metric | Threshold |\n|---|---|---|\n"
    "| C1 | blackbody color parity: max \\|appearance LUT − oracle sRGB\\| over 700–3000 K | < 0.05 |\n"
    "| C2 | Planckian locus: B/R chromaticity ratio monotone↑ in T; 900 K is red | monotone AND R>B at 900 K |\n"
    "| C3 | emission ∝ T^4 (Stefan–Boltzmann), and cold wood (≤T_on) does not emit | ratio(1500/750)=16±1e-6 AND e(700)=0 |\n"
    "| C4 | derived surface: char darkens & roughens, wet darkens, only hot char emits | all monotone AND char albedo ≈ soot ref |\n"
    "| C5 | smoke opacity vs soot column matches Beer–Lambert oracle | max abs diff < 1e-9 |"
))
C.append(new_code_cell(
    "import sys, pathlib\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "ROOT = pathlib.Path.cwd().resolve()\n"
    "while ROOT.name and not (ROOT / 'src').exists():\n"
    "    ROOT = ROOT.parent\n"
    "sys.path.insert(0, str(ROOT / 'src' / 'implementation'))\n"
    "sys.path.insert(0, str(ROOT / 'src' / 'verification' / 'oracles'))\n"
    "from nebula.geometry import appearance as ap\n"
    "import blackbody as bb\n"
    "np.seterr(all='ignore')\n"
    "R = {}\n"
    "print('appearance map + blackbody oracle loaded')"
))
C.append(new_code_cell(
    "# C1 parity vs oracle across the flame range\n"
    "Ts = np.linspace(700, 3000, 40)\n"
    "lut = ap.blackbody_rgb(Ts)\n"
    "orc = np.array([bb.blackbody_srgb(T) for T in Ts])\n"
    "R['C1_parity'] = float(np.abs(lut - orc).max())\n"
    "# C2 Planckian locus monotone\n"
    "br = (lut[:, 2] + 1e-6) / (lut[:, 0] + 1e-6)\n"
    "R['C2_mono'] = bool(np.all(np.diff(br) >= -1e-6)); R['C2_red900'] = bool(ap.blackbody_rgb(900)[0,0] > ap.blackbody_rgb(900)[0,2])\n"
    "# C3 T^4 emission\n"
    "R['C3_ratio'] = float(ap.emission_intensity(1500, T_on=0, T_full=3000) / ap.emission_intensity(750, T_on=0, T_full=3000))\n"
    "R['C3_coldoff'] = float(ap.emission_intensity(700.0))\n"
    "print(f\"C1 parity max|Δ| {R['C1_parity']:.3f} (<0.05)\")\n"
    "print(f\"C2 B/R monotone {R['C2_mono']}; 900K red {R['C2_red900']}\")\n"
    "print(f\"C3 emission(1500)/(750) {R['C3_ratio']:.2f} (=16); emission(700K)={R['C3_coldoff']}\")"
))
C.append(new_code_cell(
    "# C4 derived surface map\n"
    "bark = np.array([0.40, 0.26, 0.13])\n"
    "fresh = ap.surface_appearance(bark)\n"
    "charred = ap.surface_appearance(bark, T=1400.0, chi=0.9)\n"
    "wet = ap.surface_appearance(bark, moisture=1.0)\n"
    "R['C4_chardark'] = bool(charred['albedo'].sum() < fresh['albedo'].sum())\n"
    "R['C4_charrough'] = bool(charred['roughness'] > fresh['roughness'])\n"
    "R['C4_wetdark'] = bool(wet['albedo'].sum() < fresh['albedo'].sum())\n"
    "R['C4_emit'] = bool(charred['emission'].sum() > 0 and fresh['emission'].sum() == 0)\n"
    "R['C4_charref'] = float(np.abs(ap.surface_appearance(bark, chi=1.0)['albedo'] - ap.CHAR_ALBEDO).max())\n"
    "# C5 smoke Beer-Lambert parity\n"
    "col = np.array([0.0, 0.5, 1.0, 2.0, 5.0])\n"
    "R['C5_smoke'] = float(np.abs(ap.smoke_alpha(col) - (1 - bb.beer_lambert_transmittance(col))).max())\n"
    "print(f\"C4 char darkens {R['C4_chardark']} & roughens {R['C4_charrough']}; wet darkens {R['C4_wetdark']}; hot-char emits {R['C4_emit']}; char albedo≈soot (Δ{R['C4_charref']:.3f})\")\n"
    "print(f\"C5 smoke vs Beer–Lambert max|Δ| {R['C5_smoke']:.1e}\")"
))
C.append(new_code_cell(
    "fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))\n"
    "# Planckian locus swatches\n"
    "Tsw = np.linspace(700, 2600, 20); sw = ap.blackbody_rgb(Tsw)\n"
    "ax[0].imshow((sw*ap.emission_intensity(Tsw)[:,None])[None,:,:], aspect='auto', extent=[700,2600,0,1])\n"
    "ax[0].set_title('derived flame color (blackbody × T⁴)'); ax[0].set_xlabel('T [K]'); ax[0].set_yticks([])\n"
    "ax[1].plot(Ts, lut[:,0],'r',label='R'); ax[1].plot(Ts, lut[:,1],'g',label='G'); ax[1].plot(Ts, lut[:,2],'b',label='B')\n"
    "ax[1].plot(Ts, orc[:,0],'r--',alpha=0.5); ax[1].plot(Ts, orc[:,1],'g--',alpha=0.5); ax[1].plot(Ts, orc[:,2],'b--',alpha=0.5)\n"
    "ax[1].set_title(f'LUT (solid) vs oracle (dashed), Δ{R[\"C1_parity\"]:.3f}'); ax[1].set_xlabel('T [K]'); ax[1].legend()\n"
    "ax[2].plot(col, ap.smoke_alpha(col),'o-'); ax[2].set_title('smoke opacity (Beer–Lambert)'); ax[2].set_xlabel('soot column'); ax[2].set_ylabel('opacity')\n"
    "C1=R['C1_parity']<0.05; C2=R['C2_mono'] and R['C2_red900']; C3=abs(R['C3_ratio']-16)<1e-6 and R['C3_coldoff']==0\n"
    "C4=R['C4_chardark'] and R['C4_charrough'] and R['C4_wetdark'] and R['C4_emit'] and R['C4_charref']<0.02\n"
    "C5=R['C5_smoke']<1e-9\n"
    "allpass=C1 and C2 and C3 and C4 and C5\n"
    "fig.suptitle(f'V3.3 — Blackbody emission, smoke & derived appearance   VERDICT: {\"PASS\" if allpass else \"FAIL\"}', fontsize=13)\n"
    "fig.tight_layout(); fig.savefig('results/V3_3_emission_appearance.png', dpi=110, bbox_inches='tight')\n"
    "print('saved results/V3_3_emission_appearance.png')\n"
    "assert allpass, 'V3.3 criteria not all met'\n"
    "print('\\nV3.3 PASS — fire color is blackbody, smoke is Beer–Lambert, surface is derived from state.')"
))
nb["cells"] = C
out = pathlib.Path(__file__).resolve().parent / "V3_3_emission_appearance.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out}")
