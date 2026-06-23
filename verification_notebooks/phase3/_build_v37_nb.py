"""Builds V3_7_reflectance.ipynb via nbformat (Tier 3, reflectance grounding).

Verifies the derived appearance map's material endpoints sit in measured reflectance ranges (char,
ash, wet-darkening) and the ember emission follows blackbody T^4 — so the appearance is a derived
output whose constants are grounded, not invented. Complements V3.3 (which checked the blackbody
flame colour + Beer-Lambert smoke).

Run: .venv/bin/python verification_notebooks/phase3/_build_v37_nb.py
"""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); C = []
C.append(new_markdown_cell(
    "# V3.7 — Surface reflectance grounding  **TIER 3 / appearance constants**\n\n"
    "**Claim (pre-registered).** The derived appearance map (`nebula.geometry.appearance`) lands its "
    "material endpoints in **measured reflectance ranges** (char ≈ soot-black, ash pale-grey, wet "
    "darkening) and its ember emission follows **blackbody T⁴** — the appearance is a derived simulation "
    "output, but its constants are grounded, not invented.\n\n"
    "**Independent oracle.** `reflectance_ref.py` — representative diffuse-reflectance ranges + the "
    "wet-darkening factor + the Stefan–Boltzmann emission scaling.\n\n"
    "| # | Metric | Threshold |\n|---|---|---|\n"
    "| C1 | char albedo luminance in measured char range | ∈ [0.02, 0.06] |\n"
    "| C2 | ash albedo luminance in measured ash range | ∈ [0.22, 0.45] |\n"
    "| C3 | wet-darkening factor in measured range | ∈ [0.45, 0.75] |\n"
    "| C4 | ember emission scales as T⁴ (Stefan–Boltzmann) | ratio(1600/800) = 16 |\n"
    "| C5 | derived ordering: char ≪ bark ≪ fresh wood luminance | strictly increasing |"
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
    "import reflectance_ref as rr\n"
    "np.seterr(all='ignore')\n"
    "R = {}\n"
    "lum = rr.luminance\n"
    "R['C1_char'] = lum(ap.CHAR_ALBEDO); R['C2_ash'] = lum(ap.ASH_ALBEDO); R['C3_wet'] = ap.WET_DARKEN\n"
    "R['C4_ratio'] = float(ap.emission_intensity(1600, T_on=0, T_full=3000)/ap.emission_intensity(800, T_on=0, T_full=3000))\n"
    "bark = ap.surface_appearance(np.array([0.40,0.26,0.13]))['albedo']\n"
    "char = ap.surface_appearance(np.array([0.40,0.26,0.13]), chi=1.0)['albedo']\n"
    "fresh = ap.surface_appearance(np.array([0.5,0.42,0.30]))['albedo']\n"
    "R['C5_order'] = bool(lum(char) < lum(bark) < lum(fresh))\n"
    "print(f\"C1 char luminance {R['C1_char']:.3f} in [0.02,0.06]: {rr.in_range(R['C1_char'],'char')}\")\n"
    "print(f\"C2 ash luminance {R['C2_ash']:.3f} in [0.22,0.45]: {rr.in_range(R['C2_ash'],'ash')}\")\n"
    "print(f\"C3 wet-darken {R['C3_wet']:.2f} in [0.45,0.75]: {rr.WET_DARKEN_RANGE[0]<=R['C3_wet']<=rr.WET_DARKEN_RANGE[1]}\")\n"
    "print(f\"C4 emission T^4 ratio {R['C4_ratio']:.1f} (=16)\")\n"
    "print(f\"C5 ordering char {lum(char):.3f} < bark {lum(bark):.3f} < fresh {lum(fresh):.3f}: {R['C5_order']}\")"
))
C.append(new_code_cell(
    "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
    "names = ['char','bark','ash','fresh_wood']; vals = [R['C1_char'], lum(bark), R['C2_ash'], lum(fresh)]\n"
    "ax[0].bar(names, vals, color=['0.1','saddlebrown','0.7','wheat'])\n"
    "for k,(lo,hi) in rr.ALBEDO_RANGES.items():\n"
    "    x = names.index(k) if k in names else None\n"
    "    if x is not None: ax[0].plot([x,x],[lo,hi],'r_-',lw=2)\n"
    "ax[0].set_title('derived albedo luminance vs measured ranges (red)'); ax[0].set_ylabel('luminance')\n"
    "T = np.linspace(600,2400,80); ax[1].plot(T, ap.emission_intensity(T,T_on=0,T_full=3000)); ax[1].set_title('ember emission ∝ T⁴'); ax[1].set_xlabel('T [K]')\n"
    "C1=rr.in_range(R['C1_char'],'char'); C2=rr.in_range(R['C2_ash'],'ash'); C3=rr.WET_DARKEN_RANGE[0]<=R['C3_wet']<=rr.WET_DARKEN_RANGE[1]; C4=abs(R['C4_ratio']-16)<1e-6; C5=R['C5_order']\n"
    "allpass=C1 and C2 and C3 and C4 and C5\n"
    "fig.suptitle(f'V3.7 — Surface reflectance grounding   VERDICT: {\"PASS\" if allpass else \"FAIL\"}', fontsize=13)\n"
    "fig.tight_layout(); fig.savefig('results/V3_7_reflectance.png', dpi=110, bbox_inches='tight')\n"
    "print('saved results/V3_7_reflectance.png')\n"
    "assert allpass, 'V3.7 criteria not all met'\n"
    "print('\\nV3.7 PASS — appearance constants land in measured reflectance ranges; emission is blackbody T⁴.')"
))
nb["cells"] = C
out = pathlib.Path(__file__).resolve().parent / "V3_7_reflectance.ipynb"
nbf.write(nb, str(out)); print(f"wrote {out}")
