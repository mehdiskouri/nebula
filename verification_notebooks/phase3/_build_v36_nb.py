"""Builds V3_6_char_crack.ipynb via nbformat (Tier 3, the char alligator crackle).

Frozen criteria below measured margins. V3.6 verifies the char crack texture is DERIVED from the
char shrinkage state: the alligator-cell size follows the thickness law (spacing ∝ char depth), crack
depth grows with χ, cracks appear only on charred surface, deterministic — exported as a map (not a
painted char texture). Same discipline as the bark relief (V3.8).

Run: .venv/bin/python verification_notebooks/phase3/_build_v36_nb.py
"""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); C = []
C.append(new_markdown_cell(
    "# V3.6 — Char 'alligator' crackle (derived)  **TIER 3 / char texture**\n\n"
    "**Claim (pre-registered).** The char crack network (`nebula.geometry.char_texture`) is DERIVED from "
    "the char shrinkage state, not painted: the polygonal **cell size follows the thickness law** "
    "(spacing ∝ char depth — thicker char → bigger scales), crack **depth grows with χ**, and cracks "
    "appear only on the **charred** surface. It is exported as a displacement/normal/AO map.\n\n"
    "**Why load-bearing.** Char's distinctive alligator pattern is a defining 'it's really burnt' cue; "
    "deriving it from the shrinkage physics (not an authored texture) is causal fidelity for the surface.\n\n"
    "**Independent oracle.** `shrinkage_crack_ref.py` — the mud-crack / thermal-crack thickness law "
    "(spacing ∝ layer thickness, cell area ∝ thickness²) + an FFT cell-size estimator independent of the "
    "generator.\n\n"
    "| # | Metric | Threshold |\n|---|---|---|\n"
    "| C1 | alligator cell size vs char depth (measured FFT cell vs oracle prediction) | monotone↑ AND rel error < 0.2 |\n"
    "| C2 | crack depth grows with char fraction χ | monotone↑ |\n"
    "| C3 | cracks only on charred surface (unburnt displacement) | = 0 on unburnt AND > 0 on char |\n"
    "| C4 | determinism: identical crackle on re-evaluation | identical |"
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
    "from nebula.operators.growth import grow_tree, GrowthParams\n"
    "from nebula.geometry import char_texture as ct\n"
    "from nebula.geometry.mesh_export import tube_mesh\n"
    "import shrinkage_crack_ref as sc\n"
    "np.seterr(all='ignore')\n"
    "R = {}\n"
    "print('char_texture + shrinkage-crack oracle loaded')"
))
C.append(new_code_cell(
    "# C1 thickness law: measured alligator-cell size ∝ char depth, matching the oracle prediction\n"
    "depths = [0.004, 0.006, 0.010, 0.016]; meas = []; pred = []\n"
    "fields = {}\n"
    "for h in depths:\n"
    "    fld = ct.crack_field_2d(160, h, world_size=1.0, seed=7); fields[h]=fld\n"
    "    meas.append(sc.measure_cell_size(fld, spacing_px=1.0/160)); pred.append(float(sc.crack_spacing(h)))\n"
    "meas = np.array(meas); pred = np.array(pred)\n"
    "R['C1_mono'] = bool(np.all(np.diff(meas)>0)); R['C1_relerr'] = float(np.max(np.abs(meas-pred)/pred))\n"
    "print(f'depth {depths}')\n"
    "print(f'predicted spacing {np.round(pred,3).tolist()}')\n"
    "print(f'measured  cell    {np.round(meas,3).tolist()}')\n"
    "print(f\"C1 monotone {R['C1_mono']}, max rel error {R['C1_relerr']:.2f} (<0.2)\")"
))
C.append(new_code_cell(
    "# C2/C3/C4 on a tree (charred base)\n"
    "tree = grow_tree(seed=7, gp=GrowthParams(dim=3)); verts,faces,vnode = tube_mesh(tree)\n"
    "z = verts[:,2]; chi = np.where(z < z.min()+0.35*np.ptp(z), 0.9, 0.0)\n"
    "rel = ct.char_relief(tree, verts, vnode, chi, seed=7)\n"
    "cd = ct.crack_depth(np.array([0.0,0.3,0.6,1.0]), 0.01)\n"
    "R['C2_mono'] = bool(np.all(np.diff(cd)>=0) and cd[-1]>cd[0])\n"
    "R['C3_unburnt'] = float(rel['displacement'][chi==0].max()); R['C3_char'] = float((rel['fissure'][chi>0]>0).mean())\n"
    "rel2 = ct.char_relief(tree, verts, vnode, chi, seed=7)\n"
    "R['C4_det'] = bool(np.array_equal(rel['displacement'], rel2['displacement']))\n"
    "print(f\"C2 crack depth vs χ {np.round(cd,4).tolist()} monotone {R['C2_mono']}\")\n"
    "print(f\"C3 unburnt disp {R['C3_unburnt']:.4f} (=0); charred cracked frac {R['C3_char']:.2f}\")\n"
    "print(f\"C4 determinism {R['C4_det']}\")"
))
C.append(new_code_cell(
    "fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))\n"
    "ax[0].imshow(fields[0.016], cmap='gray_r'); ax[0].set_title('alligator crackle (deep char)'); ax[0].axis('off')\n"
    "ax[1].imshow(fields[0.004], cmap='gray_r'); ax[1].set_title('alligator crackle (thin char, finer cells)'); ax[1].axis('off')\n"
    "ax[2].plot(depths, pred, 'o-', label='oracle spacing'); ax[2].plot(depths, meas, 's--', label='measured cell')\n"
    "ax[2].set_title('cell size ∝ char depth (thickness law)'); ax[2].set_xlabel('char depth'); ax[2].legend()\n"
    "C1=R['C1_mono'] and R['C1_relerr']<0.2; C2=R['C2_mono']; C3=R['C3_unburnt']==0 and R['C3_char']>0; C4=R['C4_det']\n"
    "allpass=C1 and C2 and C3 and C4\n"
    "fig.suptitle(f'V3.6 — Char alligator crackle (derived)   VERDICT: {\"PASS\" if allpass else \"FAIL\"}', fontsize=13)\n"
    "fig.tight_layout(); fig.savefig('results/V3_6_char_crack.png', dpi=110, bbox_inches='tight')\n"
    "print('saved results/V3_6_char_crack.png')\n"
    "assert allpass, 'V3.6 criteria not all met'\n"
    "print('\\nV3.6 PASS — the char crackle cell size follows the thickness law; derived, char-only, exportable.')"
))
nb["cells"] = C
out = pathlib.Path(__file__).resolve().parent / "V3_6_char_crack.ipynb"
nbf.write(nb, str(out)); print(f"wrote {out}")
