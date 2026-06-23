"""Builds V3_4_canopy.ipynb via nbformat (Tier 3, the canopy).

Thresholds FROZEN here (below measured margins; protocol §1). V3.4 verifies that the canopy
operator deposits leaves with golden-angle phyllotaxis (the even, light-optimal packing), in a
plausible leaf-area-index range, spread through the crown — derived from the skeleton, not
authored. The falsifiable core is the angular-uniformity comparison vs a rational-angle control:
golden packs evenly, a rational angle clumps.

Run: .venv/bin/python verification_notebooks/phase3/_build_v34_nb.py
"""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
C = []
C.append(new_markdown_cell(
    "# V3.4 — Canopy generation & phyllotaxis  **TIER 3 / the foliage**\n\n"
    "**Claim (pre-registered).** The canopy operator (`nebula.operators.canopy`) deposits leaves "
    "on the grown twigs by spiral phyllotaxis at the golden angle ψ=137.5°, producing a canopy with "
    "a plausible leaf-area-index, spread through the crown, deterministically — foliage **derived** "
    "from the skeleton + a phyllotactic rule, not authored. Each leaf is also a fine-fuel element "
    "(mass/moisture/char) for the crown flash (V3.5).\n\n"
    "**Why load-bearing.** Blocker #4: Phase-0 was a bare skeleton — a tree's dominant visual mass "
    "is its canopy, and a leafless 'tree on fire' has nothing to flash. The golden angle is not "
    "decoration: it is the arrangement that packs leaves most evenly for light capture, so getting "
    "it right is what makes the canopy read as real foliage.\n\n"
    "**Independent oracle.** `phyllotaxis_ref.py` — the golden angle 360°(2−φ), Vogel's spiral, and "
    "the packing-uniformity measure (golden minimises nearest-neighbour-distance variance vs nearby "
    "non-golden controls). No tree code.\n\n"
    "**Pre-registered pass criteria (frozen below measured margins):**\n\n"
    "| # | Metric | Threshold |\n|---|---|---|\n"
    "| C1 | canopy generated: leaf count | > 1000 leaves |\n"
    "| C2 | phyllotaxis: median within-twig divergence angle | within 1.0° of golden (137.51°) |\n"
    "| C3 | **golden packs evenly**: angular-gap CV (golden) vs (90° control) | golden < control |\n"
    "| C4 | leaf-area-index (broadleaf canopy) | 2.0 ≤ LAI ≤ 8.0 |\n"
    "| C5 | crown fill: fraction of height-bands with leaves | ≥ 0.6 |\n"
    "| C6 | determinism: identical regeneration | bit-identical |"
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
    "from nebula.operators import canopy as cano\n"
    "import phyllotaxis_ref as ph\n"
    "np.seterr(all='ignore')\n"
    "tree = grow_tree(seed=7, gp=GrowthParams(dim=3))\n"
    "can = cano.generate_canopy(tree, cano.CanopyParams(), seed=7)\n"
    "R = {}\n"
    "print(f'tree {tree.n} nodes -> canopy {can.n} leaves on {len(np.unique(can.twig_node))} twigs')"
))
C.append(new_code_cell(
    "R['C1_leaves'] = can.n\n"
    "div = cano.divergence_angles_deg(can); R['C2_div'] = float(np.median(div))\n"
    "gold = ph.folded_to_180(ph.GOLDEN_ANGLE_DEG)\n"
    "u_g = cano.angular_uniformity(can)\n"
    "u_c = cano.angular_uniformity(cano.generate_canopy(tree, cano.CanopyParams(angle_deg=90.0), seed=7))\n"
    "R['C3_gold'], R['C3_ctrl'] = u_g, u_c\n"
    "R['C4_lai'] = cano.leaf_area_index(can, tree)\n"
    "R['C5_fill'] = cano.crown_fill(can, tree)\n"
    "R['C6_det'] = bool(np.array_equal(can.pos, cano.generate_canopy(tree, cano.CanopyParams(), seed=7).pos))\n"
    "print(f\"C1 leaves {R['C1_leaves']}\")\n"
    "print(f\"C2 median divergence {R['C2_div']:.2f}° (golden {gold:.2f}°)\")\n"
    "print(f\"C3 angular-gap CV golden {u_g:.3f} < control {u_c:.3f}: {u_g < u_c}\")\n"
    "print(f\"C4 LAI {R['C4_lai']:.2f} (2..8)\")\n"
    "print(f\"C5 crown fill {R['C5_fill']:.2f} (>=0.6)\")\n"
    "print(f\"C6 determinism {R['C6_det']}\")"
))
C.append(new_code_cell(
    "fig = plt.figure(figsize=(15, 5))\n"
    "ax0 = fig.add_subplot(131, projection='3d')\n"
    "s = np.random.default_rng(0).permutation(can.n)[:6000]\n"
    "ax0.scatter(can.pos[s,0], can.pos[s,1], can.pos[s,2], c='forestgreen', s=2, alpha=0.5)\n"
    "ax0.plot([], []); ax0.set_title(f'canopy ({can.n} leaves)'); ax0.set_axis_off()\n"
    "ax1 = fig.add_subplot(132)\n"
    "ax1.hist(div, bins=60, color='seagreen'); ax1.axvline(gold, color='r', ls='--', label=f'golden {gold:.1f}°')\n"
    "ax1.set_title('within-twig divergence angle'); ax1.set_xlabel('degrees'); ax1.legend()\n"
    "ax2 = fig.add_subplot(133)\n"
    "ax2.bar(['golden 137.5°','control 90°'], [u_g, u_c], color=['seagreen','gray'])\n"
    "ax2.set_title('angular-gap CV (lower = even packing)')\n"
    "C1=R['C1_leaves']>1000; C2=abs(R['C2_div']-gold)<1.0; C3=R['C3_gold']<R['C3_ctrl']\n"
    "C4=2.0<=R['C4_lai']<=8.0; C5=R['C5_fill']>=0.6; C6=R['C6_det']\n"
    "allpass=C1 and C2 and C3 and C4 and C5 and C6\n"
    "fig.suptitle(f'V3.4 — Canopy & phyllotaxis   VERDICT: {\"PASS\" if allpass else \"FAIL\"}', fontsize=14)\n"
    "fig.tight_layout(); fig.savefig('results/V3_4_canopy.png', dpi=110, bbox_inches='tight')\n"
    "print('saved results/V3_4_canopy.png')\n"
    "assert allpass, 'V3.4 criteria not all met'\n"
    "print('\\nV3.4 PASS — the tree has a real, phyllotactic, combustible canopy.')"
))
nb["cells"] = C
out = pathlib.Path(__file__).resolve().parent / "V3_4_canopy.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out}")
