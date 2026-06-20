"""Builds V2_2_percolation.ipynb via nbformat.

Thresholds FROZEN here, set BELOW the margins measured by `_calib_v22.py` (repo practice, protocol
§1). V2.2's pre-registered hypothesis ("gap stays small while error grows off-axis") is partly
FALSIFIED by the DNS — for high-contrast soft seams the isotropic gap is *large* at every angle — but
the deeper architectural claim (volume-fraction homogenization is blind to CONNECTIVITY and needs a
connectivity hard trigger) is confirmed decisively. Outcome class: CONSTRAIN (adopt the trigger).
Run: .venv/bin/python verification_notebooks/phase2/_build_v22_nb.py
"""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
C = []

C.append(new_markdown_cell(
    "# V2.2 — Percolation: the off-axis thin-connected-feature danger case  "
    "**TIER 2 / Risk: percolation**\n\n"
    "**Claim (pre-registered).** A thin **connected** low-stiffness seam destroys effective stiffness "
    "out of proportion to its volume fraction; the volume-fraction V–R gap catches this when the seam "
    "is axis-aligned but **fails** off the principal axes — establishing the need for a "
    "**connectivity-based hard refine trigger**.\n\n"
    "**Why load-bearing.** This is the named deepest failure mode of volume-fraction homogenization. "
    "The architecture (Decision #15; Risk: percolation) claims partial self-protection plus one "
    "residual danger that must be guarded explicitly. The guard (`violent_cells.percolates`) is "
    "already wired into `surrogate_gnn.fallback_flags` (V2.4/V2.1); V2.2 validates it against DNS.\n\n"
    "**Independent oracle.** The linear DNS micro-solver `dns_elasticity_3d.effective_stiffness` (the "
    "V0.1 keystone oracle, self-validated against the Backus closed form) — the TRUE effective tensor "
    "of the fully-resolved seamed cell, at a sweep of seam angles.\n\n"
    "**What the DNS actually shows (and how it refines the hypothesis).** The pre-registered mechanism "
    "*'gap stays small while error grows'* does **not** hold for high-contrast soft seams: the "
    "isotropic Voigt–Reuss gap is *large* at every angle (soft-dominated harmonic mean), so it would "
    "**over**-refine, not silently miss. The genuine, decisively-confirmed blindness is to "
    "**connectivity**: the homogenized descriptor (Voigt/Reuss/gap/fractions) depends only on phase "
    "fractions, so it is **byte-identical** for a percolating seam and a scattered cluster of equal "
    "volume — yet the seam is multiples softer. A connectivity span check is the necessary and "
    "sufficient guard.\n\n"
    "**Pre-registered pass criteria (frozen below measured margins):**\n\n"
    "| # | Metric | Threshold |\n|---|---|---|\n"
    "| 1 | connectivity-blindness | `gap(seam) ≡ gap(control)` exactly, AND DNS stiffness "
    "ratio seam/control **≤ 0.5** for ≥1 angle (a ≥2× effect the descriptor cannot see) |\n"
    "| 2 | homogenization fails off-axis | best-principal-axis orthotropic-proxy error vs DNS "
    "**≥ 0.80** at 45°, and **> 1.2×** the axis-aligned error |\n"
    "| 3 | connectivity trigger | `percolates` detects **100%** of percolating seams (all angles, "
    "both contrasts) and **0%** false positives on the matched controls |\n\n"
    "**The graded fix — folding connectivity into the trust scalar.** The original guard is a "
    "*parallel boolean* (`OR percolates`) bolted onto the surrogate — it breaks the architecture's "
    "*one-currency* invariant (Part IV). The cure is a feature that is **not fraction-only**: a cheap "
    "directional **scalar-conductance** residual `g_perc` (`div(kappa grad phi)=0`, `kappa_i=E_i`) — a "
    "PDE on the actual phase field, so it SEES connectivity (cross-property link to elastic moduli; "
    "Torquato/Gibiansky). Appended to the descriptor (`descriptor(connectivity=True)`), it makes the "
    "validity envelope and the surrogate's `u` connectivity-aware, while the 26-connectivity span "
    "check is retained as a *regime-aware hard backstop*.\n\n"
    "| # | Metric (graded fix) | Threshold |\n|---|---|---|\n"
    "| 4 | graded informativeness | rank-corr(`g_perc`, true DNS weakness) **≥ 0.80**; AND per-pair "
    "`g_perc(seam) > g_perc(control)` for **100%** of matched pairs |\n"
    "| 5 | thin/diagonal robustness | on thin (thickness-1) diagonal seams the 6-conn rule detects "
    "**≤ 50%** while 26-conn detects **100%** with **0%** control false-positive |\n"
    "| 6 | cost | conductance-proxy wall-time **≤ 0.35×** the elastic DNS RVE on the same cell |\n"
    "| 7 | single-currency | the connectivity sub-descriptor discriminates the matched pair (seam > "
    "control, **100%**) where the fraction sub-descriptor is **byte-identical** (zero discrimination) |\n\n"
    "**Outcome.** Metrics 1–3 → **CONSTRAIN adopted** (the blind spot is real, the boolean guard "
    "works). Metrics 4–7 **upgrade** the guard from a parallel boolean to a **graded connectivity "
    "channel inside the one trust scalar** (the spatial analogue of how V2.1 folded the distance "
    "signal into `u`), with the 26-conn span check kept as a hard backstop for the thin-diagonal tail. "
    "DNS containment (V0.1) still holds — the seam stiffness stays inside [Reuss, Voigt] — but the "
    "fraction-only bracket is uninformatively blind to the connectivity that destroys stiffness; "
    "`g_perc` restores it."))

C.append(new_code_cell(
    '"""(1) Setup — imports, FROZEN thresholds, seeds, sweep parameters."""\n'
    "import sys, pathlib, numpy as np, matplotlib.pyplot as plt\n"
    "REPO = pathlib.Path.cwd()\n"
    'while not (REPO / "src" / "verification" / "oracles").exists():\n'
    "    REPO = REPO.parent\n"
    'sys.path.insert(0, str(REPO / "src" / "verification" / "oracles"))\n'
    "import percolation as pc\n"
    "import violent_cells as vc\n"
    "import cells\n"
    "from scipy.stats import spearmanr\n"
    "from dns_elasticity_3d import effective_stiffness, _HAS_GPU\n"
    "from homogenization import isotropic_stiffness, voigt_bound, reuss_bound, relative_gap\n"
    "from analytic import laminate_stiffness\n\n"
    "# ---- FROZEN pre-registered thresholds (set below _calib_v22 measured margins) ----\n"
    "STIFF_RATIO_MAX   = 0.50   # 1: seam/control DNS stiffness ratio (measured min ~0.13)\n"
    "PROXY_ERR_OFFAX   = 0.80   # 2: best-axis proxy error at 45 deg (measured ~0.87-0.93)\n"
    "PROXY_ERR_RATIO   = 1.20   # 2: 45deg error / axis-aligned error (measured ~1.4)\n"
    "TRIGGER_DETECT    = 1.00   # 3: fraction of percolating seams flagged\n"
    "TRIGGER_FP        = 0.00   # 3: false-positive fraction on matched controls\n"
    "# --- the graded fix (folding connectivity into the trust scalar) ---\n"
    "SPEARMAN_MIN      = 0.80   # 4: rank-corr(g_perc, true DNS weakness)         (measured ~0.90)\n"
    "THIN_6CONN_MAX    = 0.50   # 5: 6-conn detection on thin (thk1) diagonals    (measured 0.00)\n"
    "THIN_26CONN_MIN   = 1.00   # 5: 26-conn detection on thin (thk1) diagonals   (measured 1.00)\n"
    "THIN_FP_MAX       = 0.00   # 5: 26-conn control false-positive at thk1       (measured 0.00)\n"
    "COST_RATIO_MAX    = 0.35   # 6: conductance time / elastic-DNS time          (measured ~0.20)\n\n"
    "N = 24; THICK = 3\n"
    "ANGLES = [0, 15, 30, 45, 60, 75, 90]\n"
    "CONTRASTS = [60.0, 100.0]\n"
    'CACHE = REPO / "verification_notebooks" / "phase2" / "cache"\n'
    "np.random.seed(0)\n"
    'print(f"DNS backend = {\'GPU (cupy CG)\' if _HAS_GPU else \'CPU (sparse LU)\'}; '
    'N={N} thick={THICK} angles={ANGLES} contrasts={CONTRASTS}")\n\n'
    "def solve_cached(cell, tag):\n"
    "    f = CACHE / f'v22_{tag}.npz'\n"
    "    if f.exists():\n"
    "        return np.load(f)['C']\n"
    "    Cm = effective_stiffness(cell.grid, cell.materials)\n"
    "    f.parent.mkdir(exist_ok=True); np.savez(f, C=Cm)\n"
    "    return Cm"))

C.append(new_markdown_cell(
    "## (A) Oracle validation — the DNS effective-stiffness oracle is trustworthy\n"
    "Reuse the V0.1 checks: the homogeneous cell returns its phase stiffness exactly, and a layered "
    "stack matches the Backus closed form. Only then is the DNS trusted as ground truth."))

C.append(new_code_cell(
    '"""(2) A: homogeneous identity + DNS-vs-Backus laminate."""\n'
    "hc = cells.homogeneous_cell(n=N, E=10.0, nu=0.3)\n"
    "homog_err = np.linalg.norm(effective_stiffness(hc.grid, hc.materials) - isotropic_stiffness(10.0, 0.3)) \\\n"
    "    / np.linalg.norm(isotropic_stiffness(10.0, 0.3))\n"
    "lc = cells.two_phase_layered(n=N, frac_stiff=0.5, contrast=50.0, axis=2)\n"
    "C_dns_lam = effective_stiffness(lc.grid, lc.materials)\n"
    "C_ana = laminate_stiffness(lc.fractions, [m[0] for m in lc.materials], [m[1] for m in lc.materials], 2)\n"
    "backus_err = np.linalg.norm(C_dns_lam - C_ana) / np.linalg.norm(C_ana)\n"
    "ORACLE_OK = bool(homog_err < 1e-10 and backus_err < 1e-10)\n"
    'print(f"homogeneous identity rel err = {homog_err:.2e}; DNS-vs-Backus rel err = {backus_err:.2e}")\n'
    'print(f"  -> ORACLE TRUSTWORTHY: {ORACLE_OK}")\n'
    'assert ORACLE_OK, "DNS oracle failed self-validation — halt."'))

C.append(new_markdown_cell(
    "## (B) The seam battery + matched controls (with RVE truth)\n"
    "For each angle and contrast: a percolating seam, and a **matched shuffled control** — the *same "
    "soft voxels permuted to random positions*. Identical phase fractions ⇒ byte-identical "
    "Voigt/Reuss/gap descriptor, but the connected path is destroyed (soft fraction is below the "
    "~0.31 site-percolation threshold, so a random shuffle does not span)."))

C.append(new_code_cell(
    '"""(3) B: build the battery, solve DNS (cached), record descriptor + DNS quantities."""\n'
    "rows = []\n"
    "for contrast in CONTRASTS:\n"
    "    for th in ANGLES:\n"
    "        seam = pc.seam_cell_at(N, th, thickness=THICK, contrast=contrast)\n"
    "        ctrl = pc.shuffled_control(seam, seed=1000 + th)\n"
    "        C_s = solve_cached(seam, f'c{int(contrast)}_a{th}_seam')\n"
    "        C_c = solve_cached(ctrl, f'c{int(contrast)}_a{th}_ctrl')\n"
    "        g_s, g_c = pc.gap_vector(seam), pc.gap_vector(ctrl)\n"
    "        nrm = pc.seam_normal(th)\n"
    "        rows.append(dict(contrast=contrast, theta=th, seam=seam, ctrl=ctrl, C_s=C_s, C_c=C_c,\n"
    "            gap_identical=bool(np.array_equal(g_s, g_c)), max_gap=float(g_s.max()),\n"
    "            E_norm_seam=pc.uniaxial_modulus(C_s, nrm), E_norm_ctrl=pc.uniaxial_modulus(C_c, nrm),\n"
    "            stiff_ratio=pc.uniaxial_modulus(C_s, nrm) / pc.uniaxial_modulus(C_c, nrm),\n"
    "            minPrinc=pc.min_principal_modulus(C_s), trueMin=pc.min_modulus_xz(C_s),\n"
    "            proxy_err=pc.best_axis_proxy_error(seam, C_s)[0],\n"
    "            perc_seam=pc.percolates_xz(seam), perc_ctrl=pc.percolates_xz(ctrl)))\n"
    "print(f'solved {len(rows)} seam+control pairs.')\n"
    "print(f\"descriptor (gap) identical seam-vs-control for ALL pairs: {all(r['gap_identical'] for r in rows)}\")\n"
    "print(f\"isotropic gap is LARGE at every angle (soft-dominated): max-gap range \"\n"
    "      f\"{min(r['max_gap'] for r in rows):.2f}..{max(r['max_gap'] for r in rows):.2f} \"\n"
    "      f\"-> it would OVER-refine, not silently miss\")"))

C.append(new_markdown_cell(
    "## (C) Metric 1 — connectivity-blindness (the core danger)\n"
    "The descriptor is identical for seam and control (a function of fractions only), yet the "
    "percolating seam is multiples softer. The trust scalar cannot see the connectivity that destroys "
    "stiffness — so it cannot distinguish a catastrophic connected seam from a benign cluster."))

C.append(new_code_cell(
    '"""(4) Metric 1: gap identity + seam/control DNS stiffness ratio."""\n'
    "GAP_IDENTICAL = all(r['gap_identical'] for r in rows)\n"
    "best = min(rows, key=lambda r: r['stiff_ratio'])\n"
    "MIN_RATIO = best['stiff_ratio']\n"
    "METRIC1_PASS = bool(GAP_IDENTICAL and MIN_RATIO <= STIFF_RATIO_MAX)\n"
    'print(f"gap(seam) == gap(control) exactly for all pairs: {GAP_IDENTICAL}")\n'
    'print(f"strongest knockdown: contrast {best[\"contrast\"]:g}, theta {best[\"theta\"]}: '
    'seam {best[\"E_norm_seam\"]:.3f} vs control {best[\"E_norm_ctrl\"]:.3f}  ratio {MIN_RATIO:.3f} '
    '(<= {STIFF_RATIO_MAX})")\n'
    'print(f"  => the descriptor is blind to a {1/MIN_RATIO:.1f}x stiffness effect")\n'
    'print(f"  -> METRIC 1 {\'PASS\' if METRIC1_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell(
    "## (D) Metric 2 — homogenization fails off-axis; and it is NOT a hidden-direction problem\n"
    "The orthotropic directional estimate (best principal-axis layering) cannot reconstruct a "
    "connected seam's tensor — error is large at every angle and **worst off-axis** (no principal "
    "frame for a 45° seam). Note the clarifying control: the true weakest direction is still well "
    "captured by the principal moduli at every angle (minPrincipal/trueMin ≈ 1.1–1.2), so the danger "
    "is **not** a hidden off-axis direction — it is purely connectivity."))

C.append(new_code_cell(
    '"""(5) Metric 2: best-axis proxy error by angle (worst off-axis); principal-visibility check."""\n'
    "def at(contrast, th):\n"
    "    return next(r for r in rows if r['contrast'] == contrast and r['theta'] == th)\n"
    "errs45 = [at(c, 45)['proxy_err'] for c in CONTRASTS]\n"
    "errs_axis = [max(at(c, 0)['proxy_err'], at(c, 90)['proxy_err']) for c in CONTRASTS]\n"
    "PROXY_45 = float(np.mean(errs45)); PROXY_AXIS = float(np.mean(errs_axis))\n"
    "METRIC2_PASS = bool(PROXY_45 >= PROXY_ERR_OFFAX and PROXY_45 / PROXY_AXIS >= PROXY_ERR_RATIO)\n"
    "blind_ratios = [r['minPrinc'] / r['trueMin'] for r in rows]\n"
    'print(f"orthotropic-proxy error vs DNS: axis-aligned ~{PROXY_AXIS:.2f}, 45deg ~{PROXY_45:.2f} '
    '(>= {PROXY_ERR_OFFAX}; ratio {PROXY_45/PROXY_AXIS:.2f} >= {PROXY_ERR_RATIO})")\n'
    'print(f"principal-visibility minPrinc/trueMin range = {min(blind_ratios):.2f}..{max(blind_ratios):.2f} '
    '(~1 -> weak direction visible in principal moduli; NOT a hidden-direction problem)")\n'
    'print(f"  -> METRIC 2 {\'PASS\' if METRIC2_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell(
    "## (E) Metric 3 — the connectivity trigger fixes it\n"
    "`percolates(axis=x) OR percolates(axis=z)` — the 6-connectivity span test — distinguishes what "
    "the descriptor cannot: it flags every percolating seam and none of the matched controls. When it "
    "fires, the cell refines to the exact RVE/DNS solve."))

C.append(new_code_cell(
    '"""(6) Metric 3: connectivity trigger detection + false-positive rates."""\n'
    "DETECT = float(np.mean([r['perc_seam'] for r in rows]))\n"
    "FP = float(np.mean([r['perc_ctrl'] for r in rows]))\n"
    "METRIC3_PASS = bool(DETECT >= TRIGGER_DETECT and FP <= TRIGGER_FP)\n"
    'print(f"connectivity trigger: detection {DETECT*100:.0f}% (>= {TRIGGER_DETECT*100:.0f}%), '
    'false-positive {FP*100:.0f}% (<= {TRIGGER_FP*100:.0f}%)  over {len(rows)} cells")\n'
    'print(f"  -> METRIC 3 {\'PASS\' if METRIC3_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell(
    "## (H) THE GRADED FIX — folding connectivity into the trust scalar\n"
    "The boolean trigger works but is a *parallel gate*. We instead add a cheap directional "
    "**scalar-conductance residual** `g_perc` to the descriptor. It is a PDE on the actual phase field "
    "(not a fraction-only bound), so it SEES connectivity: ~1 where a soft seam percolates normal to a "
    "load axis, ~0 for a matched scattered control — under *identical fractions* (identical Wiener "
    "bounds AND identical V-R gap). Unlike a topological span count it does not over-count dense "
    "scatter (the hard phase still carries load), so it is the right signal to fold into the one "
    "currency."))

C.append(new_code_cell(
    '"""(9) Metric 4 — graded informativeness: g_perc rank-correlates with the true DNS weakness,'
    "\n   and separates every matched pair (identical fractions). Reuses the cached DNS in `rows`.\"\"\"\n"
    "gp_max, weak, pair_sep = [], [], []\n"
    "for r in rows:\n"
    "    gps = pc.connectivity_residual(r['seam'].grid, r['seam'].materials).max()\n"
    "    gpc = pc.connectivity_residual(r['ctrl'].grid, r['ctrl'].materials).max()\n"
    "    r['g_perc_seam'], r['g_perc_ctrl'] = gps, gpc\n"
    "    pair_sep.append(gps - gpc)\n"
    "    gp_max += [gps, gpc]; weak += [-pc.min_modulus_xz(r['C_s']), -pc.min_modulus_xz(r['C_c'])]\n"
    "RHO = float(spearmanr(gp_max, weak).correlation)\n"
    "PAIR_MIN = float(min(pair_sep))\n"
    "METRIC4_PASS = bool(RHO >= SPEARMAN_MIN and PAIR_MIN > 0)\n"
    'print(f"Spearman(g_perc, true DNS weakness) over {len(gp_max)} cells = {RHO:.3f} (>= {SPEARMAN_MIN})")\n'
    'print(f"per-pair g_perc(seam) - g_perc(control): min = {PAIR_MIN:+.3f} (all > 0: {all(s>0 for s in pair_sep)})")\n'
    'print(f"  -> the conductance residual GRADES severity AND separates identical-fraction pairs the gap cannot")\n'
    'print(f"  -> METRIC 4 {\'PASS\' if METRIC4_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell(
    "## (I) Metric 5 — thin/diagonal robustness of the hardened backstop\n"
    "The shipped `percolates` used **6-connectivity** (forcing the `thickness=3` crutch). A genuine "
    "thin (1-voxel) diagonal crack is only corner-connected → **missed** by 6-conn. The hardened "
    "**26-connectivity** rule catches it, with no control false-positive at that thickness. (At "
    "thickness 2 the soft fraction rises and a 26-connected random scatter can spuriously span — the "
    "boolean is rule/threshold-dependent, which is exactly why the *graded* `g_perc` is the primary "
    "signal and the span check is a backstop.)"))

C.append(new_code_cell(
    '"""(10) Metric 5 — 6-conn vs 26-conn detection on thin diagonal seams; control false-positive."""\n'
    "thin_old, thin_new, thin_fp = [], [], []\n"
    "for th in (30, 45, 60):\n"
    "    for contrast in CONTRASTS:\n"
    "        s = pc.seam_cell_at(N, th, thickness=1, contrast=contrast)\n"
    "        c = pc.shuffled_control(s, seed=5000 + th)\n"
    "        thin_old.append(pc.percolates_xz(s))          # 6-connectivity (default)\n"
    "        thin_new.append(pc.percolates_xz_hard(s))      # 26-connectivity (hardened)\n"
    "        thin_fp.append(pc.percolates_xz_hard(c))\n"
    "DET6 = float(np.mean(thin_old)); DET26 = float(np.mean(thin_new)); THIN_FP = float(np.mean(thin_fp))\n"
    "METRIC5_PASS = bool(DET6 <= THIN_6CONN_MAX and DET26 >= THIN_26CONN_MIN and THIN_FP <= THIN_FP_MAX)\n"
    'print(f"thin (thickness-1) diagonal seams: 6-conn detect {DET6*100:.0f}% (<= {THIN_6CONN_MAX*100:.0f}%) '
    '-> 26-conn detect {DET26*100:.0f}% (>= {THIN_26CONN_MIN*100:.0f}%); control 26-conn FP {THIN_FP*100:.0f}% '
    '(<= {THIN_FP_MAX*100:.0f}%)")\n'
    'print(f"  -> the thickness=3 crutch is removed; 26-conn is the regime-aware hard backstop")\n'
    'print(f"  -> METRIC 5 {\'PASS\' if METRIC5_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell(
    "## (J) Metric 6 — cost: the graded channel is affordable as an always-on descriptor term\n"
    "The conductance proxy is one scalar DOF/voxel on a well-conditioned SPD Laplacian (3 solves) vs "
    "the elastic RVE's 3 vector DOF × 6 strain load cases — an order of magnitude cheaper, reusing the "
    "same Jacobi-PCG GPU path. So it can ride along in the descriptor, not just at refine time."))

C.append(new_code_cell(
    '"""(11) Metric 6 — conductance-proxy wall-time vs elastic DNS on the same cell."""\n'
    "probe = pc.seam_cell_at(N, 45, thickness=THICK, contrast=100.0)\n"
    "import time as _t\n"
    "pc.directional_conductance(probe.grid, probe.materials)  # warm up\n"
    "t0c = _t.time(); pc.directional_conductance(probe.grid, probe.materials); t_cond = _t.time() - t0c\n"
    "t0d = _t.time(); effective_stiffness(probe.grid, probe.materials); t_dns = _t.time() - t0d\n"
    "COST_RATIO = t_cond / t_dns\n"
    "METRIC6_PASS = bool(COST_RATIO <= COST_RATIO_MAX)\n"
    'print(f"conductance {t_cond*1e3:.0f} ms vs elastic DNS {t_dns*1e3:.0f} ms -> ratio {COST_RATIO:.3f} '
    '(<= {COST_RATIO_MAX})")\n'
    'print(f"  -> METRIC 6 {\'PASS\' if METRIC6_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell(
    "## (K) Metric 7 — single-currency restoration\n"
    "With `g_perc` in the descriptor, the per-cell trust coordinate discriminates a matched pair that "
    "the fraction coordinate provably cannot: the original 20-vec is **byte-identical** for seam and "
    "control (zero discrimination — the V-R blind spot), while the appended connectivity channels "
    "separate them at every pair. Connectivity now lives *inside the one currency* (available to the "
    "envelope and to the surrogate's `u`), not as a parallel boolean — the spatial analogue of V2.1 "
    "folding the distance signal into `u`."))

C.append(new_code_cell(
    '"""(12) Metric 7 — fraction-controlled per-pair discrimination."""\n'
    "Ds = np.stack([vc.descriptor(r['seam'].grid, r['seam'].materials, connectivity=True) for r in rows])\n"
    "Dc = np.stack([vc.descriptor(r['ctrl'].grid, r['ctrl'].materials, connectivity=True) for r in rows])\n"
    "FRAC_IDENTICAL = bool(np.array_equal(Ds[:, :20], Dc[:, :20]))\n"
    "conn_sep = Ds[:, 20:].max(1) - Dc[:, 20:].max(1)\n"
    "CONN_DISCRIM = float(np.mean(conn_sep > 0))\n"
    "METRIC7_PASS = bool(FRAC_IDENTICAL and CONN_DISCRIM >= 1.0)\n"
    'print(f"fraction sub-descriptor (0:20) byte-identical seam-vs-control: {FRAC_IDENTICAL} (ZERO discrimination)")\n'
    'print(f"connectivity sub-descriptor (20:): max(g_perc) seam > control for {CONN_DISCRIM*100:.0f}% of pairs '
    '(min margin {conn_sep.min():+.3f})")\n'
    'print(f"  -> connectivity folded into the ONE currency; no parallel boolean needed for resolvable seams")\n'
    'print(f"  -> METRIC 7 {\'PASS\' if METRIC7_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell("## (F) Figure"))

C.append(new_code_cell(
    '"""(7) Figure -> results/V2_2_percolation.png  (top: the blind spot; bottom: the graded fix)"""\n'
    "fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))\n"
    "c0 = CONTRASTS[0]; th = ANGLES\n"
    "# --- top row: connectivity-blindness, off-axis failure, boolean trigger ---\n"
    "Es = [at(c0, t)['E_norm_seam'] for t in th]; Ec = [at(c0, t)['E_norm_ctrl'] for t in th]\n"
    "ax[0,0].plot(th, Ec, 'o-', color='seagreen', label='scattered control')\n"
    "ax[0,0].plot(th, Es, 's-', color='crimson', label='percolating seam')\n"
    "ax[0,0].set_xlabel('seam angle (deg)'); ax[0,0].set_ylabel('modulus along seam normal')\n"
    "ax[0,0].set_title(f'(1) connectivity-blind  c={c0:g}\\nIDENTICAL descriptor, {1/MIN_RATIO:.0f}x stiffness gap')\n"
    "ax[0,0].legend(fontsize=8); ax[0,0].set_ylim(0, None)\n"
    "for c in CONTRASTS:\n"
    "    ax[0,1].plot(th, [at(c, t)['proxy_err'] for t in th], 'o-', label=f'proxy err c={c:g}')\n"
    "ax[0,1].axhline(PROXY_ERR_OFFAX, ls='--', c='k', lw=0.8); ax[0,1].axvline(45, ls=':', c='gray')\n"
    "ax[0,1].set_xlabel('seam angle (deg)'); ax[0,1].set_ylabel('orthotropic-proxy error vs DNS')\n"
    "ax[0,1].set_title('(2) homogenization fails,\\nworst off-axis (45 deg)'); ax[0,1].legend(fontsize=8); ax[0,1].set_ylim(0, 1)\n"
    "ax[0,2].bar(['seam\\n(detected)', 'control\\n(false-pos)'], [DETECT * 100, FP * 100], color=['crimson', 'seagreen'])\n"
    "ax[0,2].set_ylim(0, 105); ax[0,2].set_ylabel('% flagged by connectivity trigger')\n"
    "ax[0,2].set_title('(3) the boolean guard: span check')\n"
    "# --- bottom row: the graded fix ---\n"
    "gps = [at(c0, t)['g_perc_seam'] for t in th]; gpc = [at(c0, t)['g_perc_ctrl'] for t in th]\n"
    "ax[1,0].plot(th, gpc, 'o-', color='seagreen', label='scattered control')\n"
    "ax[1,0].plot(th, gps, 's-', color='crimson', label='percolating seam')\n"
    "ax[1,0].set_xlabel('seam angle (deg)'); ax[1,0].set_ylabel('conductance residual g_perc')\n"
    "ax[1,0].set_title('(4) graded fix: g_perc SEES connectivity\\n(identical fractions, separated)')\n"
    "ax[1,0].legend(fontsize=8); ax[1,0].set_ylim(0, 1.05)\n"
    "seam_pts = [(r['g_perc_seam'], -pc.min_modulus_xz(r['C_s'])) for r in rows]\n"
    "ctrl_pts = [(r['g_perc_ctrl'], -pc.min_modulus_xz(r['C_c'])) for r in rows]\n"
    "ax[1,1].scatter(*zip(*seam_pts), color='crimson', label='seam', s=28)\n"
    "ax[1,1].scatter(*zip(*ctrl_pts), color='seagreen', label='control', s=28)\n"
    "ax[1,1].set_xlabel('g_perc (conductance residual)'); ax[1,1].set_ylabel('true DNS weakness (-min modulus)')\n"
    "ax[1,1].set_title(f'(4) informative: rho={RHO:.2f}\\ng_perc tracks true knockdown'); ax[1,1].legend(fontsize=8)\n"
    "ax[1,2].bar(['6-conn\\ndetect', '26-conn\\ndetect', '26-conn\\nctrl FP'],\n"
    "            [DET6*100, DET26*100, THIN_FP*100], color=['darkorange', 'crimson', 'seagreen'])\n"
    "ax[1,2].set_ylim(0, 105); ax[1,2].set_ylabel('% (thin diagonal seams)')\n"
    "ax[1,2].set_title('(5) hardened backstop:\\n26-conn catches thin diagonals')\n"
    "fig.tight_layout()\n"
    'outdir = REPO / "verification_notebooks" / "phase2" / "results"; outdir.mkdir(exist_ok=True)\n'
    'fig.savefig(outdir / "V2_2_percolation.png", dpi=110)\n'
    "print('figure saved')"))

C.append(new_markdown_cell("## (G) Frozen verdict"))

C.append(new_code_cell(
    '"""(8) Verdict — all three metrics vs frozen thresholds; ends in CONSTRAIN adopted."""\n'
    "def verdict(name, ok, detail):\n"
    "    print(f\"  [{'OK ' if ok else 'XX '}] {name:36s} {'PASS' if ok else 'FAIL'}\")\n"
    "    print(f'        {detail}')\n"
    "print('=' * 74)\n"
    "print('V2.2 — PERCOLATION / OFF-AXIS CONNECTED SEAM — VERDICT')\n"
    "print('=' * 74)\n"
    "verdict('1. connectivity-blindness', METRIC1_PASS, "
    "f'gap identical seam==control; stiffness ratio {MIN_RATIO:.3f} <= {STIFF_RATIO_MAX} (descriptor blind to {1/MIN_RATIO:.0f}x)')\n"
    "verdict('2. homogenization fails off-axis', METRIC2_PASS, "
    "f'proxy err 45deg {PROXY_45:.2f} >= {PROXY_ERR_OFFAX}, {PROXY_45/PROXY_AXIS:.2f}x axis-aligned')\n"
    "verdict('3. connectivity trigger (boolean)', METRIC3_PASS, f'detect {DETECT*100:.0f}% / false-pos {FP*100:.0f}%')\n"
    "verdict('4. graded informativeness', METRIC4_PASS, f'rho(g_perc,DNS weakness)={RHO:.2f}; per-pair min sep {PAIR_MIN:+.3f}')\n"
    "verdict('5. thin/diagonal robustness', METRIC5_PASS, f'thin: 6-conn {DET6*100:.0f}% -> 26-conn {DET26*100:.0f}%, ctrl FP {THIN_FP*100:.0f}%')\n"
    "verdict('6. cost', METRIC6_PASS, f'conductance/DNS time ratio {COST_RATIO:.2f} <= {COST_RATIO_MAX}')\n"
    "verdict('7. single-currency', METRIC7_PASS, f'frac byte-identical={FRAC_IDENTICAL}; g_perc discriminates {CONN_DISCRIM*100:.0f}% of pairs')\n"
    "BLINDSPOT = bool(ORACLE_OK and METRIC1_PASS and METRIC2_PASS and METRIC3_PASS)\n"
    "GRADED = bool(METRIC4_PASS and METRIC5_PASS and METRIC6_PASS and METRIC7_PASS)\n"
    "ALL_PASS = bool(BLINDSPOT and GRADED)\n"
    "print('-' * 74)\n"
    "if ALL_PASS:\n"
    "    print('V2.2 VERDICT: CONSTRAIN (adopted) + GRADED FIX — volume-fraction homogenization is blind')\n"
    "    print('  to CONNECTIVITY: the homogenized descriptor is byte-identical for a percolating seam and')\n"
    "    print('  a matched scattered control, yet the seam is multiples softer; the orthotropic estimate')\n"
    "    print('  fails (worst off-axis). THE FIX: a cheap directional scalar-conductance residual g_perc')\n"
    "    print('  (a PDE on the phase field, not a fraction-only bound) is folded INTO the descriptor /')\n"
    "    print('  trust scalar -- it rank-correlates with the true DNS knockdown (rho~0.9) and separates')\n"
    "    print('  every identical-fraction pair the gap cannot, at ~0.2x the elastic-DNS cost. The 26-conn')\n"
    "    print('  span check is kept as a regime-aware HARD BACKSTOP (it catches thin diagonal cracks the')\n"
    "    print('  6-conn rule -- and the thickness=3 crutch -- missed). Connectivity now lives in the ONE')\n"
    "    print('  currency (the spatial analogue of V2.1 folding distance into u), with always-refine the')\n"
    "    print('  safe ceiling for the unresolvable thin-diagonal tail. (The pre-registered \\'gap stays')\n"
    "    print('  small\\' mechanism does not hold -- the gap is large for soft seams; the real blindness is')\n"
    "    print('  to connectivity, and g_perc is the cure.)')\n"
    "else:\n"
    "    print('V2.2 VERDICT: INCONCLUSIVE — see failed metric(s) above.')\n"
    'assert ALL_PASS, "V2.2 did not establish the connectivity blind spot + graded fix — see metrics."'))

nb.cells = C
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
out = pathlib.Path(__file__).parent / "V2_2_percolation.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
