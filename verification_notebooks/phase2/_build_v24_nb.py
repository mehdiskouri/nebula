"""Builds V2_4_surrogate_generalization.ipynb via nbformat (avoids f-string JSON-escaping bugs).

Thresholds are FROZEN here (set below the margins measured by _calib_v24.py, per protocol §1).
Run: .venv/bin/python verification_notebooks/phase2/_build_v24_nb.py
"""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
C = []

C.append(new_markdown_cell(
    "# V2.4 — Surrogate generalization, OOD fallback & PINN data efficiency  "
    "**TIER 2 / Decision #17**\n\n"
    "**Claim.** A physics-informed graph net trained on one archetype (char-wedge family) and "
    "conditioned on the homogenized descriptor (a) **generalizes** across the held-out parameter "
    "family, (b) **detectably degrades out-of-distribution** so the fallback predicate triggers, "
    "and (c) needs **less data** than a non-physics-informed baseline for equal accuracy.\n\n"
    "**Why load-bearing.** The affordable-forest / macro-surrogate story and the "
    "OOD-fallback-as-refinement story both rest on all three. V2.4 also gates V2.1 (the surrogate "
    "it calibrates).\n\n"
    "**Independent oracle.** The **damage/softening DNS** (`dns_damage_3d.py`) — genuine "
    "violent-regime ground truth where Voigt–Reuss is *invalid* (the true response leaves the "
    "bracket). Targets: normalized peak strength `y = peak_stress / (C0·k0)`.\n\n"
    "**Pre-registered pass criteria (frozen before running):**\n\n"
    "| # | Metric | Threshold | Nature |\n|---|---|---|---|\n"
    "| 1 | in-family generalization | median rel. err **< 12%** | empirical |\n"
    "| 2 | OOD detection (fallback trigger) | **≥ 99%** flagged, in-family false-pos **≤ 12%** | empirical |\n"
    "| 3 | PINN data efficiency | samples-to-5%-error ratio (PINN/baseline, 3-seed avg) **≤ 0.6** | engineering |\n\n"
    "**Failure → outcome.** Poor generalization → CONSTRAIN (more archetypes / richer descriptor); "
    "undetected OOD → links to V2.1's always-RVE fallback."))

C.append(new_code_cell(
    '"""(1) Setup — imports, sys.path, FROZEN thresholds, seeds, frozen damage-path params."""\n'
    "import sys, pathlib, numpy as np, matplotlib.pyplot as plt\n"
    "REPO = pathlib.Path.cwd()\n"
    'while not (REPO / "src" / "verification" / "oracles").exists():\n'
    "    REPO = REPO.parent\n"
    'sys.path.insert(0, str(REPO / "src" / "verification" / "oracles"))\n'
    "import violent_cells as vc\n"
    "import dns_damage_3d as dd\n"
    "from surrogate_gnn import (Ensemble, EnvelopeDetector, TrainCfg, build_dataset,\n"
    "                           fallback_flags, outcome_target, DATA_PARAMS)\n"
    "from homogenization import isotropic_stiffness\n\n"
    "# ---- FROZEN pre-registered thresholds (set below _calib_v24 measured margins) ----\n"
    "THR_GEN          = 0.12   # 1: median in-family relative error\n"
    "THR_OOD_DETECT   = 0.99   # 2: fraction of OOD cells flagged by the fallback trigger\n"
    "THR_FALSEPOS     = 0.12   # 2: max in-family false-positive rate\n"
    "THR_DATAEFF      = 0.60   # 3: max PINN/baseline samples-to-target ratio\n\n"
    "N = 12\n"
    'CACHE = str(REPO / "verification_notebooks" / "phase2" / "cache")\n'
    "np.random.seed(0)\n"
    "rng = np.random.default_rng(2024)   # SAME seed as _calib so the cache matches\n"
    "train = vc.family_battery(N, rng, 45)\n"
    "test  = vc.family_battery(N, rng, 20)\n"
    "ood   = vc.ood_battery(N, rng, 6)\n"
    'print(f"battery: {len(train)} train, {len(test)} test, {len(ood)} OOD;  '
    'params: max_strain={DATA_PARAMS.max_strain:.1e} k0={DATA_PARAMS.k0:.1e}")'))

C.append(new_markdown_cell(
    "## (A) Oracle validation — the violent-regime ground truth is trustworthy\n"
    "Confirm (i) the damage DNS reproduces the closed-form homogeneous softening bar, and (ii) the "
    "regime is genuinely *violent*: the damaged secant response falls **below the Reuss bound** "
    "(Voigt–Reuss is invalid here — the whole reason a surrogate/RVE handoff is needed). Then build "
    "(cache) the DNS dataset."))

C.append(new_code_cell(
    '"""(2) A: validate the damage oracle, then build/cache the violent-regime dataset."""\n'
    "# (i) homogeneous softening bar vs closed form\n"
    "g0 = np.zeros((10, 10, 10), dtype=np.int64)\n"
    "rb = dd.run_path(g0, [(10.0, 0.3)], dd.DamageParams(n_increments=20, max_strain=6e-3))\n"
    "C11 = isotropic_stiffness(10.0, 0.3)[0, 0]\n"
    "eps_eq = rb.strain_curve * np.sqrt(dd._M_EPS[0])\n"
    "sig_closed = (1 - dd._damage(eps_eq, dd.DamageParams())) * C11 * rb.strain_curve\n"
    "bar_err = np.abs(rb.stress_curve - sig_closed).max() / np.abs(sig_closed).max()\n"
    "# (ii) violent-regime: undamaged in V-R bracket, damaged below Reuss\n"
    "cw = vc.wedge_sample(N, 0.6, 60.0)\n"
    "rv = dd.run_path(cw.grid, cw.materials, DATA_PARAMS)\n"
    "dv, dr = dd.vr_brackets(cw.grid, cw.materials)\n"
    "d0, ds = rv.C0_linear[0, 0], rv.C_secant_final[0, 0]\n"
    "in_bracket0 = dr[0] - 1e-9 <= d0 <= dv[0] + 1e-9\n"
    "below_reuss = ds < dr[0]\n"
    'print(f"(i)  softening bar vs closed form: max rel err = {bar_err:.2e}")\n'
    'print(f"(ii) Reuss={dr[0]:.3f}  C0={d0:.3f}  Voigt={dv[0]:.3f}  C_secant={ds:.4f}")\n'
    'print(f"     undamaged in bracket={in_bracket0}; damaged below Reuss (V-R INVALID)={below_reuss}")\n'
    "ORACLE_OK = bool(bar_err < 1e-3 and in_bracket0 and below_reuss)\n"
    'assert ORACLE_OK, "damage oracle failed validation"\n\n'
    "# build/cache the dataset (DNS solves cached to .npz; ~11 min first run, instant after)\n"
    'd_tr = build_dataset(train, DATA_PARAMS, cache=f"{CACHE}/v24_train.npz")\n'
    'd_te = build_dataset(test,  DATA_PARAMS, cache=f"{CACHE}/v24_test.npz")\n'
    'd_oo = build_dataset(ood,   DATA_PARAMS, cache=f"{CACHE}/v24_ood.npz")\n'
    'print(f"dataset y (strength) range: train {d_tr[\'y\'].min():.3f}..{d_tr[\'y\'].max():.3f}; '
    'end-retention ~{np.median(d_tr[\'ret_end\']):.3f} (collapsed -> binary, as expected)")'))

C.append(new_markdown_cell(
    "## (B) In-family generalization\n"
    "Train the physics-informed deep-ensemble surrogate on the archetype family; measure relative "
    "error on the held-out in-family parameters."))

C.append(new_code_cell(
    '"""(3) B: train PINN ensemble, evaluate held-out in-family generalization."""\n'
    "ens = Ensemble.train(train, d_tr['y'], TrainCfg(epochs=400, physics=True), M=5, base_seed=0)\n"
    "pred = ens.predict(test)\n"
    "rel = np.abs(pred['mean'] - d_te['y']) / np.abs(d_te['y'])\n"
    "GEN = float(np.median(rel))\n"
    "METRIC1_PASS = GEN < THR_GEN\n"
    'print(f"in-family median rel err = {GEN:.4f}  (p90={np.quantile(rel,0.9):.4f})  '
    'threshold < {THR_GEN}")\n'
    'print(f"  -> METRIC 1 {\'PASS\' if METRIC1_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell(
    "## (C) OOD detection via the fallback trigger\n"
    "The learned tier's validity guard = **envelope-exit OR percolation** (the operator schema's "
    "'envelope-exit OR residual-spike' plus V2.2's connectivity check — the seam is the percolation "
    "blind spot the volume-fraction descriptor cannot see). Must flag essentially all OOD cells with "
    "a low in-family false-positive rate."))

C.append(new_code_cell(
    '"""(4) C: multi-signal fallback trigger — OOD detection & in-family false-positive."""\n'
    "env = EnvelopeDetector.fit(train)\n"
    "z_thr = float(np.quantile(env.score(test), 0.95))\n"
    "f_oo, s_oo = fallback_flags(ood, env, z_thr)\n"
    "f_te, s_te = fallback_flags(test, env, z_thr)\n"
    "OOD_DETECT = float(f_oo.mean()); FALSEPOS = float(f_te.mean())\n"
    "METRIC2_PASS = (OOD_DETECT >= THR_OOD_DETECT) and (FALSEPOS <= THR_FALSEPOS)\n"
    'print(f"OOD detected = {OOD_DETECT*100:.1f}% (>= {THR_OOD_DETECT*100:.0f}%); '
    'in-family false-pos = {FALSEPOS*100:.1f}% (<= {THR_FALSEPOS*100:.0f}%)")\n'
    'print(f"  -> METRIC 2 {\'PASS\' if METRIC2_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell(
    "## (D) PINN data efficiency\n"
    "Train PINN and a pure-data baseline at increasing training-set sizes; compare samples-to-"
    "target-accuracy. Physics priors (range + monotone-in-contrast) should let the PINN reach the "
    "target with substantially less data."))

C.append(new_code_cell(
    '"""(5) D: PINN vs pure-data baseline — samples-to-5%-target ratio (3-seed averaged)."""\n'
    "sizes = [5, 8, 12, 18, 30]; SEEDS = [10, 20, 30]; TARGET = 0.05\n"
    "def curve(physics):\n"
    "    M = np.zeros((len(SEEDS), len(sizes)))\n"
    "    for i, sd in enumerate(SEEDS):\n"
    "        for j, k in enumerate(sizes):\n"
    "            e = Ensemble.train(train[:k], d_tr['y'][:k], TrainCfg(epochs=400, physics=physics),\n"
    "                               M=5, base_seed=sd)\n"
    "            M[i, j] = np.median(np.abs(e.predict(test)['mean'] - d_te['y']) / np.abs(d_te['y']))\n"
    "    return M.mean(0)\n"
    "e_pinn, e_base = curve(True), curve(False)\n"
    "def s2t(e):\n"
    "    hit = [s for s, v in zip(sizes, e) if v <= TARGET]\n"
    "    return hit[0] if hit else np.inf\n"
    "sp, sb = s2t(e_pinn), s2t(e_base)\n"
    "RATIO = sp / sb if np.isfinite(sb) and sb > 0 else np.inf\n"
    "LOWDATA = float(e_pinn[:3].mean() / e_base[:3].mean())\n"
    "METRIC3_PASS = RATIO <= THR_DATAEFF\n"
    'print(f"PINN(3-seed)={np.round(e_pinn,4)}  base={np.round(e_base,4)}")\n'
    'print(f"samples-to-{TARGET:.0%}: PINN N={sp} baseline N={sb}; ratio={RATIO:.2f} (<= {THR_DATAEFF}); '
    'scarce-regime err ratio={LOWDATA:.2f}")\n'
    'print(f"  -> METRIC 3 {\'PASS\' if METRIC3_PASS else \'FAIL\'}")'))

C.append(new_markdown_cell("## (E) Figure"))

C.append(new_code_cell(
    '"""(6) Multi-panel figure -> results/V2_4_surrogate_generalization.png"""\n'
    "fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))\n"
    "ax[0].scatter(d_te['y'], pred['mean'], c=pred['u'], cmap='viridis', s=28)\n"
    "lim = [min(d_te['y'].min(), pred['mean'].min()), max(d_te['y'].max(), pred['mean'].max())]\n"
    "ax[0].plot(lim, lim, 'k--', lw=1); ax[0].set_xlabel('DNS strength y'); ax[0].set_ylabel('surrogate')\n"
    "ax[0].set_title(f'(B) in-family fit  med rel={GEN:.3f}')\n"
    "ax[1].hist(s_te, bins=14, alpha=0.7, label='in-family', color='steelblue')\n"
    "ax[1].hist(s_oo, bins=14, alpha=0.7, label='OOD', color='crimson')\n"
    "ax[1].axvline(z_thr, ls='--', c='k'); ax[1].set_xlabel('fallback envelope score'); ax[1].legend()\n"
    "ax[1].set_title(f'(C) OOD detect {OOD_DETECT*100:.0f}%  fp {FALSEPOS*100:.0f}%')\n"
    "ax[2].plot(sizes, e_pinn, 'o-', label='PINN', color='seagreen')\n"
    "ax[2].plot(sizes, e_base, 's--', label='pure-data', color='darkorange')\n"
    "ax[2].axhline(TARGET, ls=':', c='k'); ax[2].set_xlabel('# training cells'); ax[2].set_ylabel('median rel err')\n"
    "ax[2].set_title(f'(D) data efficiency  ratio={RATIO:.2f}'); ax[2].legend()\n"
    "fig.tight_layout()\n"
    'outdir = REPO / "verification_notebooks" / "phase2" / "results"; outdir.mkdir(exist_ok=True)\n'
    'fig.savefig(outdir / "V2_4_surrogate_generalization.png", dpi=110)\n'
    "print('figure saved')"))

C.append(new_markdown_cell("## (F) Frozen verdict"))

C.append(new_code_cell(
    '"""(7) Verdict — all metrics vs frozen thresholds; ends in assert ALL_PASS."""\n'
    "def verdict(name, ok, detail):\n"
    "    print(f\"  [{'OK ' if ok else 'XX '}] {name:36s} {'PASS' if ok else 'FAIL'}\")\n"
    "    print(f'        {detail}')\n"
    "print('=' * 72)\n"
    "print('V2.4 — SURROGATE GENERALIZATION / OOD / DATA-EFFICIENCY — VERDICT')\n"
    "print('=' * 72)\n"
    "verdict('1. in-family generalization', METRIC1_PASS, f'median rel err {GEN:.4f} < {THR_GEN}')\n"
    "verdict('2. OOD detection + low false-pos', METRIC2_PASS, "
    "f'detect {OOD_DETECT*100:.1f}% >= {THR_OOD_DETECT*100:.0f}%, fp {FALSEPOS*100:.1f}% <= {THR_FALSEPOS*100:.0f}%')\n"
    "verdict('3. PINN data efficiency', METRIC3_PASS, f'ratio {RATIO:.2f} <= {THR_DATAEFF}')\n"
    "ALL_PASS = bool(ORACLE_OK and METRIC1_PASS and METRIC2_PASS and METRIC3_PASS)\n"
    "print('-' * 72)\n"
    "print('V2.4 VERDICT:', 'PASS' if ALL_PASS else 'FAIL')\n"
    'assert ALL_PASS, "V2.4 did not pass — see metrics above"'))

nb.cells = C
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
out = pathlib.Path(__file__).parent / "V2_4_surrogate_generalization.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
