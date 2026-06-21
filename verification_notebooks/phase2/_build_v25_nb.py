"""Builds V2_5_inverse_design.ipynb via nbformat.

Thresholds FROZEN here, set BELOW the margins measured by `_calib_v25.py` (repo practice, protocol §1).
V2.5 verifies Decision #18 / ARCHITECTURE §III.5-III.6: inverse design converges to parameters that
produce a target outcome; the candidate, re-simulated with the REAL operators, matches within tolerance
(a *verified candidate, not an oracle*); and an incoherent target is reported IMPOSSIBLE. Two domains:
material (V2.4 surrogate strength, verified by damage-DNS) and physiology / survival-spectrum (regulator,
verified by the basin oracle). Outcome: PASS (with the adopted "verified-candidate gate"). Marks Tier 2
COMPLETE. Run: .venv/bin/python verification_notebooks/phase2/_build_v25_nb.py
"""
import pathlib
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def code(*lines):
    return new_code_cell("\n".join(lines))


nb = new_notebook()
C = []

C.append(new_markdown_cell(
    "# V2.5 — Inverse design convergence & candidate verification  **TIER 2 / Decision #18**\n\n"
    "**Claim (pre-registered).** Gradient descent on input parameters through the differentiable "
    "surrogate converges to parameters producing a target outcome; the resulting candidate, "
    "**re-simulated with the REAL operators, matches the target within tolerance** (a *verified "
    "candidate, not an oracle*); and a **known-incoherent target is reported impossible**, not "
    "silently 'solved'.\n\n"
    "**Why load-bearing.** This is the 'set the result, gradient-descend to the parameters' non-render "
    "analysis (§III.5) that powers the dev dial-in-a-target workflow and the **survival-spectrum** "
    "('can this creature survive in world X' is *derived, not authored*, §III.6).\n\n"
    "**The core idea — and the load-bearing risk.** Inverse design optimizes a *cheap differentiable* "
    "model, so it can be **exploited at the model's weak points**: the surrogate confidently 'hits' a "
    "target the real operator misses, especially near the achievable boundary. So the heart of V2.5 is "
    "the **verification + trust gate**: every candidate is re-simulated with the real operator and "
    "gated by its distance-to-manifold (V2.4/V2.1 `FeatureDensity`). An unreachable target returns a "
    "**precise impossibility** ('fail informatively'), not a fabrication. The adopted artifact is "
    "`inverse_design.Candidate(theta, surrogate_pred, verified_real, trust, status ∈ {verified, "
    "untrusted→RVE, impossible})` — the 'verified candidate, not oracle' discipline made concrete.\n\n"
    "**Differentiable path.** For the 2-phase char-wedge family the featurization is ANALYTIC in "
    "theta=(depth,contrast): the descriptor is fraction-only and the region-graph node features are "
    "smooth in depth (the wedge profile), so `strength(theta)=surrogate(features(theta))` is "
    "torch-autodiff — verified against finite difference (M1). The surrogate forward is differentiable; "
    "only `Ensemble.predict` blocks grad, so `inverse_design` adds a thin differentiable forward over "
    "the public members/normalizer WITHOUT editing surrogate_gnn (V2.4/V2.1 byte-identical). In the "
    "physiological domain the homeostatic pressure is the smooth root of the regulator residual, "
    "differentiated w.r.t. the setpoint by the implicit function theorem.\n\n"
    "**Independent oracle.** The REAL operators re-simulating the candidate: the damage-DNS "
    "(`dns_damage_3d.run_path`, the same oracle V2.4 trained on) for material; the brute-force basin "
    "(`regulator.basin_map`) + viability margin for physiology.\n\n"
    "**Pre-registered pass criteria (frozen below `_calib_v25` measured margins):**\n\n"
    "| # | Metric | Threshold |\n|---|---|---|\n"
    "| M1 | differentiable path correct | autodiff d(strength)/d(theta) vs finite-diff rel err **< 1e-3** |\n"
    "| M2 | convergence | **≥ 90%** of in-range targets reach surrogate error **< 0.02** |\n"
    "| M3 | verified candidate fidelity | median real-operator |achieved−target|/target **< 0.12** "
    "(the V2.4 in-family bound) on the verified set |\n"
    "| M4 | candidate-not-oracle gate | **0** fabricated successes on incoherent targets (all "
    "impossible/untrusted); and trust rank-correlates with the real error (Spearman **≥ 0.55**) |\n"
    "| M5 | physiology inverse | homeostatic target achieved **< 5%**, verified in the basin "
    "(basin_area>0, viability margin>0) |\n"
    "| M6 | physiology impossibility | a world below `r_critical` returns **impossible** (0 fabricated); "
    "the survival-spectrum basin area contracts to 0 |\n\n"
    "**Outcome.** All → **PASS** (with the adopted verified-candidate gate). Convergence fails → "
    "**REDESIGN**; large surrogate-vs-real gap caught only by the gate → **CONSTRAIN (adopted)**: trust "
    "low-distance candidates, route the rest to real-operator verification (V2.1 always-RVE) or report "
    "impossible. **This is the final Tier-2 verification — its pass marks TIER 2 COMPLETE.**"))

C.append(code(
    '"""(1) Setup — imports, FROZEN thresholds, seeds, cached damage-DNS, train the V2.4 surrogate."""',
    "import sys, pathlib, numpy as np, matplotlib.pyplot as plt, torch",
    "from scipy.stats import spearmanr",
    "REPO = pathlib.Path.cwd()",
    'while not (REPO / "src" / "verification" / "oracles").exists():',
    "    REPO = REPO.parent",
    'sys.path.insert(0, str(REPO / "src" / "verification" / "oracles"))',
    "import violent_cells as vc, regulator as reg, inverse_design as idz",
    "from surrogate_gnn import Ensemble, TrainCfg, build_dataset, set_determinism, DEVICE, DATA_PARAMS",
    "from dns_damage_3d import _HAS_GPU",
    "",
    "# ---- FROZEN pre-registered thresholds (set below _calib_v25 measured margins) ----",
    "GRAD_TOL      = 1e-3    # M1: autodiff vs finite-diff grad rel err (measured ~6e-7)",
    "SURR_TOL      = 0.02    # convergence definition: |surrogate - target|",
    "CONV_MIN      = 0.90    # M2: fraction of in-range targets that converge",
    "REAL_TOL      = 0.12    # M3: median real-operator rel error on the verified set (V2.4 in-family)",
    "TRUST_RHO_MIN = 0.60    # M4: Spearman(trust, real error) — the gate is predictive (measured 0.92)",
    "PHYS_TOL      = 0.05    # M5: homeostatic target rel error",
    "",
    'CACHE = str(REPO / "verification_notebooks" / "phase2" / "cache")',
    "set_determinism(0)",
    "np.set_printoptions(precision=4, suppress=True)",
    "print(f\"backend {'GPU' if _HAS_GPU else 'CPU'}; device {DEVICE}; cell n={idz.N}\")",
    "",
    "# train the V2.4 physics-informed surrogate (RPF -> distance-aware trust) on the char-wedge family",
    "rng = np.random.default_rng(2025)",
    "train = vc.family_battery(idz.N, rng, 45)",
    "y_tr = build_dataset(train, DATA_PARAMS, cache=CACHE + '/v25_train.npz')['y']",
    "# beta=0: the trust signal is FeatureDensity (descriptor distance, fit independently of the net),",
    "# so RPF is unnecessary here. Training is ~1 min (the 45 ground-truth damage-DNS are cached).",
    "ens = Ensemble.train(train, y_tr, TrainCfg(epochs=400, beta=0.0), M=5, base_seed=0)",
    "print('surrogate trained (45 cells, M=5, 400ep).')"))

C.append(new_markdown_cell(
    "## (A) M1 — the differentiable path is correct\n"
    "The analytic featurizer reproduces the discrete grid pipeline, and autodiff d(strength)/d(theta) "
    "matches finite difference — so gradient-based inverse design is differentiating the real surrogate."))

C.append(code(
    '"""(2) M1: featurizer agreement + autodiff vs finite difference."""',
    "d0, c0 = 0.6, 50.0",
    "Xn_a, _, Xd_a = idz.features_theta(torch.tensor(d0, dtype=torch.float64, device=DEVICE),",
    "                                   torch.tensor(c0, dtype=torch.float64, device=DEVICE))",
    "Xn_g, _, Xd_g = idz.features_grid(d0, c0)",
    "desc_err = float(torch.norm(Xd_a - Xd_g) / torch.norm(Xd_g))",
    "node_err = float(torch.norm(Xn_a - Xn_g) / torch.norm(Xn_g))",
    "depth = torch.tensor(0.55, dtype=torch.float64, device=DEVICE, requires_grad=True)",
    "contrast = torch.tensor(45.0, dtype=torch.float64, device=DEVICE, requires_grad=True)",
    "s = idz.differentiable_strength(ens, depth, contrast); s.backward()",
    "g_ad = np.array([float(depth.grad), float(contrast.grad)])",
    "h = 1e-4",
    "sv = lambda d, c: float(idz.differentiable_strength(ens, torch.tensor(d, dtype=torch.float64, device=DEVICE),",
    "                                                    torch.tensor(c, dtype=torch.float64, device=DEVICE)).detach())",
    "g_fd = np.array([(sv(0.55+h,45.)-sv(0.55-h,45.))/(2*h), (sv(0.55,45.+h)-sv(0.55,45.-h))/(2*h)])",
    "GRAD_ERR = float(np.linalg.norm(g_ad - g_fd) / (np.linalg.norm(g_fd) + 1e-12))",
    "M1_PASS = bool(desc_err < 0.02 and node_err < 0.05 and GRAD_ERR < GRAD_TOL)",
    "print(f'featurizer agreement: descriptor {desc_err:.4f}, node-feature {node_err:.4f}')",
    "print(f'autodiff vs finite-diff grad rel err {GRAD_ERR:.2e} (ad={g_ad}, fd={g_fd})')",
    "print(f'  -> M1 {\"PASS\" if M1_PASS else \"FAIL\"}')"))

C.append(new_markdown_cell(
    "## (B) The target battery — interior, boundary, and incoherent\n"
    "Targets are built from the surrogate's strength at known interior reference thetas (so an "
    "in-distribution solution provably exists), plus boundary targets that push theta off-manifold "
    "(where the surrogate is exploitable) and incoherent targets beyond the achievable range."))

C.append(code(
    '"""(3) B: achievable ranges + the inverse-design sweep (cached real verification)."""',
    "flo, fhi = idz.achievable_range(ens, *idz.FAMILY_BOX)",
    "wlo, whi = idz.achievable_range(ens, *idz.DEPTH_BOX, *idz.CONTRAST_BOX)",
    "print(f'family strength range [{flo:.3f}, {fhi:.3f}]; wide-box reachable [{wlo:.3f}, {whi:.3f}]')",
    "# interior targets: surrogate strength at reference (depth, contrast) well inside the family",
    "interior_refs = [(d, c) for d in (0.30, 0.45, 0.60, 0.75) for c in (35.0, 65.0)]",
    "def surr_at(d, c):",
    "    with torch.no_grad():",
    "        return float(idz.differentiable_strength(ens, torch.tensor(d, dtype=torch.float64, device=DEVICE),",
    "                                                 torch.tensor(c, dtype=torch.float64, device=DEVICE)))",
    "interior_targets = sorted(set(round(surr_at(d, c), 4) for d, c in interior_refs))",
    "# boundary targets: reachable only by going OFF the family manifold (the surrogate-exploit zone)",
    "boundary_targets = [round(fhi + 0.4 * (whi - fhi), 4), round(fhi + 0.8 * (whi - fhi), 4)]",
    "# incoherent targets: beyond the WIDE-box reach -> truly unachievable",
    "incoherent_targets = [round(whi + 0.3 * (whi - wlo), 4), round(wlo - 0.3 * (whi - wlo), 4)]",
    "",
    "def run(target):",
    "    return idz.inverse_design(float(target), ens, cache_dir=CACHE, iters=200)",
    "interior = [run(t) for t in interior_targets]",
    "boundary = [run(t) for t in boundary_targets]",
    "incoherent = [run(t) for t in incoherent_targets]",
    "print(f'interior {len(interior)}, boundary {len(boundary)}, incoherent {len(incoherent)} candidates')"))

C.append(new_markdown_cell(
    "## (M2) Convergence + (M3) verified-candidate fidelity\n"
    "Convergence: in-range targets reach the surrogate target. Fidelity: on the candidates the gate "
    "marks **verified**, the REAL damage-DNS matches the target within the V2.4 in-family tolerance — "
    "they are verified candidates, not oracle outputs."))

C.append(code(
    '"""(4) M2 + M3."""',
    "in_range = interior + boundary",
    "CONV = float(np.mean([abs(c.surrogate_pred - c.target) <= SURR_TOL for c in in_range]))",
    "M2_PASS = bool(CONV >= CONV_MIN)",
    "verified = [c for c in interior if c.status == 'verified']",
    "real_rel = [abs(c.verified_real - c.target) / abs(c.target) for c in verified]",
    "M3_MED = float(np.median(real_rel)) if real_rel else float('nan')",
    "M3_PASS = bool(len(verified) >= 1 and M3_MED < REAL_TOL)",
    "print(f'M2 convergence: {CONV*100:.0f}% of in-range targets reach the surrogate target (>= {CONV_MIN*100:.0f}%)')",
    "print(f'M3 verified set: {len(verified)}/{len(interior)} interior targets verified; '",
    "      f'median real rel err {M3_MED:.3f} (< {REAL_TOL}); p90 {np.percentile(real_rel,90):.3f}')",
    "print(f'  -> M2 {\"PASS\" if M2_PASS else \"FAIL\"}; M3 {\"PASS\" if M3_PASS else \"FAIL\"}')"))

C.append(new_markdown_cell(
    "## (M4) The verified-candidate gate — candidate, not oracle\n"
    "Incoherent targets must be reported impossible (never fabricated). And the trust signal "
    "(distance-to-manifold) must rank-correlate with the real-operator error, so untrusted candidates "
    "are caught *before* paying for the real solve — the spatial analogue of V2.1's calibrated handoff."))

C.append(code(
    '"""(5) M4: incoherent -> impossible (0 fabricated); trust predicts the real error."""',
    "INC_OK = all(c.status == 'impossible' for c in incoherent)",
    "# never a fabricated success: no incoherent candidate is reported 'verified'",
    "NO_FABRICATION = all(c.status != 'verified' for c in incoherent)",
    "graded = [c for c in (interior + boundary) if c.verified_real is not None]",
    "tr = [c.trust for c in graded]",
    "er = [abs(c.verified_real - c.target) / abs(c.target) for c in graded]",
    "RHO = float(spearmanr(tr, er).correlation) if len(graded) > 2 else float('nan')",
    "M4_PASS = bool(INC_OK and NO_FABRICATION and RHO >= TRUST_RHO_MIN)",
    "print(f'incoherent statuses = {[c.status for c in incoherent]} -> all impossible: {INC_OK}')",
    "print(f'trust (distance) vs real error Spearman = {RHO:+.3f} (>= {TRUST_RHO_MIN}) over {len(graded)} candidates')",
    "print(f'  -> M4 {\"PASS\" if M4_PASS else \"FAIL\"} (verified candidate, not oracle)')"))

C.append(new_markdown_cell(
    "## (M5) Physiology inverse + (M6) survival-spectrum impossibility\n"
    "Invert the regulator's setpoint to a target resting homeostatic pressure (implicit-diff Newton), "
    "verified by the independent brute-force basin oracle + viability margin. A world whose reserve is "
    "below `r_critical` returns a precise impossibility — 'can this creature survive here' derived."))

C.append(code(
    '"""(6) M5 + M6: homeostatic inverse, basin verification, and the survival spectrum."""',
    "base = reg.RegulatorParams()",
    "rc = reg.r_critical(base)",
    "P_base = idz.homeostatic_pressure(base, base.r0)",
    "P_targets = [P_base * f for f in (0.85, 1.0, 1.15)]",
    "phys = [idz.inverse_homeostasis(P, base, base.r0) for P in P_targets]",
    "phys_err = [abs(c.surrogate_pred - P) / P for c, P in zip(phys, P_targets)]",
    "M5_PASS = bool(all(c.status == 'verified' for c in phys) and max(phys_err) < PHYS_TOL)",
    "dead = idz.inverse_homeostasis(P_base, base, 0.5 * rc)",
    "r_grid = np.linspace(0.05, base.r0, 12)",
    "spectrum = idz.survival_spectrum(base, r_grid)",
    "M6_PASS = bool(dead.status == 'impossible' and spectrum[0] <= 1e-9 and spectrum[-1] > 0)",
    "print(f'M5 homeostatic inverse: targets {[round(P,3) for P in P_targets]} -> achieved '",
    "      f'{[round(c.surrogate_pred,3) for c in phys]}, basin_area {[round(c.verified_real,2) for c in phys]}, '",
    "      f'margin {[round(c.trust,2) for c in phys]}; max rel err {max(phys_err):.4f} (< {PHYS_TOL})')",
    "print(f'M6 world r={0.5*rc:.3f} < r_crit={rc:.3f} -> {dead.status}; survival spectrum basin_area '",
    "      f'{spectrum[0]:.2f}..{spectrum[-1]:.2f} (contracts to 0 below r_crit)')",
    "print(f'  -> M5 {\"PASS\" if M5_PASS else \"FAIL\"}; M6 {\"PASS\" if M6_PASS else \"FAIL\"}')"))

C.append(new_markdown_cell("## (F) Figure"))

C.append(code(
    '"""(7) Figure -> results/V2_5_inverse_design.png (2x3)."""',
    "fig, ax = plt.subplots(2, 3, figsize=(15.5, 9))",
    "allm = interior + boundary",
    "col = {'verified': 'seagreen', 'untrusted_rve': 'darkorange', 'impossible': 'crimson'}",
    "# [0,0] convergence: surrogate achieved vs target",
    "ax[0,0].plot([flo, whi], [flo, whi], 'k--', lw=0.8)",
    "for c in allm:",
    "    ax[0,0].scatter(c.target, c.surrogate_pred, color=col.get(c.status,'gray'), s=30)",
    "ax[0,0].set_xlabel('target strength'); ax[0,0].set_ylabel('surrogate achieved')",
    "ax[0,0].set_title(f'(M2) convergence: {CONV*100:.0f}% hit target')",
    "# [0,1] verified gap: surrogate-pred vs REAL-verified (the candidate-not-oracle panel)",
    "gv = [c for c in allm if c.verified_real is not None]",
    "for c in gv:",
    "    ax[0,1].scatter(c.surrogate_pred, c.verified_real, color=col.get(c.status,'gray'), s=30)",
    "mn = min([c.verified_real for c in gv] + [c.surrogate_pred for c in gv])",
    "mx = max([c.verified_real for c in gv] + [c.surrogate_pred for c in gv])",
    "ax[0,1].plot([mn,mx],[mn,mx],'k--',lw=0.8)",
    "ax[0,1].set_xlabel('surrogate prediction'); ax[0,1].set_ylabel('REAL damage-DNS')",
    "ax[0,1].set_title(f'(M3) verified candidate, not oracle\\nmedian real err {M3_MED:.3f}')",
    "# [0,2] the gate: trust (distance) vs real error",
    "ax[0,2].scatter(tr, er, c='crimson', s=30)",
    "ax[0,2].set_xlabel('trust = distance-to-manifold'); ax[0,2].set_ylabel('real-operator rel error')",
    "ax[0,2].set_title(f'(M4) the gate: rho={RHO:.2f}\\ntrust predicts the surrogate-real gap')",
    "# [1,0] achievable range + targets (impossible region shaded)",
    "ax[1,0].axvspan(whi, max(incoherent_targets)+0.05, color='crimson', alpha=0.12, label='out of reach')",
    "for c in allm + incoherent:",
    "    ax[1,0].scatter(c.target, 0.0, color=col.get(c.status,'gray'), s=30)",
    "ax[1,0].axvline(flo, ls=':', c='gray'); ax[1,0].axvline(fhi, ls=':', c='gray')",
    "ax[1,0].axvline(whi, ls='--', c='crimson', lw=0.8)",
    "ax[1,0].set_xlabel('target strength'); ax[1,0].set_yticks([])",
    "ax[1,0].set_title('(M4) incoherent targets reported impossible'); ax[1,0].legend(fontsize=8)",
    "# [1,1] physiology inverse: target vs achieved homeostatic pressure",
    "ax[1,1].plot(P_targets, P_targets, 'k--', lw=0.8)",
    "ax[1,1].scatter(P_targets, [c.surrogate_pred for c in phys], color='seagreen', s=40)",
    "ax[1,1].set_xlabel('target homeostatic P'); ax[1,1].set_ylabel('achieved (verified in-basin)')",
    "ax[1,1].set_title('(M5) survival-spectrum inverse\\n(regulator setpoint -> resting state)')",
    "# [1,2] survival spectrum: basin area vs reserve, impossibility boundary at r_crit",
    "ax[1,2].plot(r_grid, spectrum, 'o-', color='steelblue')",
    "ax[1,2].axvline(rc, ls='--', c='crimson', label=f'r_critical={rc:.2f}')",
    "ax[1,2].set_xlabel('reserve r'); ax[1,2].set_ylabel('basin area (viability)')",
    "ax[1,2].set_title('(M6) survival spectrum:\\nviability contracts to impossibility'); ax[1,2].legend(fontsize=8)",
    "fig.tight_layout()",
    'outdir = REPO / "verification_notebooks" / "phase2" / "results"; outdir.mkdir(exist_ok=True)',
    'fig.savefig(outdir / "V2_5_inverse_design.png", dpi=110)',
    "print('figure saved')"))

C.append(new_markdown_cell("## (G) Frozen verdict"))

C.append(code(
    '"""(8) Verdict — all metrics vs frozen thresholds; PASS marks Tier 2 COMPLETE."""',
    "def verdict(name, ok, detail):",
    "    print(f\"  [{'OK ' if ok else 'XX '}] {name:32s} {'PASS' if ok else 'FAIL'}\")",
    "    print(f'        {detail}')",
    "print('=' * 76)",
    "print('V2.5 — INVERSE DESIGN CONVERGENCE & CANDIDATE VERIFICATION — VERDICT')",
    "print('=' * 76)",
    "verdict('M1 differentiable path', M1_PASS, f'autodiff vs FD {GRAD_ERR:.1e} (< {GRAD_TOL})')",
    "verdict('M2 convergence', M2_PASS, f'{CONV*100:.0f}% of in-range targets hit (>= {CONV_MIN*100:.0f}%)')",
    "verdict('M3 verified fidelity', M3_PASS, f'median real err {M3_MED:.3f} (< {REAL_TOL}) on {len(verified)} verified')",
    "verdict('M4 candidate-not-oracle gate', M4_PASS, f'incoherent->impossible {INC_OK}; trust rho {RHO:.2f} (>= {TRUST_RHO_MIN})')",
    "verdict('M5 physiology inverse', M5_PASS, f'max rel err {max(phys_err):.4f} (< {PHYS_TOL}); all verified in-basin')",
    "verdict('M6 survival impossibility', M6_PASS, f'sub-r_crit -> {dead.status}; spectrum {spectrum[0]:.2f}..{spectrum[-1]:.2f}')",
    "CORE_CLAIM = bool(M1_PASS and M2_PASS and M3_PASS and M4_PASS and M5_PASS and M6_PASS)",
    "print('-' * 76)",
    "if CORE_CLAIM:",
    "    print('V2.5 VERDICT: PASS — inverse design converges through the differentiable surrogate; the')",
    "    print('  candidate, re-simulated with the REAL operators, matches the target (a VERIFIED CANDIDATE,')",
    "    print('  not an oracle); the trust gate (distance-to-manifold) predicts the surrogate-real gap so')",
    "    print('  off-manifold exploits are caught and routed to RVE; and incoherent targets -- material')",
    "    print('  (out of achievable range) and physiological (a world below r_critical) -- are reported as')",
    "    print('  a PRECISE IMPOSSIBILITY, not fabricated. The survival-spectrum (can this creature survive')",
    "    print('  in world X) is derived, not authored. Decision #18 holds. ** TIER 2 COMPLETE. **')",
    "else:",
    "    print('V2.5 VERDICT: see failed metric(s) above (REDESIGN parameterization / CONSTRAIN the gate).')",
    "assert CORE_CLAIM, 'V2.5 did not establish convergence + verified-candidate + impossibility.'"))

nb.cells = C
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
out = pathlib.Path(__file__).parent / "V2_5_inverse_design.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
