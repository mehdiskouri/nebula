# Nebula — Tier 1 (V1) Handoff

**Purpose.** A session-to-session handoff for the **Tier-1 verification** work (the `V1.x`
notebooks). It records what is done, the headline results, how the work is structured, and exactly
what remains — **V1.7, V1.8, V1.9** — so the next session can pick up cold.

**One-line status.** Tier 0 ✅ (V0.1–V0.5). **Tier 1 phase-gates ✅ (V1.1–V1.6).** Remaining:
**V1.7–V1.9**, which gate their *own* subsystems and are **not** exit-criteria for any phase
already unblocked.

> Read order for a newcomer: `AGENT/ARCHITECTURE.md` → `AGENT/verification_protocol.md` →
> `verification_notebooks/phase0/results/tier0_report.md` →
> `verification_notebooks/phase1/results/tier1_report.md` → this file.

---

## 1. What Nebula is verifying, and the discipline

Nebula is a simulation-first 3D creation language (assets are *grown/simulated*, not modeled).
Before committing engineering effort, every load-bearing architectural assumption is **falsified**
in a pre-registered notebook with: one falsifiable claim, an **independent oracle** computed a
different way, and **pass criteria frozen before the notebook runs**. A red result is a *result*
(it redirects the design while it's cheap), not a setback. Folder convention:
`verification_notebooks/phaseN` = verification **Tier N** (not architecture "Phase N").

**Exit criteria (protocol §9), and where we stand:**
- **Phase-0 tree slice** — needs Tier 0 + V1.1–V1.3. ✅ **unblocked.**
- **Authoring / spectral (arch. Phase 1)** — needs V1.4 (on the proven symmetry core). ✅ **unblocked.**
- **Living things (arch. Phase 2)** — needs V1.5 + V1.6. ✅ **unblocked.**
- Tier 2 scale/exotic work is gated on V2.1/V2.4 (not started; out of scope for this handoff).

---

## 2. Done — Tier 0 (foundational invariants)

**All five PASS (V0.1–V0.5).** Homogenization bound (Voigt–Reuss containment + tightness),
criticality coincidence, conservation-under-composition + composite-OOD detection, complexity
scaling (`O(n_active·log n)`), and determinism under GPU float non-associativity. Details:
`verification_notebooks/phase0/results/tier0_report.md`. These are the substrate the Tier-1 work
builds on; **guard against regression, do not re-verify.**

---

## 3. Done — Tier 1 (V1.1–V1.6)

Full prose + numbers: `verification_notebooks/phase1/results/tier1_report.md`. Summary:

| # | Claim (one line) | Verdict | Headline results |
|---|---|---|---|
| **V1.1** | additive contributions compose order-free; declared cascade makes transitions deterministic | ✅ | order-independent across **120** orderings (div < 1e-9, bit-identical fixed-order); transitions ambiguous without the cascade, deterministic with it |
| **V1.2** | additive operator split is consistent (order 1); the stiff fire loop is stable at production steps via rate sub-stepping | ✅ | convergence order **1.035**; naïve split corrupts char by **266%** at production dt; sub-stepping **0.02%**, semi-implicit **0.24%**; stable step **11.7×** the naïve limit |
| **V1.3** | mean-only lumping under-estimates nonlinear rates & silently extinguishes a burn; variance correction recovers it; ε is the refine trigger | ✅ | mean-only under-estimates **57/83/93%**; corrected ≤ **5.3%** for ε<0.5; ε* catches **100%** of >10% cells; extinction demo **4.6%** vs **92.7%** fuel |
| **V1.4** | the lift→GFT→reconstruct pipeline reproduces the silhouette; truncation is graceful LOD; irrep lock guarantees symmetry; C¹ quilt removes seams | ✅ | IoU **0.939 / 0.919** (biped/seraph); truncation monotone, max IoU dip **0.015**; macro/micro **37.9×**; symmetry-lock residual **1.9e-15 / 9.2e-16**; seam residual **~1e-13** |
| **V1.5** | regulated loop recovers small perturbations & dies past a reserve-dependent saddle-node; viability margin = exact basin | ✅ | `r_crit`=**0.237**; critical-bleed monotone **0.76→0.17→0**; margin vs brute-force basin **100%**; saddle eig **+3.10**; fatal vs survivable bleed shown |
| **V1.6** | passivity (reserve-dissipating) actuation has a strictly larger oscillation-free gain region than a naïve force controller | ✅ | naïve Hopf **1.487** (eig) vs **1.538** (nonlinear), 3.5% err; `K_hopf` passive/naïve **2.47×**; 2-D area **0.21→0.66**; production box **28/28** naïve trembles vs **0/28** passive |

**Net:** the substrate, homogenization currency, conservation/composition, scaling, determinism,
order-free composition, stiff-split stability, nonlinear-rate lumping, the spectral coupling
pipeline, emergent mortality, and regulator passivity are all verified.

---

## 4. Repo layout & how to run

- **Oracles** (`src/verification/oracles/`, throwaway ground truth): `fire_operators.py`,
  `bus_runtime.py`, `monolithic_fire.py`, `multirate.py`, `semi_implicit_fire.py` (V1.2),
  `jensen_rate.py` (V1.3), `regulator.py` (V1.5), `regulator_stability.py` (V1.6),
  `transitions.py`, `determinism.py`, `homogenization.py`, `analytic.py`, `dns_elasticity_3d.py`,
  `cells.py`, `octree.py`, `octree_gpu.py`, `failure.py`. Each has a `__main__` self-check.
- **Coupling pipeline** (`src/verification/`): `symmetry_basis.py` + `coupling_pipeline.py` (V1.4),
  on the untouched §8 baseline `coupling_operator_core.py` (Z₂) / `coupling_operator_c6.py` (C₆).
- **Notebooks**: `verification_notebooks/phase1/V1_{1..6}_*.ipynb`; figures + report in
  `verification_notebooks/phase1/results/`.

**Environment.** Python 3.13 venv at `/workspace/nebula/.venv` (NumPy/SciPy/Matplotlib + jupyter).
All of V1.1–V1.6 are **CPU/NumPy/SciPy by design** (tiny fields, sequential adaptive sub-steps,
small ODE phase-space analyses — latency-bound; GPU buys nothing here). torch/cupy/warp/taichi are
present for later GPU work but unused by Tier 1.
- ⚠️ **The `.venv` was wiped once by an instance recycle** — if imports fail (`No module named
  numpy`), the package environment was lost; reinstall numpy/scipy/matplotlib/jupyter into `.venv`.
- Run an oracle self-check: `.venv/bin/python src/verification/oracles/<name>.py`.
- Execute a notebook headless: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace
  verification_notebooks/phase1/V1_X_*.ipynb` (V1.6 is the slow one, ~5–6 min; others < 1 min).

**Conventions to follow for new V1.x (mirror V1.5/V1.6):**
1. Build a self-checking **oracle module** in `src/verification/oracles/` (rich docstring tying to
   protocol §/Decision/ARCHITECTURE §; pure numpy/scipy; a params dataclass; `__main__` asserts).
2. **Calibrate** with a scratch `_calib_vNN.py` (uncommitted), freeze thresholds *with margin*.
3. **Notebook** = header md (claim, oracle, frozen criteria table) → setup (frozen constants) →
   one experiment per criterion → multi-panel figure saved to `results/` → verdict cell ending in
   `assert ALL_PASS`. (For notebooks with f-strings, building via an `nbformat` script avoids
   JSON-escaping bugs — see how V1.6 was generated.)
4. Append a section to `tier1_report.md`, update the summary table + module list.
5. **Regression discipline:** never edit a passing oracle/notebook; re-run prior `__main__`
   self-checks after adding a module.

---

## 5. Remaining — V1.7, V1.8, V1.9

These three gate their **own subsystems** (skeleton, growth, dual-cloud); none blocks a phase
already unblocked in §1. Each below: claim, oracle, pre-registered pass bar, dependency, failure
class, and a suggested build path consistent with the conventions above.

### V1.7 — Skeleton precipitation under load  *(Decision #22; Wolff's law as a generative rule)*
- **Claim.** Stress-driven deposition under a creature's load case (gravity + self-weight + ability
  loads) yields a **connected, load-bearing** skeleton, and changing the load (low gravity; a
  radiance support law-domain) changes the skeleton accordingly.
- **Oracle.** Standard **topology optimization** (compliance-minimization, e.g. SIMP) on the same
  load/domain — a mature, trusted method.
- **Pass (pre-registered).** Produced structures fully connected and load-bearing (**compliance
  within 2×** of the topology-opt reference); morphology responds in the expected direction to load
  changes; the support-field (seraph) case precipitates **near-nothing**.
- **Failure → REDESIGN** (use topology optimization directly as the precipitation operator).
  **Depends on:** — (none). **Nature:** empirical/engineering.
- **Suggested build.** A 2-D/3-D linear-elastic FE grid (reuse/adapt the Tier-0 elasticity solver
  in `oracles/dns_elasticity_3d.py` for the compliance evaluation); a Wolff's-law deposition rule
  (add material where strain-energy density exceeds a threshold, remove where idle) vs a SIMP
  topology-opt reference; sweep Earth gravity / low gravity / support-field. Metrics: connectivity
  (graph components of the material set), compliance ratio, morphology sensitivity.

### V1.8 — Growth memoization & write-back correctness  *(Decision #11)*
- **Claim.** Evaluating a **memoized growth trace** at a given (time, LOD) is deterministic and
  matches a fresh growth run; a **write-back** (a recorded cut) correctly invalidates stale
  sub-results so later growth **heals around the wound** rather than replaying pre-wound state.
- **Oracle.** A from-scratch (non-memoized) growth run.
- **Pass (pre-registered).** memoized = fresh within determinism tolerance; **0 stale-replay
  incidents** across the write-back suite.
- **Failure → REDESIGN** the cache-key / invalidation scheme. **Depends on:** **V0.5** (determinism
  — reuse `oracles/determinism.py` patterns). **Nature:** theorem-check (determinism) + empirical
  (healing).
- **Suggested build.** A small seeded, field-biased **L-system** growth front (ARCHITECTURE §III.1)
  producing a `growth trace`; a memo cache keyed on (seed, params, time, LOD, **write-back state**);
  evaluate at several (time, LOD) points vs fresh; apply a cut → confirm the key includes the
  write-back, healing callus grows over the recorded cut, and no pre-wound replay. The whole thing
  is deterministic numpy; the crux is the **cache key must include write-back state** (the likely
  failure mode to probe).

### V1.9 — Dual-cloud skinning fidelity  *(Decision #6)*
- **Claim.** A dense **render cloud** skinned to a coarse **physics cloud** reproduces the
  deformation of a full-resolution physics sim within a visual-error tolerance, at a large
  node-count reduction.
- **Oracle.** A full-resolution physics sim where **every** render-detail point is itself simulated.
- **Pass (pre-registered).** Geometric error below tolerance at large deformations for a **≥10×**
  node reduction; error grows **gracefully** (no popping) as deformation increases.
- **Failure → CONSTRAIN** (cap supported deformation per coarse resolution; add adaptive
  physics-cloud refinement under extreme deformation). **Depends on:** — (none).
  **Nature:** empirical/engineering (quality-vs-cost).
- **Suggested build.** A coarse mass-spring / XPBD body (the physics cloud) driving a dense point
  set via linear-blend skinning weights; the oracle is the same dense set fully simulated. Sweep
  deformation severity × coarse/dense ratio; metrics: per-point geometric error vs full sim, a
  perceptual proxy, and the ratio achieved at tolerance. This is the one V1.x where a Taichi/Warp
  XPBD step could be justified if node counts get large, but a numpy XPBD at modest N is the simplest
  first pass.

---

## 6. Known issues / loose ends
- **`AGENT/v1_4plan.md` shows as deleted** in `git status` (not by me; carried across sessions) —
  restore or remove deliberately.
- `__pycache__/*.pyc` are tracked in the repo and show as modified on each run — pre-existing
  hygiene quirk, harmless; consider gitignoring `__pycache__/` and `_calib_v*.py` scratch files.
- Scratch calibration files `oracles/_calib_v1{5,6}.py` are intentionally uncommitted helpers.
- Nothing in this work has been committed yet; the V1.4/V1.5/V1.6 notebooks, figures, oracle
  modules, and the updated `tier1_report.md` are all unstaged.
