# Nebula — Tier 1 Verification Report

**Status: V1.1, V1.2, and V1.3 PASS — the Phase-0 gate (V1.1–V1.3) is COMPLETE.**
Per the protocol's exit criteria (§9), the Phase-0 tree slice may begin once Tier 0 (✅ complete — see [`../../phase0/results/tier0_report.md`](../../phase0/results/tier0_report.md)) **and** V1.1–V1.3 pass — **both conditions are now met, so the Phase-0 tree slice is unblocked.** This report covers Tier 1's operator-mechanism verifications.

> **Folder convention.** `verification_notebooks/phaseN` = verification **Tier N**, *not* architecture "Phase N". This is Tier 1.

Each verification follows the protocol discipline: one falsifiable claim, an **independent oracle** obtained a different way, and **pre-registered pass criteria frozen before running**. A failed verification is a *result*, not a setback.

| # | Verification | Claim (one line) | Verdict |
|---|---|---|---|
| V1.1 | Composition order & cascade | additive contributions compose order-free; declared cascade makes transitions deterministic | ✅ PASS |
| V1.2 | Operator-split stability | the additive split is consistent (order 1) and the stiff fire loop is stable at production steps via rate-refinement sub-stepping | ✅ PASS |
| V1.3 | Jensen sub-cell variance | mean-only lumping under-estimates nonlinear rates and silently extinguishes a burn; the variance correction recovers it, and ε is the refine trigger | ✅ PASS |

**Environment.** Python 3.13 + NumPy/SciPy/Matplotlib (`.venv`). V1.1–V1.3 are **CPU/NumPy by design**: the fire fields are tiny (6³–24³ cells) and the cost is a *sequential* chain of adaptive sub-steps (latency-bound, not throughput-bound), so GPU dispatch would add per-op launch overhead with nothing to parallelize across — the protocol tags `monolithic_fire` as "pure numpy (fast at N≤24)". (The Tier-0 GPU oracles — cupy CG FEM, Warp octree — were genuinely throughput-bound.) Notebooks run headless via `jupyter nbconvert --execute`; figures saved alongside this report.

**Shared oracle modules** (`src/verification/oracles/`): `fire_operators.py`, `bus_runtime.py`, `monolithic_fire.py`, `multirate.py`, **`semi_implicit_fire.py`** (new for V1.2), **`jensen_rate.py`** (new for V1.3), `transitions.py`, `determinism.py`.

---

## V1.1 — Operator composition order-independence & cascade determinism

**Targets:** Decision #12 — the *contributions-vs-transitions* split. **Depends on** V0.3 (the conserved-bus runtime); ties to V0.5 (reduction-order determinism).

**Approach.** Additive contributions staged into conserved buses must commit identically under any operator ordering ("phenomena emerge by free composition"); competing **transitions** on the same variable must be deterministic only under a declared cascade priority.

**Results.** Contribution order-independence holds across all **120** orderings (float divergence below `SOLVER_TOL=1e-9`; **bit-identical** under fixed-order / integer-exact reduction — the V0.5 tie-in). Transitions are order-**ambiguous** without the cascade (≥2 distinct committed maps) and **deterministic with** it (exactly 1 map, equal to the highest-priority-firing reference), in both the categorical-`phase` and mass-`fate` variants. Notebook: `V1_1_composition_order.ipynb`; figure `V1_1_composition_order.png`. **PASS.**

---

## V1.2 — Operator-splitting stability for stiff cyclic coupling

**Targets:** Decision #12 on the flagship stiff phenomenon — the **char ↔ conduction ↔ pyrolysis ↔ combustion** loop; rate-refinement as the multi-rate handler. **Depends on** V0.3 (conserved-bus runtime + monolithic oracle).

**The mechanism, stated precisely.** Nebula stages **all** operators at the **same time level** and commits once (`new = old + Σ contribsᵢ·dt`), so the production "split" (`bus_runtime.step_split`) is **plain forward Euler on `fire_operators.coupled_rhs` — there is no Lie/Strang commutator error term.** Two consequences frame the test: (1) consistency is pure temporal discretization → the split must converge to the monolithic reference at **order 1**; (2) the entire difficulty is **stiff-explicit stability** — the Arrhenius rate constants are O(10²), so past an explicit step limit a reactant is over-subscribed, the commit clamps it to zero, and the integrated burn outcome is **silently corrupted** (not blown up). The architecture's fix is the refinement predicate's **rate term → a finer local timestep** (`multirate.step_split_substep`).

**Implementation.**
- `monolithic_fire.py` — the §7 oracle (adaptive sub-stepped RK4, no splitting), validated vs scipy stiff Radau in V0.3.
- `multirate.py` — the mechanism under test (rate-driven sub-stepping); left **untouched** from V0.3/V1.1.
- **`semi_implicit_fire.py` (new)** — the protocol's named **REDESIGN** alternative: a linearly-implicit (IMEX) split treating the stiff reactant depletion implicitly (`m_s^{n+1}=m_s^n/(1+k_py·dt)`, combustion likewise, O₂-limited), so a reactant can never overshoot — unconditionally stable on the reaction terms, no clamp. Self-check: reactant non-negativity at `dt` up to **100** without clamping; first-order convergence to the monolithic oracle on a single cell.
- Notebook: `V1_2_split_stability.ipynb`; figure `V1_2_split_stability.png`.

**Verification.** Three-scene battery (frozen): **gentle** (forward Euler stable across the sweep — isolates convergence order), **stiff** (bounded burn, self-limiting as core O₂ depletes), **near-runaway** (hotter, fuel-richer ignition pocket). The monolithic oracle is the reference for every split path; the stability sweep computes it **once per scene** (mass-consumed and char are sampling-independent). Pre-registered criteria: (A) convergence order ∈ [0.9, 1.3]; (B) naive single-step **>5%** corrupted at production `dt` (necessity); (C) sub-stepped split **<5%** on mass/peak-T/char; (D) semi-implicit split **<5%** in a single step/global; (E) sub-stepped & semi-implicit stable step **≥5×** the naive limit.

**Results.**
- **A — convergence order 1.035** (errors halve from 1.6e-4 → 9.1e-6 as `dt` halves) — confirms the additive split is consistent with **no commutator error**.
- **B — naive corrupted:** integrated **char** error **266%** (stiff) / **242%** (runaway) at the production `dt` (peak-T also off by 15.5% / 6.5%); the burn stays finite but the clamps silently distort it — the stiff loop genuinely needs a fix.
- **C — sub-stepping fix:** **0.02% / 0.02%** max outcome error (72 / 120 sub-steps) — the CONSTRAIN path holds.
- **D — semi-implicit fix:** **0.24% / 0.30%** max outcome error in a single step per global step — the REDESIGN path *lifts* the sub-stepping requirement.
- **E — stability boundary:** naive ceiling **6.8e-3**; sub-stepped and semi-implicit both **≥8.0e-2** → **11.7×** the naive limit.

**Verdict: PASS.** The conserved-bus compose-on-shared-state model is consistent (order 1) and reliable for stiff couplings *with* the multi-rate handler. **Standing CONSTRAINT (documented):** the plain explicit split requires rate-driven sub-stepping — or the validated semi-implicit treatment — on the stiff char↔conduction↔pyrolysis loop; this is wired through the refinement predicate's rate term, so it is not new machinery. Decision #12 holds for stiff coupling.

---

## V1.3 — Nonlinear-rate lumping: the Jensen sub-cell-variance correction

**Targets:** Decision #16; ARCHITECTURE §III.4 "the nonlinear trap". **Depends on** V0.1 (the homogenization cell).

**The trap, stated precisely.** A homogenized cell carries one mean temperature `T̄`, but the reaction rate is the Arrhenius law `g(T)=exp(−Ta/T)`, which is **convex** over the physical range (`Ta/T ≫ 2`). By Jensen's inequality `⟨g(T)⟩ ≥ g(T̄)`: lumping at the mean **systematically under-estimates** the rate, and in a coupled burn that error can **silently extinguish the fire**. The architecture's fix carries the sub-cell **variance** and applies the second-order correction `g_corr = g(T̄) + ½·g″(T̄)·σ²_T` (`g″>0 ⇒ correction>0`). This is the **second tracked homogenization error** — the *variance* term that joins V0.1's Voigt–Reuss *responses* term; the dimensionless `ε = ½σ²|g″/g|` is its refine-trigger scalar.

**Implementation.**
- **`jensen_rate.py` (new)** — the Arrhenius value/derivatives (`g`,`g1`,`g2`), the three estimators (`mean_only_rate`, `true_mean_rate` = the fine-scale oracle `A·⟨g(T)⟩`, `variance_corrected_rate`), the `variance_error_scalar` ε, a 3-profile sub-cell **T-field battery** (`ramp_field` linear hot-face/cold-core, `boundary_layer_field` exponential skin, `bimodal_field` hot sub-volume in a cold matrix), and a lumped/fine coupled mini-burn pair for the extinction demo (`run_lumped`, `run_fine`). Imports `fire_operators` constants only; leaves it (and the V1.2 modules) **untouched**. The fine reference reuses the V1.2-validated `multirate` substepper (every sub-voxel resolved, no homogenization).
- Notebook: `V1_3_jensen_variance.ipynb`; figure `V1_3_jensen_variance.png`.

**Verification.** 3-profile battery × a hot-face sweep (cold core 600 K, faces 630→1050 K, ΔT 30→450 K). The fine-scale rate integral is the oracle. Pre-registered criteria (frozen): (A1) mean-only **>50%** under-estimate at the documented steep face, all profiles; (A2) variance-corrected error **<10%** for cells with `ε < ε*` (`ε*=0.5`); (B) `ε>ε*` catches **100%** of >10%-error cells (no false negatives); (C) in a cool-mean/hot-face mini-burn, mean-only burns **<25%** of the fine truth's fuel (extinguished) while corrected burns **>50%** (sustained).

**Results.**
- **A1 — the hazard:** mean-only under-estimates the true rate by **57% / 83% / 93%** (ramp / boundary-layer / bimodal) at the steep face — Jensen sign and magnitude confirmed.
- **A2 — the correction:** variance-corrected error **≤ 5.3%** for every cell with `ε<0.5` (< 10%). The linear **ramp** is benign (symmetric ⇒ 2nd-order nearly exact, ≤5% throughout); non-Gaussian profiles break sooner.
- **B — the refine trigger:** `ε>ε*=0.5` catches **100%** of cells whose corrected error exceeds 10% (minimum breakdown `ε=0.60` > `ε*`); the **bimodal** profile (largest higher moments) breaks earliest.
- **C — spurious extinction:** in the cool-mean/hot-face cell, mean-only consumes only **4.6%** of the fine truth's fuel (the burn is silently extinguished) while the variance-corrected cell consumes **92.7%** — tracking the fine-scale truth.

**Verdict: PASS.** The Jensen correction recovers nonlinear rates over a documented validity region, mean-only lumping demonstrably kills a real burn, and `ε` is the wired refine trigger past the 2nd-order edge. **Standing CONSTRAINT (documented):** carry sub-cell variance for nonlinear-rate cells; refine when `ε > ε*` — the variance homogenization error, companion to the Voigt–Reuss gap.

---

## Phase-0 status & remaining Tier 1
**The Phase-0 gate (V1.1–V1.3) is COMPLETE** — together with the passing Tier 0, the §9 exit criteria for the **Phase-0 tree slice are met**: the substrate, the homogenization currency, conservation/composition, scaling, determinism, order-free composition, stiff-split stability, and nonlinear-rate lumping are all verified. The remaining Tier-1 verifications gate *later* phases, not Phase 0: **V1.4** (coupling-operator pipeline) gates authoring work and rests on the already-proven symmetry core; **V1.5–V1.6** gate living-asset work; **V1.7–V1.9** (skeleton precipitation, growth write-back, dual-cloud) gate their own subsystems.
