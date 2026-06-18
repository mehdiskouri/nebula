# Nebula — Verification Protocol

**Purpose.** Falsify the architecture's load-bearing assumptions *before* committing engineering effort to Phase 0. Every claim that, if false, would force redesign is mapped here to a planned **verification notebook** with an **independent oracle** and a **pass criterion fixed in advance**. This document specifies *what each notebook must do and how it is judged*; it deliberately contains **no notebook code**.

- **Status:** Protocol v0.1. Companion to `NEBULA_ARCHITECTURE.md` (cross-references use its Decision `#N` and Risk `#N` numbering).
- **Rule of engagement:** No subsystem enters Phase-0 implementation until its gating Tier-0 verifications pass. A failed verification is a *result*, not a setback — it redirects the design while it is still cheap to move.

---

## 1. Verification philosophy

Five disciplines apply to every notebook. They are the reason this protocol exists rather than ad-hoc testing during the build.

**Falsifiability.** Each verification states one claim that a measurement could prove *wrong*. "The homogenization bound is useful" is not a claim; "the homogenized estimate lies within the Voigt–Reuss gap for 100% of test cells and the gap is under 30% of mean stiffness for >80% of quiescent cells" is.

**Pre-registered pass criteria.** The numeric threshold is written down *before* the notebook runs. This prevents the universal failure mode of moving the goalposts to whatever the implementation happened to produce. Proposed thresholds below are first drafts to be reviewed and frozen before execution, not after.

**An independent oracle is mandatory.** A verification compares the *approximation we intend to ship* against a *ground truth obtained a different way*. The oracle is almost always slower, simpler, and unshippable — a direct fine-scale solve, a monolithic implicit integrator, an analytic closed form. Building these throwaway oracles is the bulk of the work and is the whole point: **the expensive reference exists so the cheap production path never has to be trusted on faith.**

**Claim nature is labeled.** Three kinds, because they are judged differently:
- *Theorem-check* — the claim is mathematically guaranteed; the notebook verifies the **implementation**, so the bar is exact (any violation is a bug).
- *Empirical* — the claim is about physical/numerical reality; the bar is a tolerance and a coverage fraction.
- *Engineering* — the claim is about a tradeoff (cost, stability margin, calibration); the bar is a Pareto target.

**Failure has a declared outcome class.** Decided in advance so a red result triggers a known response, not a debate:
- **KILL** — foundational; if false, the architecture cannot work as designed and needs fundamental rethink.
- **REDESIGN** — the *mechanism* fails but the *goal* is reachable another way; localized rework.
- **CONSTRAIN** — the claim holds only in a sub-regime; keep it, document the boundary, and add a guard (usually a refine trigger or an "unsupported here" flag).

---

## 2. Notebook template

Every entry below — and every notebook built from it — uses these fields.

- **Targets** — the assumption/risk and its architecture cross-reference.
- **Claim** — one falsifiable sentence.
- **Load-bearing because** — what downstream design collapses if it is false.
- **Nature** — theorem-check / empirical / engineering.
- **Oracle** — the independent ground truth and how it is obtained.
- **Design** — what to construct, what to vary (the sweep), what to hold fixed. Conceptual, not coded.
- **Metrics** — the quantities measured.
- **Pass criteria (pre-registered)** — the thresholds, frozen before running.
- **Failure → outcome** — KILL / REDESIGN / CONSTRAIN, with the specific fallback.
- **Depends on** — verifications that must pass first.

---

## 3. Sequencing & dependency overview

Three tiers, run roughly in order. Tier 0 gates everything; a KILL there stops the project to rethink. Tier 1 gates its own subsystem's Phase-0 work. Tier 2 are the known-hard risks; they may run in parallel with early building but block the *scale* and *exotic* claims.

```
TIER 0  (foundational — architecture is wrong if any KILL)
  V0.1 Homogenization bound  ──┬─► V0.2 Criticality coincidence
                               ├─► V2.2 Percolation (off-axis)
                               └─► V2.3 Geo/physical misalignment
  V0.3 Conservation+composite-OOD ─┬─► V1.1 Composition order
                                   ├─► V1.2 Split stability
                                   └─► V2.4 Surrogate generalization
  V0.4 Complexity scaling
  V0.5 Determinism

TIER 1  (subsystem mechanisms)
  V1.1, V1.2, V1.3 (operators)   V1.4 Coupling pipeline [extends proven core]
  V1.5 Regulator/mortality ──► V1.6 Regulator stability
  V1.7 Skeleton precipitation    V1.8 Growth write-back    V1.9 Dual-cloud fidelity

TIER 2  (hard risks)
  V2.1 RVE↔surrogate handoff [Risk #1]   V2.2 Percolation   V2.3 Geo/physical
  V2.4 Surrogate generalization+OOD       V2.5 Inverse design
```

**Shared oracles to build first** (throwaway scaffolding, listed in §7): a fine-scale DNS micro-solver, a monolithic implicit reference integrator, and an analytic-solution library for the handful of cases with closed forms.

---

## TIER 0 — Foundational invariants

### V0.1 — Homogenization bound validity & tightness *(the keystone)*
- **Targets:** Decision #15; the "one scalar, four jobs" claim. Risk-adjacent to #1.
- **Claim:** For a heterogeneous cell (bark + sapwood + heartwood, ± a char wedge), the homogenized effective-stiffness estimate lies *within* the Voigt–Reuss gap in every direction, and for quiescent (low-contrast) cells the gap is tight enough to be useful.
- **Load-bearing because:** This scalar gates refinement, conservation tolerance, surrogate trust, and LOD. If the bound is violated, the trust signal is meaningless; if it is always loose, everything refines and the efficiency thesis dies.
- **Nature:** Containment is a *theorem-check* (any violation = bug). Tightness is *empirical*.
- **Oracle:** Direct fine-scale solve (DNS) of the fully-resolved heterogeneous cell under prescribed strain/stress states → the *true* effective stiffness tensor.
- **Design:** Construct cells across a sweep of layer geometries and material contrasts (intact wood ↔ heavy char). For each, compute (a) the homogenized orthotropic proxy + V–R gap, (b) the DNS effective tensor. Compare per principal direction and in shear. Separately verify the *layered-medium exactness* claim: for clean concentric layers, the directional estimate should match DNS to solver tolerance along/around the layers.
- **Metrics:** signed position of DNS result within `[Reuss, Voigt]` per direction; relative gap width `(Voigt−Reuss)/mean`; principal-direction residual for layered cells.
- **Pass criteria (pre-registered):** DNS within gap for **100%** of cells (theorem). Layered principal-direction residual **< 1%** (exactness claim). Relative gap **< 30%** for **>80%** of low-contrast cells (usefulness).
- **Failure → outcome:** Containment violation → **KILL** (implementation or theory error; halt). Persistently loose gap → **REDESIGN** (replace V–R with tighter Hashin–Shtrikman bounds, or move to measured-RVE descriptors).
- **Depends on:** DNS micro-solver (§7).

### V0.2 — The criticality coincidence
- **Targets:** the claim that "worst case for homogenization = most important case for the sim," i.e. the gap blows open exactly when the cell is structurally critical.
- **Claim:** As a char wedge deepens, the V–R gap (trust scalar) degrades monotonically and crosses the refine threshold *at or before* the cell reaches structural criticality (load approaching cohesive capacity).
- **Load-bearing because:** This is what makes "refine where untrustworthy" automatically spend resolution where it matters. If the gap spikes *after* failure, the system refines too late.
- **Nature:** Empirical.
- **Oracle:** DNS of the charring cell under sustained load, tracking true distance-to-failure (stored elastic energy vs cohesive toughness).
- **Design:** Sweep char fraction χ from 0 → failure under a fixed load. Plot trust-scalar trajectory against true distance-to-failure. Check ordering of threshold-crossings.
- **Metrics:** χ at refine-threshold crossing vs χ at criticality onset; correlation between gap width and inverse distance-to-failure.
- **Pass criteria:** refine-trigger χ **≤** criticality-onset χ in **100%** of load cases; monotone gap-vs-χ (no late spike) in **>95%**.
- **Failure → outcome:** **CONSTRAIN** — add an explicit stress/energy criterion to the refinement predicate so criticality triggers refinement independently of the homogenization gap.
- **Depends on:** V0.1.

### V0.3 — Conservation under composition & composite-OOD detection
- **Targets:** Decisions #12, #14. The conserved-bus discipline and the "residual spike is the primary monitor" claim.
- **Claim:** With the four fire operators (combustion, conduction, pyrolysis, char-weakening) composed via staged contributions, every conserved bus audits to ≈0 in-distribution; and a composite off-distribution event (a lightning heat/charge impulse) produces a conservation-residual spike even when each operator's individual envelope still reports "in distribution."
- **Load-bearing because:** Composition's correctness and the surrogate-fallback trigger both rest on the audit seeing cross-terms that per-operator checks miss.
- **Nature:** Conservation is *theorem-check*; OOD-detection-via-residual is *empirical*.
- **Oracle:** A monolithic implicit solve of the coupled fire PDE system (no operator splitting) as the trajectory reference; closed-form total energy/mass budget as the conservation reference.
- **Design:** Run the composed in-distribution burn; audit energy/mass/O₂ ledgers each step. Then drive a lightning impulse past the regime any single operator's learned tier was trained on; log each operator's self-reported envelope status *and* the global residual. Confirm the residual flags the event when envelopes do not.
- **Metrics:** per-bus residual (in-distribution and during impulse); detection latency (steps from impulse to residual-threshold crossing); false-negative rate of per-operator envelopes on the composite event.
- **Pass criteria:** in-distribution residual **< 1e-6** of bus throughput (theorem). Residual detects the composite OOD with **0 false negatives** across the impulse sweep, where per-operator envelopes miss it in **≥1** case (proving necessity).
- **Failure → outcome:** Conservation drift → **KILL/REDESIGN** of the bus/ledger mechanism. Residual fails to detect → **REDESIGN** the monitor (add a learned composite-envelope alongside the residual).
- **Depends on:** monolithic reference integrator (§7).

### V0.4 — Complexity scaling
- **Targets:** Decision #8; the `O(n_active · log n_total)` efficiency thesis. Risk: global-activation degradation.
- **Claim:** Interaction-query and solve cost scales as `n_active · log n_total`, not `n_total` and not `n_active²`; and the structure degrades *gracefully* (toward dense cost, no cliff) as `n_active → n_total`.
- **Load-bearing because:** The entire feasibility argument. If scaling is wrong, nothing built on the hierarchy is affordable.
- **Nature:** Empirical (measured scaling exponent).
- **Oracle:** A brute-force all-pairs / fully-refined solve for small `n` to anchor correctness; the asymptotic shape is read from the cost curve itself.
- **Design:** Build the linearized-octree hierarchy with valid coarse proxies. Hold `n_active` fixed and sweep `n_total` (measure the log factor). Hold `n_total` fixed and sweep `n_active` (measure linearity). Then sweep `n_active/n_total → 1` (measure graceful degradation). Verify early-termination actually fires: count tree levels descended per query.
- **Metrics:** fitted exponents for cost vs `n_total` and vs `n_active`; mean descent depth vs `log n_total`; cost ratio at full activation vs the dense oracle.
- **Pass criteria:** cost vs `n_total` exponent consistent with log (sublinear; fitted power **< 0.2**); cost vs `n_active` linear (**0.9–1.1**); full-activation cost **≤ 1.5×** dense oracle (no superlinear cliff).
- **Failure → outcome:** Superlinear in `n_active` → **REDESIGN** the active-set bookkeeping. No log behavior (proxies not enabling early stop) → **KILL** of the self-similar-physics premise (re-examine V0.1-style proxy validity at every level).
- **Depends on:** V0.1 (coarse proxies must be valid for early termination to be sound).

### V0.5 — Determinism under GPU float non-associativity
- **Targets:** Decision #3; Risk: determinism vs float non-associativity.
- **Claim:** With fixed reduction orders, a Nebula computation produces bit-reproducible (or within a declared deterministic tolerance) results across repeated runs and across hardware; *without* fixing them, reductions visibly diverge.
- **Load-bearing because:** "The program is the asset," memoization, and reproducible verification all assume determinism.
- **Nature:** Engineering.
- **Oracle:** The computation's own repeated runs (self-consistency); a CPU fixed-order reference for the same reduction.
- **Design:** Take a representative reduction-heavy kernel (a bus reduction over many contributions). Run it (a) with default/atomic-order GPU reduction repeatedly, (b) with an enforced deterministic reduction order, across ≥2 GPU configurations. Measure divergence.
- **Metrics:** max bitwise/relative divergence across runs and devices, for each reduction strategy.
- **Pass criteria:** deterministic-order divergence **= 0** bitwise (or **< 1e-12** relative if a tolerance regime is adopted) across runs and devices; default-order divergence demonstrably **> 0** (confirming the hazard is real and the fix necessary).
- **Failure → outcome:** Cannot achieve determinism affordably → **CONSTRAIN** (declare a tolerance-based determinism regime and document that exact bit-reproducibility is not guaranteed cross-hardware), revisiting memoization-key assumptions accordingly.

---

## TIER 1 — Subsystem mechanisms

### V1.1 — Operator composition order-independence & cascade determinism
- **Targets:** Decision #12; contributions-vs-transitions split.
- **Claim:** Additive contributions yield the same committed state regardless of operator evaluation order; transitions on a shared variable are deterministic *only* with the declared cascade priority and ambiguous without it.
- **Load-bearing because:** "Phenomena emerge by free composition" requires order-independence; determinism requires the cascade.
- **Nature:** Theorem-check (contributions) + empirical demonstration (cascade necessity).
- **Oracle:** A canonical fixed-order run as reference.
- **Design:** Run the composed operator set under many randomized operator orderings; compare committed state. Separately, run the contested-mass case (drying/pyrolysis/smoldering on one variable) with and without the cascade.
- **Metrics:** max state divergence across orderings (contributions); divergence with vs without cascade (transitions).
- **Pass criteria:** contribution divergence **< solver tolerance** across **all** orderings; transition divergence **> 0** without cascade and **< solver tolerance** with it.
- **Failure → outcome:** Order-dependence in contributions → **REDESIGN** (a contribution is secretly stateful — find and refactor it). **Depends on:** V0.3.

### V1.2 — Operator-splitting stability for stiff cyclic coupling
- **Targets:** Decision #12; the char↔conduction↔pyrolysis stiff loop; rate-refinement as the multi-rate handler.
- **Claim:** Operator-split integration of the stiff fire loop converges to the monolithic reference as the timestep shrinks, and the rate-refinement local-timestep mechanism keeps it stable at production step sizes.
- **Load-bearing because:** If splitting is unstable on the flagship phenomenon, the whole compose-on-shared-state model is unreliable for stiff couplings.
- **Nature:** Engineering (stability/convergence).
- **Oracle:** Monolithic implicit solve (§7).
- **Design:** Run the burn with operator splitting at a range of global timesteps, with and without rate-triggered local sub-stepping; compare trajectories to the monolithic reference. Map the stability boundary.
- **Metrics:** trajectory error vs reference; largest stable step with/without local sub-stepping; observed order of convergence.
- **Pass criteria:** convergence to reference at expected order; production step stable *with* sub-stepping; error **< 5%** on integrated burn outcomes (mass consumed, peak T, char depth).
- **Failure → outcome:** **CONSTRAIN** (mark stiff loops as requiring sub-stepping / implicit treatment) or **REDESIGN** (adopt a semi-implicit split). **Depends on:** V0.3.

### V1.3 — Nonlinear-rate lumping: the Jensen correction
- **Targets:** Decision #16; sub-cell variance correction for Arrhenius-type rates.
- **Claim:** For a cell with a steep internal temperature gradient, the rate computed at the cell mean *underestimates* the true integrated rate, and the variance-corrected estimate recovers it; mean-only lumping can spuriously extinguish a burn.
- **Load-bearing because:** Without this, homogenized cells silently kill the fire — a correctness failure masquerading as physics.
- **Nature:** Empirical.
- **Oracle:** Fine-scale integration of the rate over the resolved sub-cell temperature field (the true cell-averaged rate).
- **Design:** Construct cells with hot-face/cold-core gradients of increasing steepness. Compare (a) rate at mean T, (b) variance-corrected rate, (c) fine-scale true rate. Confirm the mean-only failure and the correction's recovery; find where second-order correction itself breaks down (→ a refine trigger).
- **Metrics:** relative error of mean-only and corrected estimates vs true; gradient steepness at which the corrected error exceeds tolerance.
- **Pass criteria:** corrected error **< 10%** of true rate up to a documented steepness; mean-only error demonstrably large (**>50%**) in steep cases; corrected-error-exceeds-tolerance point identified and wired as a refine criterion.
- **Failure → outcome:** **CONSTRAIN** (lower the variance-correction validity threshold / refine earlier). **Depends on:** V0.1.

### V1.4 — Coupling-operator full pipeline & graceful spectral LOD
- **Targets:** Decision #24, #25; extends the *already-proven* symmetry core (`coupling_operator_core.py`, `coupling_operator_c6.py`).
- **Claim:** The full pipeline (reference + skeleton + thickness knob → within-part lift → symmetry-adapted GFT → coefficient tensor → reconstructed 3D form) reproduces the reference silhouette; and coefficient truncation is a *graceful* LOD where low-frequency terms carry macro-geometry.
- **Load-bearing because:** The symmetry resolution is verified, but the end-to-end lift (silhouette fidelity, the macro=low-freq claim, truncation-as-LOD) is not.
- **Nature:** Empirical.
- **Oracle:** The input reference silhouette/multi-view as ground truth for reconstruction error; a dense reconstruction as the truncation reference.
- **Design:** On a bilateral biped then the C₆ seraph: reconstruct the surface at descending truncation levels; measure silhouette/Chamfer error vs reference and vs the full reconstruction. Verify that perturbing a single low-`(m,k)` coefficient changes macro-proportion (limb length) and a high one changes micro-detail. Test the piecewise-patch stitching `C¹` continuity at scaffold seams for branching topology.
- **Metrics:** silhouette IoU / Chamfer distance vs truncation level; macro-vs-micro response of designated coefficients; seam continuity error.
- **Pass criteria:** full reconstruction silhouette IoU **> 0.9**; error decays monotonically and smoothly with added coefficients (graceful — no instability); designated low coefficients move macro-geometry, high ones do not; seam `C¹` residual below a set tolerance.
- **Failure → outcome:** Non-graceful truncation → **REDESIGN** the basis (per-part normalization). Poor silhouette fidelity from one view → **CONSTRAIN** (require multi-view or more anchors). **Depends on:** proven symmetry core (baseline, §8).

### V1.5 — Regulator + bounded reserve → emergent mortality & envelope shape
- **Targets:** Decisions #19, #21; the viability-envelope and "mortality from conservation, not scripting" claims.
- **Claim:** A minimal regulated subsystem (cardiovascular loop + finite blood-volume reserve) recovers from small perturbations and, past a reserve-dependent threshold, ignites a positive-feedback cascade to an absorbing (death) state — with the viable region shrinking as reserve depletes.
- **Load-bearing because:** The viability margin is the living-asset currency; its meaning depends on the envelope being real and reserve-dependent.
- **Nature:** Empirical (dynamical-systems characterization).
- **Oracle:** Direct phase-space / basin-of-attraction analysis of the regulator ODE system (the envelope computed exactly for the minimal model).
- **Design:** Build the minimal loop. Map the basin of attraction of the healthy fixed point at several reserve levels (apply graded perturbations, observe recover vs collapse). Confirm the cascade is positive-feedback-driven and that the basin contracts as reserve falls.
- **Metrics:** basin boundary location vs reserve level; existence/stability of the healthy fixed point; cascade onset threshold.
- **Pass criteria:** healthy fixed point exists and is stable at full reserve; basin contracts monotonically with reserve; collapse is an absorbing state reached only past the boundary (no spurious recovery).
- **Failure → outcome:** **REDESIGN** the regulator/reserve coupling. **Depends on:** —

### V1.6 — Regulator numerical stability (no spurious limit cycles)
- **Targets:** Risk: regulator gain tuning / limit cycles; the passivity/energy guardrail.
- **Claim:** Formulating regulator actuation as a dissipative draw on a reserve (passivity) prevents spurious limit-cycle oscillations across a wider gain range than a naïve force-style controller.
- **Load-bearing because:** Limit cycles look like the creature trembling — the model ringing, mistaken for physiology.
- **Nature:** Engineering (stability margin).
- **Oracle:** Linear stability analysis of the coupled regulator system (eigenvalues) as the analytic boundary.
- **Design:** Sweep regulator gains for both the passivity formulation and a naïve formulation; detect limit cycles (sustained oscillation in steady environment); compare stable regions against the linear-analysis boundary.
- **Metrics:** stable gain region area; agreement with predicted boundary; oscillation amplitude where unstable.
- **Pass criteria:** passivity formulation's stable region strictly larger; no sustained oscillation within the declared production gain envelope.
- **Failure → outcome:** **CONSTRAIN** (publish gain bounds + require sub-stepping). **Depends on:** V1.5.

### V1.7 — Skeleton precipitation under load
- **Targets:** Decision #22; "skeleton is a precipitate, Wolff's law as a generative rule."
- **Claim:** Stress-driven deposition under a creature's load case (gravity from world X + self-weight + ability loads) produces a *connected, load-bearing* skeletal structure, and changing the load (e.g. low gravity, or a support law-domain) changes the resulting skeleton accordingly.
- **Load-bearing because:** If precipitation yields disconnected or non-load-bearing junk, the "origin of structure" mechanism fails and skeletons must be authored after all.
- **Nature:** Empirical/engineering.
- **Oracle:** Standard topology-optimization (e.g. compliance-minimization) reference for the same load/domain — a mature, trusted method.
- **Design:** Run stress-driven deposition for a biped under Earth gravity, under low gravity, and with a radiance support-field (seraph) reducing load to near-zero. Compare connectivity, load capacity, and morphology against the topology-optimization reference; confirm the seraph case precipitates near-nothing.
- **Metrics:** structural connectivity; compliance/load capacity vs reference; sensitivity of morphology to gravity and support-field.
- **Pass criteria:** produced structures are fully connected and load-bearing (compliance within **2×** of the topology-opt reference); morphology responds in the expected direction to load changes; support-field case yields minimal skeleton.
- **Failure → outcome:** **REDESIGN** (use topology optimization directly as the precipitation operator rather than a bespoke rule).

### V1.8 — Growth memoization & write-back correctness
- **Targets:** Decision #11; the growth-trace memoization and interaction write-back loop.
- **Claim:** Evaluating a memoized growth trace at a given (time, LOD) is deterministic and matches a fresh growth run; and a write-back (e.g. a recorded cut) correctly invalidates stale memoized sub-results so subsequent growth heals around the wound rather than replaying pre-wound state.
- **Load-bearing because:** The "growth → interaction → heal" loop and the kilobyte-storage claim both depend on memoization being correct under write-back.
- **Nature:** Theorem-check (determinism) + empirical (healing correctness).
- **Oracle:** A from-scratch (non-memoized) growth run as reference.
- **Design:** Grow a tree; cache the trace; evaluate at several (time, LOD) points and compare to fresh runs. Apply a cut (write-back), advance growth, and confirm (a) the memoization key includes the write-back state, (b) healing tissue grows around the wound, (c) no stale pre-wound replay occurs.
- **Metrics:** memoized-vs-fresh divergence; correctness of cache invalidation on write-back; determinism across repeated evaluations.
- **Pass criteria:** memoized = fresh within determinism tolerance; **0** stale-replay incidents across the write-back suite.
- **Failure → outcome:** **REDESIGN** the cache-key/invalidation scheme. **Depends on:** V0.5.

### V1.9 — Dual-cloud skinning fidelity
- **Targets:** Decision #6; coarse physics cloud driving a dense render cloud.
- **Claim:** A dense render cloud skinned to a coarse physics cloud reproduces the deformation of a full-resolution physics sim within a visual-error tolerance, at a large node-count reduction.
- **Load-bearing because:** The "simulate thousands, render millions" economy depends on the skinned render cloud not visibly diverging from true physics.
- **Nature:** Empirical/engineering (quality-vs-cost).
- **Oracle:** A full-resolution physics sim where every render-detail point is itself simulated.
- **Design:** Deform a body via the coarse physics cloud; skin the render cloud; compare against the full-resolution sim under a range of deformation severities and coarse/dense ratios.
- **Metrics:** per-point geometric error vs full sim; perceptual error proxy; physics/render node ratio achieved at tolerance.
- **Pass criteria:** geometric error below tolerance at large deformations for a **≥10×** node reduction; error grows gracefully (no popping) as deformation increases.
- **Failure → outcome:** **CONSTRAIN** (cap supported deformation per coarse resolution; add adaptive physics-cloud refinement under extreme deformation).

---

## TIER 2 — Known hard risks

### V2.1 — RVE ↔ learned-surrogate handoff in the violent regime *(Risk #1 — highest)*
- **Targets:** Risk #1; the decision rule for *pay-for-RVE-solve* vs *trust-estimated-uncertainty-surrogate* where the analytic V–R bound is invalid (large deformation, active fracture, the death cascade).
- **Claim:** In the violent regime there exists a decision rule — keyed on surrogate uncertainty and a cost budget — that avoids both *stalling* (always solving) and *lying* (always trusting), keeping outcome error bounded while keeping cost within budget.
- **Load-bearing because:** The architecture explicitly admits the rigorous bound runs out here, and that everything elegant rests on getting this handoff right. It is the project's single largest engineering risk and the death cascade lives in it.
- **Nature:** Engineering (the central tradeoff).
- **Oracle:** Expensive full RVE/DNS of violent-regime cells as ground truth; the surrogate's own predicted uncertainty as the quantity being calibrated against that truth.
- **Design:** Generate a battery of violent-regime cells (fracturing, large-strain, cascading). For each, obtain the true outcome (RVE) and the surrogate's prediction + self-uncertainty. (1) **Calibration:** is predicted uncertainty a faithful estimate of actual error? (2) **Rule design:** sweep an uncertainty threshold and a cost budget; for each policy, measure outcome error and total cost. (3) Map the stall↔lie frontier and locate an operating point.
- **Metrics:** uncertainty calibration (predicted vs actual error correlation, over/under-confidence); for each policy: mean & tail outcome error, fraction sent to RVE, total cost.
- **Pass criteria (pre-registered, staged):** *calibration* — predicted uncertainty correlates with actual error (rank correlation **> 0.8**) and is not systematically over-confident. *Rule* — an operating point exists with tail outcome error below a set bound **and** RVE-fraction below a set budget. If calibration fails, the rule stage is moot.
- **Failure → outcome:** Uncalibrated uncertainty → **CONSTRAIN hard**: in the violent regime, *always* RVE-solve (accept the cost; forbid surrogate trust there) and document the resulting performance ceiling. This is a survivable but expensive fallback — and identifying it early is precisely why this verification runs before scale work.
- **Depends on:** V0.1, V2.4.

### V2.2 — Percolation: the off-axis thin-connected-feature danger case
- **Targets:** Risk: percolation; the claim that axis-aligned seams self-report but off-axis ones may not.
- **Claim:** A thin *connected* low-stiffness seam destroys effective stiffness out of proportion to its volume fraction; the directional V–R gap catches this when the seam is axis-aligned but **fails** to catch it when the seam is off the principal axes — establishing the need for a connectivity-based hard refine trigger.
- **Load-bearing because:** This is the named deepest failure mode of volume-fraction homogenization; the architecture claims partial self-protection and one residual danger that must be guarded explicitly.
- **Nature:** Empirical.
- **Oracle:** DNS of the seamed cell at varying seam angles.
- **Design:** Insert a thin connected seam at sweep of angles 0°→45°→90° relative to principal axes. Compare DNS effective stiffness, the homogenized estimate, and the V–R gap. Confirm: axis-aligned → gap blows open (self-reports); off-axis → estimate wrong but gap *does not* widen enough (the danger). Then test a connectivity-detector (e.g. a percolation/graph-connectivity check) as the proposed hard trigger.
- **Metrics:** homogenization error vs DNS by angle; gap width by angle; detection rate of the connectivity trigger.
- **Pass criteria:** demonstrate the off-axis blind spot exists (homogenization error large while gap small for ≥1 angle); the connectivity trigger detects **100%** of percolating seams regardless of angle.
- **Failure → outcome:** **CONSTRAIN** — connectivity check becomes a mandatory hard refine trigger wherever seams can form (char, cracks). **Depends on:** V0.1.

### V2.3 — Geometric vs physical smoothness misalignment
- **Targets:** Decision #24 caveat; the char-layer "geometrically thin, physically dominant" problem.
- **Claim:** A low-frequency *geometric* truncation can discard a feature with large *physical* impact; and a physics-aware (co-designed) basis or a physics-weighted error metric mitigates this.
- **Load-bearing because:** The unified "truncation = LOD = homogenization" elegance breaks if geometric and physical coarse spaces disagree silently.
- **Nature:** Empirical.
- **Oracle:** DNS physical response of the full-resolution feature vs the truncated representation.
- **Design:** Take the thin-char-layer cell. Apply geometric low-pass truncation; measure the physical (stiffness/conduction) error it introduces vs DNS. Then test whether a physics-weighted basis / error metric preserves the feature at the same coefficient budget.
- **Metrics:** physical error from geometric truncation; physical error from physics-aware truncation at equal budget.
- **Pass criteria:** demonstrate the misalignment (geometric truncation introduces large physical error in ≥1 case); physics-aware basis reduces that error substantially at equal budget.
- **Failure → outcome:** **CONSTRAIN/REDESIGN** — adopt a physics-weighted truncation metric; document that geometric and physical LOD are coupled, not free. **Depends on:** V0.1.

### V2.4 — Surrogate generalization, OOD fallback, and PINN data efficiency
- **Targets:** Decision #17; train-on-archetype / condition-on-descriptor / monitor-by-predicate.
- **Claim:** A physics-informed graph-net trained on one archetype and conditioned on the homogenized descriptor (a) generalizes across the held-out parameter family, (b) detectably degrades out-of-distribution so the predicate triggers fallback, and (c) needs less data than a non-physics-informed baseline to reach equal accuracy.
- **Load-bearing because:** The affordable-forest / macro-surrogate story and the OOD-fallback-as-refinement story both depend on all three.
- **Nature:** Empirical (generalization, calibration) + engineering (data efficiency).
- **Oracle:** Analytic operators / DNS for held-out and OOD test cases; a pure-data-loss network as the data-efficiency baseline.
- **Design:** Train on the archetype's max-refinement micro-sims. Test on (a) held-out in-family parameters, (b) deliberately OOD inputs (e.g. fire surrogate hit by an atomizing flux). Measure generalization error, whether conservation-residual/uncertainty flags the OOD cases, and the data-vs-accuracy curve for PINN vs pure-data.
- **Metrics:** in-family generalization error; OOD detection rate & latency; samples-to-target-accuracy (PINN vs baseline).
- **Pass criteria:** in-family error below tolerance; OOD detection **≥99%** with bounded latency; PINN reaches target accuracy with **<50%** of baseline data.
- **Failure → outcome:** Poor generalization → **CONSTRAIN** (more archetypes per family / richer descriptor). Undetected OOD → links to V2.1 fallback (force RVE). **Depends on:** V0.1 (descriptor), V0.3 (residual monitor).

### V2.5 — Inverse design convergence & candidate verification
- **Targets:** Decision #18; the "set the result, get the properties" non-render analysis.
- **Claim:** Gradient descent on input parameters through the differentiable operators/surrogate converges to parameters producing a target outcome, and the resulting candidate, when re-simulated with the *real* operators, matches the target within tolerance (it is a verified candidate, not an oracle).
- **Load-bearing because:** Inverse design powers the survival-spectrum analysis and the dev "dial-in-a-target" workflow.
- **Nature:** Engineering (optimization) + empirical (candidate fidelity).
- **Oracle:** The real analytic operators re-simulating the optimized candidate (the verification step itself).
- **Design:** Pick target outcomes (a burn pattern; a creature's resting homeostatic state in world X). Run gradient-based inverse design through the differentiable path; then re-simulate the result with analytic operators. Measure convergence and the surrogate-vs-real gap on the candidate. Include a known-incoherent target to confirm the analysis *reports impossibility* rather than fabricating a solution.
- **Metrics:** convergence rate / iterations; target error after optimization (surrogate); target error after real-operator verification; correct flagging of incoherent targets.
- **Pass criteria:** converges on coherent targets; verified candidate within tolerance of target; incoherent targets reported as such (not silently "solved").
- **Failure → outcome:** Non-convergence → **REDESIGN** (better-conditioned parameterization). Large surrogate-vs-real gap → tighten via V2.4. **Depends on:** V2.4.

---

## 7. Shared reference oracles (throwaway scaffolding)

These are *not* product code — they exist only to ground the verifications and may be slow, simple, and discarded after.

- **DNS micro-solver.** Fully-resolved fine-scale solve of a single heterogeneous cell under prescribed boundary conditions → true effective properties and true responses. Underpins V0.1, V0.2, V1.3, V2.2, V2.3, V2.1.
- **Monolithic implicit reference integrator.** The coupled fire/physiology PDE systems solved *without* operator splitting → trajectory ground truth for V0.3, V1.2.
- **Analytic-solution library.** Closed forms where they exist (layered-medium effective stiffness, simple regulator basins, single-operator steady states) → cheap exact checks anchoring V0.1, V1.5, V0.3.
- **Topology-optimization reference.** Mature compliance-minimization for V1.7.

Building these first is the largest up-front cost of the protocol and is intentional: it is how "build then verify" is replaced by "verify then build."

---

## 8. Already-established results (do not re-verify; guard against regression)

- **Symmetry-adapted graph transform.** The degeneracy-resolution core of the coupling operator is proven on `Z₂` (`coupling_operator_core.py`) and on `C₆` with genuine 2-D irreps (`coupling_operator_c6.py`): commutation exact, irrep labeling exact, residuals at 10⁻¹⁴, authoring energy splits as predicted. **Action:** convert these into a regression test that must keep passing as the full pipeline (V1.4) is built on top of them. No new verification required for the symmetry resolution itself.

---

## 9. Exit criteria — "verified enough to build"

- **Phase 0 (tree slice) may begin** when all of Tier 0 passes (V0.1–V0.5) and V1.1–V1.3 pass. These establish that the substrate, the homogenization currency, conservation/composition, scaling, and determinism are sound — the irreducible core the tree slice exercises.
- **Living-asset work (Phase 2) may begin** when V1.5–V1.6 pass.
- **Authoring/spectral work (Phase 1) may begin** when V1.4 passes (resting on the already-proven symmetry core).
- **Scale & exotic claims (Phase 4, surrogates/forest)** are gated on V2.1 and V2.4; until V2.1 resolves, the violent-regime cost ceiling is treated as unknown and the always-RVE fallback is the planning assumption.

A failed Tier-0 verification is the cheapest possible discovery: it changes the architecture record (`NEBULA_ARCHITECTURE.md`) before a line of production code commits to the assumption.

---

*End of protocol v0.1. Each entry above is the specification for one notebook; notebooks are to be implemented against these pre-registered criteria, not the reverse.*