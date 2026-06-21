# Nebula — Consolidated Verification Report (Tiers 0–2)

**Status: the verification protocol is COMPLETE. All 19 verifications PASS** (Tier 0: V0.1–V0.5; Tier 1:
V1.1–V1.9; Tier 2: V2.1–V2.5). Every load-bearing architectural assumption that, if false, would force a
redesign has been falsifiable-tested against an **independent oracle obtained a different way**, with
**pass criteria frozen before running**. Two verifications carry an explicit **CONSTRAIN** verdict with an
adopted graded fix (V2.2, V2.3); two more PASS with a documented **standing constraint** (V1.2, V1.3); V2.1
first read CONSTRAIN and was resolved to PASS. The rest are clean PASS. **No KILL.**

This document consolidates the three tier reports and converts their results into **actionable findings for
implementation**. It is the authoritative pre-implementation record; the per-tier reports
(`verification_notebooks/phase{0,1,2}/results/tier{0,1,2}_report.md`) hold the full method/figures.

> **Folder convention.** `verification_notebooks/phaseN` = verification **Tier N**, *not* architecture
> "Phase N". Architecture phases (the build roadmap) are referenced as "arch Phase N".

---

## 1. Executive summary — what is now established

The architecture's central economy is that **five subsystems share one substrate (a typed hypergraph) and
one currency (a per-cell trust/viability scalar)**, and that **new capability reduces to machinery already
built**. Verification confirmed this empirically, end to end:

- **One trust scalar, four jobs** — the Voigt–Reuss gap is valid (100% containment), tight where it must be,
  and fires *before* structural criticality; it gates refinement, conservation tolerance, surrogate trust,
  and LOD. (V0.1, V0.2) It extends cleanly to **connectivity** (V2.2), **geometric-LOD danger** (V2.3), and
  **living matter** as the viability margin (V1.5).
- **One compute pattern, three guarantees** — *gather → stage into buses → reduce → commit* is simultaneously
  operator composition, the conservation audit, and the race-free GPU reduction. Composition is order-free;
  the audit catches composite-OOD that per-operator checks miss; the reduction is bit-reproducible with a
  fixed order. (V0.3, V1.1, V0.5)
- **One tree, the log factor** — a Morton-linearized octree delivers `O(n_active·log n_total)` and degrades
  gracefully to dense cost. (V0.4)
- **The recurring reduction is real** — skeleton = reaction wood (V1.7), growth heals via write-back (V1.8),
  dual-cloud "simulate hundreds, render millions" (V1.9), spectral truncation = LOD = homogenization (V1.4,
  V2.3), learned-surrogate fallback = the refinement predicate (V2.1, V2.4), inverse design = the non-render
  analysis verified against real operators (V2.5).

**Every architecture phase gate is unblocked:** arch Phase 0 (tree slice), Phase 1 (spectral authoring),
Phase 2 (living things), and Phase 4 (surrogates/scale) all have their gating verifications passed.

---

## 2. Consolidated results

| # | Verification | Verdict | Headline number |
|---|---|---|---|
| **V0.1** | Homogenization bound (the keystone) | ✅ PASS | containment 100% (234/234); layered exact 4.5e-3; tightness 100% <30% |
| **V0.2** | Criticality coincidence | ✅ PASS | gap crosses T_hi at χ=0.021 ≤ criticality, 100% of load cases |
| **V0.3** | Conservation + composite-OOD | ✅ PASS | in-dist residual ≤5.3e-11; 7/7 OOD caught incl. 3 envelope-green "stealth" |
| **V0.4** | Complexity scaling | ✅ PASS | n_total exp 0.161, n_active exp 1.009; ≤0.65× dense at full activation |
| **V0.5** | Determinism | ✅ PASS | fixed-order & integer bit-exact incl. CPU↔GPU; atomic = 12 bit-patterns |
| **V1.1** | Composition order & cascade | ✅ PASS | 120 orderings bit-identical; transitions ambiguous w/o cascade |
| **V1.2** | Operator-split stability | ✅ PASS *(CONSTRAIN)* | order 1.035; naive 266% corrupt → sub-step 0.02% / semi-implicit 0.30% |
| **V1.3** | Jensen sub-cell variance | ✅ PASS *(CONSTRAIN)* | mean-only −57…−93%; corrected ≤5.3% for ε<0.5; extinction 4.6% vs 92.7% |
| **V1.4** | Coupling pipeline & spectral LOD | ✅ PASS | silhouette IoU 0.939/0.919; macro 37.9×; symmetry lock 1e-15; C¹ seam 8e-13 |
| **V1.5** | Regulator + reserve → mortality | ✅ PASS | r_crit=0.237; viability margin = true basin 100%/600; saddle eig +3.10 |
| **V1.6** | Regulator stability (passivity) | ✅ PASS | passive/naive gain 2.47×; production box naive 28/28 vs passive 0/28 oscillate |
| **V1.7** | Skeleton precipitation (Wolff) | ✅ PASS | compliance 1.03–1.17× SIMP; solid-frac 0.855→0.208 w/ gravity; seraph 0.0 |
| **V1.8** | Growth memoization & write-back | ✅ PASS | 115/115 bit-exact; write-back omitted → 150 stale-replay nodes |
| **V1.9** | Dual-cloud skinning | ✅ PASS | LBS err 0.0034 flat 6×→954×; 1.54M pts/0.049s; translation foil 7× worse |
| **V2.4** | Surrogate generalization / OOD / PINN | ✅ PASS | in-family 1.7%; OOD 100% detect, 5% FP; PINN data ratio 0.28 |
| **V2.1** | RVE↔surrogate handoff (Risk #1) | ✅ PASS *(was CONSTRAIN)* | calib ρ 0.964; op-point RVE-frac 0.286 / P95 err 0.076 |
| **V2.2** | Percolation (off-axis seam) | ✅ PASS *(CONSTRAIN + graded fix)* | descriptor blind to 8× knockdown; g_perc ρ 0.90; 26-conn 100% |
| **V2.3** | Geometric vs physical smoothness | ✅ PASS *(CONSTRAIN + graded fix)* | misalignment 391%; co-designed exact; lod_trust ρ 1.0 vs energy 0.66 |
| **V2.5** | Inverse design & candidate verification | ✅ PASS | convergence 100%; verified err 0.024; trust ρ 0.84; 0 fabricated |

---

## 3. Actionable findings — the rules implementation MUST honor

These are the non-negotiable engineering disciplines the verifications established. Each is **wired through
existing machinery** (no new subsystems); skipping one re-opens a failure mode that was demonstrated directly.

### 3.1 Determinism & memoization (foundational — affects every kernel)
- **Fix the reduction order in every bus reduce.** Use a *fixed-order float* reduction for on-device
  bit-reproducibility; use *integer / fixed-point accumulation* for any value used as a **memoization key**
  (bit-exact cross-hardware). **Never** use atomic/default-order reductions where reproducibility matters —
  atomics produced 12 distinct bit-patterns from identical work. *(V0.5)*
- **Derive every RNG stream by hashing a stable key** (e.g. `blake2b` of the element's lineage/identity),
  not the global draw order, and **never Python's salted `hash()`**. This is what makes per-element
  memoization sound. *(V1.8)*
- **Memoization keys must include the write-back state.** Omitting it served stale pre-wound geometry
  (150 stale nodes) — demonstrated. *(V1.8)*

### 3.2 The trust scalar & refinement predicate (the single currency)
- **The refinement predicate carries two homogenization errors, not one:** the Voigt–Reuss directional gap
  (constitutive *responses*) and the **sub-cell variance** term `ε = ½σ²|g″/g|` (nonlinear *rates*). Refine
  when `gap > T_hi` **OR** `ε > ε*` (≈0.5). Mean-only lumping silently extinguished a real burn (consumed
  4.6% vs 92.7% of fuel). *(V0.1, V1.3)*
- **Carry sub-cell variance of hot fields** for any cell running an Arrhenius-type (convex) rate; apply the
  second-order Jensen correction. *(V1.3)*
- **Fold connectivity into the trust scalar.** Voigt/Reuss are fraction-only → provably blind to
  connectivity (byte-identical for a percolating seam and an 8×-stiffer scattered control). Append the
  directional **scalar-conductance residual `g_perc`** to the descriptor, and keep a **26-connectivity span
  check as a hard backstop** wherever seams form (char, cracks). *(V2.2)*
- **Hysteresis + 2:1 balance + interface (hanging-node) constraint** remain mandatory on the adaptive octree
  (architecture §III.2); the octree itself gives the log factor and degrades gracefully. *(V0.4)*

### 3.3 Operators, composition, and the stiff fire loop
- **Operators only stage additive contributions into conserved buses** (`gather→stage→reduce→commit`); they
  never call each other or mutate state. Composition is then order-free (120 orderings bit-identical). *(V1.1)*
- **Resolve competing transitions by declared cascade priority**, never evaluation order (ambiguous without
  it). *(V1.1)*
- **The conservation-residual audit is the PRIMARY composite-OOD monitor** — it caught 3 "stealth" events
  every per-operator envelope passed. Per-operator validity checks are necessary but not sufficient. *(V0.3)*
- **The additive split is plain forward Euler (no Lie/Strang commutator error).** The stiff
  char↔conduction↔pyrolysis loop therefore needs **rate-driven local sub-stepping** (validated, 0.02% error,
  11.7× larger stable step) **or** the **semi-implicit (IMEX) treatment** (0.30% error, lifts the
  sub-stepping requirement). Wire it through the refinement predicate's rate term. *(V1.2)* **Without it,
  the burn is silently corrupted (266% char error), not blown up — so it won't announce itself.**

### 3.4 Learned tier, violent regime, and inverse design (scale & exotic)
- **The learned surrogate is a physics-informed graph net conditioned on the homogenized descriptor**; its
  OOD trigger is *descriptor-envelope-exit OR percolation*; physics priors buy real data efficiency
  (0.28× the data). *(V2.4)*
- **Plain deep-ensemble self-uncertainty is distance-unaware and saturates on the violent tail.** For the
  RVE↔surrogate handoff use **randomized-prior functions** (rank-reliable `u`) + a **held-out temperature** +
  a **distance-keyed conformal** guarantee. **Always-RVE remains the safe ceiling for descriptor-far cells**
  — now triggered by a *calibrated, distance-aware* signal, so the violent-regime cost is a measured handoff
  within budget, not an unknown. *(V2.1)*
- **Inverse design returns a CANDIDATE, never an oracle.** Always (a) re-simulate the candidate with the real
  operator and (b) gate it by distance-to-manifold (FeatureDensity). ~Half of in-range candidates land
  off-manifold where the surrogate over-promises; the gate (trust ρ 0.84 with real error) catches them and
  routes to RVE. **Report a precise impossibility** for unreachable targets (out-of-range material;
  sub-`r_critical` physiology). *(V2.5)*
- **Geometric LOD ≠ physical LOD.** A naive low-frequency truncation over-stiffened a thin char layer by
  391%. Gate refine-vs-truncate with the **physics-weighted `lod_trust = V-R gap × (1 + g_perc)`**; co-design
  the basis (compliance/Reuss for series channels — exact for axis-aligned layers); **always-refine the
  off-axis thin-connected tail** (no global basis truncates it). *(V2.3)*

### 3.5 Living things, form, and structure
- **Regulator actuation must spend a bounded conserved reserve** → mortality is emergent (the reserve runs
  out), not a hit-point counter; the **viability margin is the living currency** (matched the true basin
  100%/600 states). Use the **passivity (dissipative-draw) formulation, not raw force** — strictly larger
  oscillation-free gain region (2.47×; production box: naive trembles 28/28, passive 0/28). *(V1.5, V1.6)*
- **Skeleton precipitation uses the fully-stressed MULTIPLICATIVE update** `ρ ← ρ·(S/k)^η` with a Mullender
  sensor radius (mesh independence) and resorb-from-full start (keeps the load path connected). A naive
  *linear* SED rule is stiff, oscillates, and fragments the structure. *(V1.7)*
- **Spectral form uses the symmetry-adapted (character-projector) basis** for symmetric creatures;
  truncation is graceful LOD with macro-geometry in the low frequencies; **lock the symmetry-breaking irreps**
  for symmetry-by-construction; stitch parts with the **C¹ quilt** (the hanging-node fix, reused). *(V1.4)*
- **Dual-cloud render uses rotation-aware LBS** (per-node polar-decomposition rotation); translation-only
  collapses (candy-wrapper, 7× worse). Error is **reduction-independent** → the physics cloud can be made
  arbitrarily coarse. *(V1.9)*

---

## 4. Where verification *refined or falsified* the design (read before building)

The protocol earned its keep here — these are cases where running the test changed the plan:

- **V2.2 — the percolation hypothesis was falsified, the conclusion strengthened.** The pre-registered
  mechanism ("the V-R gap stays *small* off-axis while error grows") does **not** hold: for high-contrast
  soft seams the isotropic gap is *large* at every angle (it would over-refine, not silently miss). The real
  blindness is to **connectivity** (identical fraction-only descriptor, multiples-different stiffness). The
  fix (graded `g_perc` + 26-conn backstop) is more decisive than the original story.
- **V1.2 — "operator splitting" has no commutator error.** Nebula stages all operators at the same time
  level and commits once, so the "split" *is* forward Euler on the coupled RHS. The whole difficulty is
  stiff-explicit *stability*, not splitting accuracy — which reframes the fix as sub-stepping/IMEX, not a
  higher-order splitting scheme.
- **V2.3 — the off-axis case is genuinely unfixable by any global basis.** The co-designed compliance basis
  is *exact* for axis-aligned layers but **no** global basis truncates an off-axis thin-connected seam → it
  must refine. The adopted "fix" is therefore a refine-vs-truncate **gate** (`lod_trust`), not a better basis.
- **V2.1 — the data-efficiency prior is also the violent-regime liability.** The physics-monotonicity prior
  that wins V2.4 pins ensemble members onto one smooth surface, so they agree off-manifold and `u` saturates
  exactly where error climbs. Resolved by RPF + conformal; the lesson (a prior that helps in-distribution can
  blind uncertainty out-of-distribution) generalizes.
- **V0.4 — a logarithmic law read as a power law.** The cost exponent only resolves below the 0.2 threshold
  over a wide enough window (3.5 decades); the unambiguous proof was the descent depth = 1.0×log₈(n).

---

## 5. Standing constraints & residual risks (budget for these)

These survived as **documented constraints**, not failures. Implementation must plan around them:

1. **Two always-refine/always-RVE ceilings stand.** (a) Violent-regime cells far from the surrogate's
   descriptor manifold → always-RVE *(V2.1)*. (b) The off-axis thin-connected / sub-resolution percolation
   tail → always-refine *(V2.2, V2.3)*. Both are now triggered by *calibrated, distance-aware* signals and
   are bounded — but they are real cost floors for the forest-fire / fracture-scale regimes.
2. **Stiff couplings require sub-stepping or IMEX** (V1.2) and **nonlinear-rate cells require variance
   tracking** (V1.3) — both wired through the refinement predicate; do not ship the plain explicit
   mean-only path.
3. **Inverse design is unsafe without verification.** ~Half of in-range candidates exploit the surrogate;
   the candidate gate + real re-simulation is mandatory, not optional. *(V2.5)*
4. **Multi-GPU determinism is only proxied.** V0.5 used CPU-as-second-device + input permutations; bit-exact
   cross-*GPU* reproducibility on a second physical device is unverified. Validate on real heterogeneous
   hardware before relying on memoization keys across machines.
5. **The spectral pipeline is validated on synthetic ground-truth-by-construction**, not real image assets;
   single-view depth is *provably* supplied by the thickness knob, and single-image inverse is
   underdetermined (architecture Part VII). Real-reference authoring needs priors / multi-view / the
   vitruvian anchors as human-in-the-loop.
6. **Biological parameters are representative/invented.** Nebula is a *plausibility engine* —
   internally consistent and cross-comparable — **not** a clinical/surgical simulator. Do not oversell.
7. **Minor:** the V1.7 low-gravity (g=0.4) run did not hit the strict convergence tolerance (morphology
   unambiguous); fixed-topology spectral parts need the piecewise C¹ quilt for branching/genus-changing
   morphology (global continuity is aspirational).

---

## 6. Reusable assets — what the verification work hands to implementation

All live in `src/verification/oracles/` (+ `src/verification/` for the spectral pipeline). Two classes:

**Production mechanisms (the shippable path — port/keep):**
`homogenization.py` (V-R proxy + directional estimate), `bus_runtime.py` (conserved-bus runtime + audit),
`fire_operators.py` (transfer operators), `multirate.py` + `semi_implicit_fire.py` (stiff-loop integrators),
`jensen_rate.py` (variance correction), `octree.py` / `octree_gpu.py` (the one tree; Warp),
`determinism.py` (reduction-order discipline), `symmetry_basis.py` + `coupling_pipeline.py` (spectral form),
`regulator.py` / `regulator_stability.py` (regulator + viability margin + passivity), `wolff.py`
(precipitation), `growth.py` (growth trace + write-back), `dualcloud.py` (rotation-aware skinning; Warp),
`surrogate_gnn.py` (learned tier + uncertainty), `percolation.py` (`g_perc` connectivity channel),
`spectral_lod.py` (`lod_trust` refine gate), `inverse_design.py` (verified-candidate gate + survival
spectrum). **Determinism regime:** fixed reduction order on-device, integer/fixed-point for memo keys.

**Throwaway oracles (verification scaffolding — kept for regression, not shipped):**
`dns_elasticity_3d.py` (periodic-homogenization FEM), `dns_damage_3d.py` (damage DNS), `analytic.py` (Backus
closed form), `monolithic_fire.py` (RK4 reference), `topology_opt.py` (SIMP), `cells.py` / `violent_cells.py`
(deterministic test microstructures), `failure.py` (distance-to-failure), `handoff_rule.py` (calibration
tooling — partly production for the V2.1 calibrated gate).

**Hardware/runtime baseline that worked:** Python 3.13 + NumPy/SciPy; **GPU via cupy (sparse CG) and NVIDIA
Warp (traversal/reduction)** on an RTX 4090; PyTorch (+CUDA) for the surrogate. Tier-0 oracles are
throughput-bound (GPU genuinely wins); Tier-1 fire/ODE/spectral work is latency-bound and CPU is correct by
design. **Cache the expensive ground-truth (DNS) to `.npz`** — it dominates runtime (e.g. ~17s/cell), while
surrogate training is ~1 min.

---

## 7. Recommended implementation order (arch Phase 0 first)

Every gate below is **green**; build in this order so each step rests on a verified mechanism.

1. **Arch Phase 0 — the tree slice (now unblocked: Tier 0 + V1.1–V1.3).** Heightfield + SDF + the typed
   hypergraph substrate; the conserved-bus runtime (`bus_runtime`) with **fixed-order/integer reductions**;
   the fire operator set (`fire_operators`) on the bus, integrated with **`multirate`/`semi_implicit_fire`**;
   the **restriction operator** emitting the V-R gap **and** the variance ε; the **coarse-to-fine predicate**
   (proximity / proxy-error gap / rate-ε / pin, with hysteresis + 2:1 + interface) on the **octree**; glTF
   export; deterministic seeding via **hashed sub-keys**. This forces every Tier-0/1 invariant into real code.
2. **Arch Phase 1 — spectral authoring (V1.4).** Extend `symmetry_basis` + `coupling_pipeline`: anchors →
   within-part lift (+ thickness knob) → symmetry-adapted GFT → coefficient tensor → grown geometry; target
   the bilateral biped then the C₆ seraph; **lock symmetry-breaking irreps**; C¹ quilt at seams.
3. **Arch Phase 2 — living things (V1.5, V1.6).** Regulators + bounded reserves + the **viability margin** as
   currency; **passivity formulation**; the surgery scenario (margin positive through every phase).
4. **Arch Phase 4 — surrogates & scale (V2.4, V2.1, V2.5).** PINN graph-net distillation conditioned on the
   homogenized descriptor; the **calibrated RVE↔surrogate handoff** (RPF + temperature + conformal) with the
   **always-RVE ceiling**; the **verified-candidate** inverse-design / survival-spectrum workflow. Connectivity
   (`g_perc`) and `lod_trust` ride along inside the one trust scalar.

**Bottom line for the build:** the substrate, the single currency, conservation/composition, scaling,
determinism, the spectral form pipeline, living-asset homeostasis, and the surrogate/inverse-design loop are
all verified against independent oracles. Honor the §3 disciplines, budget for the §5 ceilings, reuse the §6
mechanisms, and the implementation rests on tested ground rather than faith.

---

*Companion documents: `ARCHITECTURE.md` (design + decision log), `verification_protocol.md` (the frozen
specs), and the per-tier reports under `verification_notebooks/phase{0,1,2}/results/`.*
