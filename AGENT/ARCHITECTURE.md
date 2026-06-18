# Nebula

**A simulation-first 3D creation language.**
Assets are not modeled — they are *grown, simulated, and derived*. A Nebula program describes the **causes** (materials, processes, fields, laws, developmental programs); the runtime solves for what those causes **imply**, at whatever time and resolution it is asked for.

- **Status:** Architecture consolidation — pre-implementation. Conceptual frame complete; spectral symmetry core prototyped and validated.
- **Version:** 0.1 (design record)
- **Implementation target (Phase 0):** Python host + Taichi/Warp GPU kernels.

---

## 0. How to read this document

This is both an **architecture overview** (Parts I–V) and a **decision log** (Part VI), followed by **open problems** (Part VII), a **roadmap** (Part VIII), and a **glossary** (Part IX). The recurring theme — stated once here and demonstrated throughout — is that Nebula repeatedly reduces apparently-new capability to machinery already built. Five subsystems share **one substrate** (a typed hypergraph) and **one currency** (a per-cell trust/viability scalar). The phrase "this is the same machinery, one domain over" is not rhetoric; it is the central architectural claim, and most of the design's economy comes from it being literally true.

---

## Part I — Foundational thesis

Nebula inverts the usual content pipeline. A modeling format stores a *result* ("80,000 vertices"); a Nebula program stores a *cause* ("granite that sat in a riverbed for 10,000 years") and solves for the result. Four consequences follow and are treated as non-negotiable foundations:

**Assets are functions of time and resolution, not static data.** `village @ (founded + 80 years, fire + 3 years)` is a valid expression. You sample the asset; you mesh only at export (to glTF/USD). Storage is the *generative description* — a program, a seed, a field history — typically kilobytes that expand on demand.

**Materials are physical, not visual.** A material declares density, elasticity, erodibility, conductivity, toughness. Color and texture are *derived* properties, outputs of simulation (weathering, charring, wetting), not authored inputs. This single reframing is most of what separates "looks real" from "looks CG."

**Determinism is mandatory.** Seeded randomness, fixed timesteps, versioned operators: the same program always yields the same asset. This is what makes the program *be* the asset, makes sub-simulations memoizable, and makes level-of-detail "the same solve at a different resolution." It also forces a discipline — fixed GPU reduction orders — because floating-point non-associativity actively fights reproducibility.

**Fidelity is tiered.** Every operator offers a cheap phenomenological approximation and an expensive ground-truth mode, with aggressive caching of sub-simulations. Without this, iteration is unusable; with it, you pay full cost only where it matters.

**Representation is volumetric/implicit, not mesh-native.** Meshes fight simulation (erosion, fracture, growth all break fixed topology). Internally: signed distance fields for solids, sparse-voxel / MPM particles for stateful matter, heightfields for terrain. Mesh is an export artifact only.

---

## Part II — The substrate: a typed hypergraph

The single data structure underneath everything.

**Nodes carry fast-changing per-element STATE** — position, velocity, stress, temperature, pressure, charge. Small, per-element, mutable each step.

**Hyperedges carry shared, lawful structure** — constitutive models, material constants, governing equations, interaction rules. An n-ary edge is essential because physical coupling is rarely binary (a tetrahedral element binds 4 nodes; a muscle bundle binds dozens; a "this is sapwood" tag binds thousands). Law is shared and amortized; ten thousand sapwood nodes reference *one* sapwood hyperedge.

This split is the source of both economy and speed: **state per-node is small; law per-edge is shared** (size win), and processing happens **per hyperedge category** in uniform GPU kernels with independent edges colored to run in parallel without write conflicts (speed win). A naïve hypergraph is GPU-hostile; the **categorize → color → flatten** discipline is what rescues it.

The same object serves three roles at once: **storage format**, **coupling topology**, and **the solver's constraint graph** (a hyperedge *is* an XPBD constraint).

### Typed hyperedge taxonomy

| Type | Binds | Carries |
|---|---|---|
| **Material** | a node set | physical constants, constitutive model (analytic or learned) |
| **Layer** | a node set | a material tag + its physics + deposition history (e.g. bark, sapwood, heartwood) |
| **Constraint** | nodes | a positional/energetic constraint (contact, suture, clamp, interface) |
| **Law-domain** | a region of nodes | governing equations / fields for that region (gravity, meteorology, custom physics) |
| **Front** | a generative locus | growth sensing + decision rule + deposition rule |
| **Regulator** | a sensed var + actuator | closed-loop control law + a bounded conserved reserve |
| **Transient-contact** | an interacting pair | the live interaction physics; created and reclaimed on demand |
| **Interface** | a fine/coarse seam | the hanging-node constraint binding fine nodes to their coarse parent |

A "layer," a "law domain," a structural constraint, and a custom-physics region are therefore not separate systems — they are all typed hyperedges. This collapse is the architecture.

### Rendering is decoupled: the dual cloud

Appearance and physics want different sampling (rendering crowds detail at silhouettes; physics wants near-uniform). So: a **coarse physics cloud** (carries mass/stress/state, wired into the hypergraph) drives a **dense render cloud** (Gaussian splats bound by skinning weights, riding the deformation of their physics neighbors). Simulate thousands of nodes; render millions of splats. Note: a captured Gaussian cloud is **hollow** (outer shell only) — internals must be *generated*, never captured.

### Custom physics: law domains

"Define gravity / meteorology / material behavior per region and per layer" falls out for free: a law-domain is a typed hyperedge binding nodes and carrying governing equations. Gravity is a *field* per domain, not a global constant. A node's behavior is the composition of every law-hyperedge it belongs to; overlaps resolve by a **declared cascade** (CSS-specificity-like) with explicit **interface conditions** at domain boundaries.

---

## Part III — The subsystems

### III.1 Growth — writing the implicit field

Growth is an **active front** (a Front hyperedge) that each step: senses local fields → runs a small local decision rule → deposits new nodes plus their material/layer hyperedges → advances, branches, or terminates. Generalizes across tree (cambium sheet + apical meristems + root tips), bone, coral, crystal, and animal morphogenesis.

- **Sensing:** light, moisture, nutrients, hormone fields (auxin / apical dominance), and crucially **mechanical stress** — trees lay reaction wood under load, bone remodels along stress lines (Wolff's law).
- **Decision rule:** a seeded, field-biased L-system (branching topology) coupled to space-colonization (venation/limbs pulled toward light/open space).
- **Layers are the integral of the front's history**, not authored strata. Rings, taper, reaction wood, branch scars, heartwood (= xylem old enough to chemically transition) are all the time-record of one front against an environment.
- **Mostly offline.** Growth is seasonal, deterministic, memoized into a compact **growth trace**; the runtime *evaluates* the trace at any (time, LOD). Growth writes the **implicit/procedural substrate**, never explicit fine nodes.
- **Not only additive.** Bone removes; wounds heal by callus over a recorded cut. Interaction **writes back** into the growth substrate, and later growth reads it: growth → asset → interaction perturbs → growth heals. (Memoization keys must include the write-back state.)

### III.2 Coarse-to-fine — fractal physics decomposition

The efficiency engine. The key idea: **self-similar physics** — every hierarchy level holds a *valid coarser physics* of the level below (a real reduced model, not a thumbnail), so a distant interaction terminates its descent early. That early stop is where the log factor comes from.

- **One tree, three hats:** a single linearized octree (Morton codes, GPU-coalesced) serves as LOD/activation structure, multigrid solve hierarchy, and Barnes-Hut/FMM far-field structure simultaneously.
- **Restriction (coarsening) and refinement (when to go finer) are duals.** Refine exactly where the coarse proxy's error exceeds tolerance, nowhere else.
- **Refinement predicate** `D = max` over normalized criteria: *proximity* (contact source within a margin sized to contact scale), *proxy-error* (homogenization error / equation residual against state gradient), *rate* (`|∂state/∂t|`, which also sets a finer local timestep — spatial and temporal refinement are one trigger), *pin* (authored override).
- **Stability guardrails (non-negotiable):** hysteresis (split at `T_hi`, merge only after `D < T_lo` for τ quiet steps — prevents thrash/popping); the **2:1 balance** condition (adjacent cells differ by ≤1 level); the **hanging-node fix** via an Interface hyperedge (so the LOD seam is just another constraint category, no special case).
- **Honest complexity:** `O(n_active · log n_total)`, *not* flat `O(n log n)`. The architecture's whole job is keeping `n_active` sublinear. Global-activation events (forest fire, building collapse) degrade gracefully toward dense cost — the regime where you swap in a learned surrogate.

### III.3 Operators & composition — the conserved-bus discipline

The rule that makes composition both free and correct:

> **Operators never call each other and never mutate state directly. They only read shared state, and they only write by staging contributions into conserved-quantity buses that the runtime reduces and audits.**

This single discipline is *simultaneously* the conservation-audit mechanism and the race-free GPU parallelism pattern: **gather → stage into buses → reduce → commit** is both at once. The discipline that keeps physics honest is the discipline that makes it fast.

**Operator declaration schema:**

```
operator <name>
  binds        <hyperedge type>                 # material / contact-pair / law-domain
  reads        <state fields + fluxes>           # the gather set
  constants    <params, bound at compile>        # specialized against the scene
  contributes  <flux → named conserved bus>      # ADDITIVE, order-independent
  transitions  <state var : new value, cascade>  # NOT commutative → declared priority
  ledger       <conserved qty: from-pool → to-pool>   # the audit entries
  timescale    <characteristic rate / stiffness> # for multi-rate integration
  tiers        { analytic: cost, learned: cost }
  envelope     <on-distribution region of learned tier>
  fallback     <refine | drop-tier | both>       # on envelope-exit OR residual-spike
```

- **Two write modes:** *contributions* are additive into a bus → order-independent → compose freely; *transitions* change a variable's value/category → resolved by **declared cascade priority**, never by evaluation order (this is load-bearing for determinism).
- **Phenomena emerge; they are not primitives.** "Tree on fire" is the fixed point of combustion + conduction + pyrolysis + char-weakening coupled through `T`, `g` (volatiles), `O₂`, `χ` (char fraction). You **package the transfer operator (constitutive response), not the phenomenon** — this is what avoids combinatorial explosion. Fire + wet + lightning + load compose automatically because each operator reads/writes shared buses without knowing the others exist.
- **The audit catches what per-operator checks cannot.** Per-operator validity envelopes are *necessary but not sufficient*. A composite can go off-distribution while every operator individually reports "in distribution" — the symptom is a **conservation residual spike**, which is the *primary monitor* and is identical to the refinement predicate's proxy-error signal.

### III.4 The restriction / homogenization operator — the keystone

Every other subsystem depends on this one, because it produces the single scalar they all read.

- **Half of restriction is exact.** Conserved-extensive quantities (mass, energy, momentum, fuel, charge) restrict by **summation — zero error**. The conserved buses *are* this half. All homogenization error lives in the other half: **constitutive responses** (stiffness, conductivity, rate, toughness), which do not add.
- **The error bound is analytic and nearly free.** The true effective stiffness lies between the **Voigt** bound (uniform strain → arithmetic mean) and the **Reuss** bound (uniform stress → harmonic mean). The Voigt–Reuss gap *is* the error bar. For **layered media** (concentric tree shells, skin/fat/muscle), these bounds are achieved *exactly* in the principal directions — so the proxy is essentially exact along grain and around rings, with error confined to shear and where layering breaks.
- **The worst case and the most important case coincide.** A char wedge spikes stiffness contrast → the directional Reuss estimate collapses → the gap blows open → the trust scalar craters → the cell refines — *exactly* when the cell is structurally critical. Thermally, the same directional homogenization (series path → harmonic mean → dominated by the low-conductivity char layer) **derives** the `k(χ)` insulation coupling rather than assuming it.
- **The nonlinear trap:** the average of an exponential ≠ the exponential of the average (Jensen). A cell with a 700 °C face and 60 °C core has a mean temperature that *underestimates* face pyrolysis. Fix: carry **sub-cell variance** of hot fields and apply a second-order correction. Two homogenization errors are tracked: the Voigt–Reuss gap (responses) and the variance term (nonlinear rates).

> **ONE SCALAR, FOUR JOBS.** The cell's error bound is simultaneously: (1) the refinement predicate's proxy-error term, (2) the conservation audit's tolerance, (3) the learned tier's trust envelope, and (4) the LOD budget. Refinement, conservation, surrogate trust, and level-of-detail read one number.

### III.5 Learned surrogates — physics-informed graph networks

When analytic simulation is too expensive (global-activation regimes), drop in a learned surrogate that runs **on the same hypergraph** (graph-network simulators operate on exactly this structure).

- **Train on the archetype, condition on the descriptor.** Run one asset at max refinement, distill the coarse observable, condition on the **homogenized descriptor** — which is the *same vector* the restriction operator emits. (Coarsening feature = surrogate input = validity-check state.) One archetype need only span a *parameter family*, not every morphology.
- **Physics-informed loss:** trajectory match + governing-equation residual + conservation as a (hard or penalized) constraint. Buys data efficiency, energy-conserving extrapolation, and (for graph nets) physically-local message passing.
- **A surrogate is just a coarse proxy with a bounded error**, monitored by the *same* refinement predicate. Off-distribution → residual spikes → "refine" means *drop from learned tier to analytic operators* in that region. OOD fallback is not new machinery.
- **Three distinct gradients, three times:** *training* `∂loss/∂weights` (offline), *runtime* forward eval (no gradients), **inverse design** `∂outcome/∂input-params` — the "non-render analysis": *set the target result, gradient-descend to the parameters that produce it.* This yields a **candidate** to be verified against the real operators, not an oracle.
- **Compilation = partial evaluation.** The compiler specializes the general operator library against scene constants (local gravity, humidity, ambient O₂, law-domain equations), baking a fast operator for *this asset in this world*. Far-field instances compile straight to a validated macro-surrogate; hero instances compile to the live operator stack.
- **Dev workflow fork:** compose from existing operators (free) vs. **new physics needs new ground truth** (simulate-and-distill a new operator). The non-render analysis is the tool that tells the dev which fork they are on.

### III.6 Regulators — living things

A creature adds exactly **one** new primitive to the passive-asset machinery.

- **The regulator:** a closed-loop controller hyperedge — sense a variable, compare to a setpoint, drive an actuator to close the error (negative feedback — the thing a tree utterly lacks).
- **Guardrail (the same guardrail as custom physics):** a regulator may only actuate by **spending a bounded, conserved reserve against a finite capacity**. This is what makes a creature **killable correctly** — bleed it, the reserve depletes, the controller saturates, correction fails, it dies. Mortality is *emergent from a conserved quantity running out*, not a hit-point counter.
- **Networks classify cleanly:** *vascular* = pumped + regulated sap transport (a **mass/energy bus**); *neural* = **signal, not mass** (a fast propagation field that reads state and commands regulators — schema MUST flag *conserved-transport* vs *signal-field* or the audit silently corrupts); *endocrine* = **mass-as-slow-signal** (a diffusing field that retunes regulator gains / unlocks reserve — this is how fight-or-flight, stress, adrenaline are modeled).
- **Viability envelope:** the set of states the regulators can recover from. It **shrinks as reserves deplete**. Death = leaving it → a positive-feedback cascade (low volume → low perfusion → failing pump → less delivery → ...) running to an absorbing state.
- **Viability margin:** the normalized minimum distance to the envelope boundary across all regulated axes — the "how alive" scalar. It plays the *exact role* the homogenization bound plays for passive assets: sets physiological LOD, gates which subsystems/surrogates run, decides which abilities are affordable, and *is* the alive/dead predicate.
- **Survival spectrum = the non-render inverse analysis**, now physiological: specialize the creature's regulators + reserves against world X's law domains and solve for *whether a stable homeostatic fixed point exists inside the viable set, and how large the envelope is.* "Can this creature survive in this world" is **derived, not authored**.
- **Surgery is the tree's axe wound with regulation on.** Incision (sever cohesion + cut vascular edges → ledgered mass sink → regulators respond), processes (clamp = constraint hyperedge; reserves deplete from working above baseline; anesthesia = an operator suppressing neural signal flux), stitching (sutures = constraints; write-back to growth; multi-day healing front). Phases run on the existing multi-rate local-timestep machinery. "Keep it alive" = *viability margin stays positive on every axis through every phase.*

### III.7 The morphogenetic scaffold — origin of form

The skeleton is **not** the origin — it is a **precipitate**. Bone condenses under a stress field (Wolff's law = the tree's reaction-wood rule). Give a creature its loads and the skeleton falls out where the loads demand structure. The actual origin is upstream:

**The scaffold = axis/polarity seed + declared symmetry group + recursive organizer grammar.** (The egg's polarity, not the adult's shape.)

- **Axis & polarity:** a coordinate frame (head–tail, back–belly, left–right) set up like embryonic morphogen gradients; everything is positioned *relative* to it.
- **Symmetry group (a dial):** bilateral (`Z₂`), pentaradial (`C₅`), six-fold seraph (`C₆`/`D₆`), segmented (bilateral + repetition). The whole body inherits it. "Six wings" is one wing-organizer replicated by the group.
- **Recursive organizer grammar:** organizers are demand/attachment fields ("a limb here," "an eye here," "a heart-demand here"). **Recursive** — a limb is itself an axis with its own segmentation; digits are repeated units; the Alien's inner jaw is a *nested sub-body-plan*. The body plan is fractal — the same coarse-to-fine structure wearing a developmental hat.
- **Chain:** scaffold → growth fronts deposit tissue + vasculature (angiogenesis = the roots-toward-moisture algorithm) → skeleton precipitates under load → regulators switch on.
- **Two origins, composable:** (A) **reference-inverse** — infer the generative program whose grown output matches a picture (differentiable inverse design); (B) **generative-forward** — specify mechanism directly. Reconciled by the non-render survival analysis, which can **fail informatively** (return a precise impossibility as an actionable constraint).
- **Discrete organs** (one heart at a demand field) vs **distributed organs** (eye-density as a surface field) — both expressible in one grammar.
- **The honest bottom of the stack:** the organizer *vocabulary* (limb, eye-field, segment, symmetry group, organizer) is authored or learned — a curated developmental dictionary. "Infinite creatures" means *infinite compositions of a finite, human-seeded alphabet.*

### III.8 Spectral form — the vitruvian pipeline

How a user authors *form* by pointing at a reference, and the realization that names the whole language.

**The vitruvian skeleton hypergraph = two graphs:** a symmetry/proportion graph (which *is* the scaffold, painted onto the reference) and a range-of-motion graph (which *is* the kinematic layer — joint hyperedges with limits).

**The coupling operator** (the corrected form of "spectrally combine image + skeleton" — a **fiber-bundle transform**, not a cross-domain multiply):

1. **Within-part lift.** Treat each bone as a local medial axis; the surface is a generalized cylinder `r(s,θ)` (blobs use spherical harmonics over the part). The image, sampled perpendicular to the bone, gives in-plane half-width `w(s)`; **depth is provably unrecoverable from one view**, so the **thickness knob supplies the missing axis** `κ`. Expand each part in its shape basis → a coefficient vector `cᵢ` per node.
2. **Across-part transform.** Stack `{cᵢ}` as channels on the graph and apply the **Graph Fourier Transform** (a true tensor product only for homogeneous parts; generally a bundle).
3. **Joint coefficient tensor `Ĉ`** = the parametric model. Low `(m,k)` = macro proportions; high = micro detail. **Truncation IS level-of-detail** — so spectral truncation is the geometry-side fractal decomposition.

> **The language, named.** *Form is spectral and truncatable; behavior is declarative and specializable; and ONE operation — reduce the general continuous/declared object to a finite specialized one for the current need — is LOD, homogenization, growth-detail, and compilation all at once.* Coarse = low-frequency. Detail = high-frequency.
> *(Caveat: geometric smoothness ≠ physical smoothness — a char layer is geometrically thin but physically dominant. Co-designing the geometric basis with the physics coarse space is a goal, not a freebie; where it fails is the thin-connected-feature case the predicate must catch.)*

**Symmetry resolution (PROTOTYPED & VALIDATED).** The Laplacian commutes with the symmetry group, so its eigenspaces decompose into the group's irreducible representations. Degenerate eigenspaces have no canonical basis — so a naïve GFT is ill-posed *exactly* on symmetric creatures. Fix: **diagonalize the group action within each Laplacian eigenspace** (the character projector `Π^(α) = (d_α/|G|) Σ_g χ^(α)(g)* P_g`). This resolves degeneracy canonically and labels every mode by irrep.

- Validated on **bilateral `Z₂`**: modes split symmetric / antisymmetric; symmetry-preserving edits live entirely in the trivial irrep (energy 6.0 / 0.0), symmetry-breaking forces antisymmetric energy (1.5 / 1.5). Locking the antisymmetric coefficients to zero *guarantees* bilateral symmetry by construction.
- Validated on **six-fold `C₆`** (the real test — genuine 2-D irreps): the spectrum organizes by angular momentum (`m=0, ±1, ±2, 3`); raw eigenvectors in an `m=±1` space satisfy `⟨v,Rv⟩=0.5`, `‖Rv−⟨v,Rv⟩v‖=0.866` (= cos/sin 60°) — i.e. `ρ` *rotates* them, they have **no definite m** — while the adapted modes recover definite `m` with residuals at 10⁻¹⁴. Authoring: all-wings-equal → pure `m=0` (author one wing, the trivial irrep replicates it to six); alternating → `m=0 + m=3`; single wing → spread across *all* `m`, so symmetry-breaking is unrepresentable without resolving the 2-D irreps.
- Reference implementations: `coupling_operator_core.py` (Z₂), `coupling_operator_c6.py` (C₆).

### III.9 The metaphysical knob — origin of being

A single authoring surface over **three escape hatches already built**, which is why arbitrary creatures (no organs, runs on light, breathes hydrogen) and "impossible" places (floating cities) are expressible:

- *"Runs on light / breathes hydrogen / blood of X"* = **substituting the conserved quantity a regulator defends** (a radiance reserve fed by an ambient field; the viability margin is then defined over light).
- *"Full of light, no organs / bone hardness X"* = **material-constant + organizer-grammar override**.
- *"Floating city / places that don't make sense"* = **local law-domain override** (gravity → a support potential).

**Why it is safe to be this wild:** *only* because of the two guardrails enforced everywhere — laws are **energies/potentials**, not raw forces (so a support field conserves and stays stable), and exotic metabolisms are **real reserves on real conserved buses** (so the audit still holds). **The knob proposes the impossible; the survival analysis either finds the minimum law-domain that makes it coherent, or reports the specific impossibility** ("declared mass exceeds support potential"; "world X has no hydrogen for this reserve"). Impossibility returned as an actionable constraint is a *feature*. **Form = the vitruvian/spectral side; being = the metaphysical knob; the two are coupled at compile through the non-render analysis.**

---

## Part IV — Cross-cutting invariants

Three through-lines recur in every subsystem and are the real spine of the design.

**The single currency.** A per-cell scalar — homogenization trust for passive matter, viability margin for living matter — drives refinement, conservation tolerance, surrogate trust, affordability, LOD, and the alive/dead predicate. The system never juggles independent policies; it reads one number.

**The two guardrails.** (1) Express laws as **energies, potentials, and constitutive relations**, never raw forces — energy-derived dynamics conserve and stay stable; raw force fields inject energy and blow up. (2) Active behavior (regulation, "magic," exotic metabolism) must **spend a bounded conserved reserve** — never pin a value for free. Every "arbitrary freedom" the user is offered routes through these two, which is what lets the metaphysical knob be unbounded without becoming incoherent.

**The universal compute pattern.** **Gather → stage into buses → reduce → commit.** This is, simultaneously, operator composition, the conservation audit, and the race-free GPU color-and-flatten pass. One pattern; three guarantees.

**The recurring reduction.** Skeleton = reaction wood. Angiogenesis = roots-toward-moisture. Surgery = the axe wound with regulation on. Healing = callus growth. Learned-surrogate fallback = the refinement predicate. LOD = spectral truncation = homogenization = compilation. The interface seam (hanging node) appears in physics, in LOD, and in geometry stitching — same fix each time. New capability almost never means new machinery.

---

## Part V — Implementation architecture

Per-element physics **never** runs in the host language — it runs in compiled GPU kernels. The host orchestrates, compiles, and authors. So "what language" is three questions:

| Layer | Role | Decision | Why |
|---|---|---|---|
| **Compiler / research** | parse `.pkg`, build spectral operators (eigendecomp, projectors), partial-evaluate/specialize, train surrogates | **Python — permanent** | offline, latency-insensitive, math-heavy; already prototyped here |
| **GPU kernel substrate** | XPBD/MPM, field updates, operator kernels, color-and-flatten | **Taichi + Warp** (Phase 0) | Python-authored, JIT-to-GPU, **autodiff over kernels** (required for inverse design); MPM/XPBD are their flagship demos |
| **Shipping runtime** | per-frame, in-engine, 16 ms budget | **Deferred.** C++ or **Rust** for native/AAA; **C#** *iff* targeting Unity (DOTS/ECS ≈ the hypergraph, Burst SIMD) | only layer where GC/latency truly bite; determinism wants fixed memory layout + reduction order |

**Rationale for deferral.** The package format is **declarative and host-language-agnostic** — the Python compiler reads it now, a native runtime reads it later. Taichi/Warp **ahead-of-time compilation** is the bridge: kernels validated in the Python phase are emitted for the native runtime, so the eventual port is the *orchestration* layer only, not the physics. Julia was considered and set aside: steady-state throughput is fine, but GC and a thin engine-embedding ecosystem (not JIT latency per se) disqualify it for the shipping runtime.

---

## Part VI — Decision log

Terse, dated-by-sequence record. Format: **Decision — Rationale — Rejected alternatives.**

1. **Simulation-first, cause-not-result.** — Realism and AI-legibility both come from describing causes. — Rejected: mesh/parametric modeling (stores results, fights simulation).
2. **Assets are time/resolution functions; mesh only at export.** — Enables LOD-as-resolution, kilobyte storage, time-indexing. — Rejected: baked static assets.
3. **Determinism mandatory (seed + fixed step + versioned ops + fixed reduction order).** — Program *is* the asset; memoization; reproducibility. — Rejected: nondeterministic GPU reductions.
4. **Typed hypergraph substrate; state on nodes, law on hyperedges.** — One object = storage + topology + constraint graph; size + speed wins. — Rejected: attribute-per-point clouds; pairwise graphs (can't express n-ary coupling).
5. **Categorize → color → flatten** for all edge processing. — Removes GPU divergence; enables parallel constraint solve. — Rejected: naïve per-element dispatch (GPU-hostile).
6. **Dual cloud (coarse physics drives dense render).** — Decouples sim cost from render detail. — Rejected: simulating render-resolution Gaussians.
7. **Custom physics as law-domain hyperedges; energy/potential formulation only.** — Stability + conservation for free; per-region gravity/meteorology. — Rejected: hand-authored raw force fields (inject energy, blow up).
8. **Self-similar fractal decomposition; restriction ⟂ refinement duals.** — Gives `O(n_active·log n)`. — Rejected: flat uniform resolution (O(n²) interaction).
9. **Single octree = LOD + multigrid + far-field.** — One structure, three jobs. — Rejected: separate structures per role.
10. **Refinement predicate (proximity/proxy-error/rate/pin) + hysteresis + 2:1 balance + interface hyperedge.** — Stable adaptive resolution without popping or seam artifacts. — Rejected: unguarded refine/coarsen (thrash, energy leak).
11. **Growth = active front writing the implicit substrate; layers = deposition history.** — Tiny storage; emergent realism; healing via write-back. — Rejected: authored layer strata.
12. **Operators communicate only via conserved buses (gather-stage-reduce-commit).** — Composition + conservation + parallelism in one discipline. — Rejected: operators calling/mutating each other.
13. **Package the transfer operator, not the phenomenon.** — Kills combinatorial explosion; composition is free. — Rejected: monolithic "tree-on-fire" packages.
14. **Conservation residual audit is the primary composite-validity monitor.** — Per-operator envelopes miss cross-terms. — Rejected: trusting per-operator OOD checks alone.
15. **Voigt–Reuss gap as the homogenization error bound; one scalar, four jobs.** — Nearly-free analytic bound; exact for layered media in principal directions. — Rejected: unbounded mean-field homogenization.
16. **Carry sub-cell variance for nonlinear-rate cells (Jensen correction).** — Mean-only lumping silently extinguishes fire. — Rejected: cell-mean-only proxies.
17. **Learned tier = physics-informed graph net, conditioned on the homogenized descriptor, monitored by the refinement predicate.** — Same structure/feature/check vector; OOD → refine. — Rejected: black-box surrogates without conservation constraints.
18. **Inverse design via `∂outcome/∂params`; result is a verified candidate.** — "Set the result, get the properties." — Rejected: blind parameter search; treating surrogate as oracle.
19. **Regulator primitive; actuation spends a bounded conserved reserve.** — Emergent, correct mortality; one new primitive covers all life. — Rejected: scripted HP/state machines.
20. **Neural = signal-field, not a conserved bus (schema-flagged).** — Prevents silent conservation corruption. — Rejected: modeling signals as mass.
21. **Viability margin as the living-asset currency.** — Mirrors homogenization bound; "can it survive" is derived. — Rejected: authored survival rules.
22. **Skeleton precipitates under load; origin is the morphogenetic scaffold (axis + symmetry + recursive organizers).** — Covers Alien, seraph, novel creatures; rigging falls out. — Rejected: fixed anatomical atlas (fails on non-animals).
23. **Two origins: reference-inverse and generative-forward, reconciled by survival analysis.** — Picture-driven and mechanism-driven authoring; informative failure. — Rejected: pure sculpting; pure procedural.
24. **Form is spectral; coupling operator is a fiber-bundle transform; symmetry resolved by character projectors.** — LOD-native continuous form; symmetric creatures well-posed (validated). — Rejected: "multiply image-FT × graph-FT" (mismatched domains).
25. **Thickness knob supplies the depth axis the 2D spectrum cannot.** — Principled, not a hack: injects the provably-unrecoverable dimension. — Rejected: pretending single-view recovers depth.
26. **Metaphysical knob = unified surface over reserve/material/law overrides, made safe by the two guardrails.** — Exotic & "impossible" assets expressible and coherent. — Rejected: ad-hoc special-casing of magic.
27. **Implementation: Python compiler (permanent) + Taichi/Warp kernels (Phase 0) + deferred native runtime; declarative host-agnostic package format.** — Performance lives in kernels; autodiff for inverse design; port is orchestration-only. — Rejected: committing the shipping language now; Julia for the runtime.

---

## Part VII — Open problems & honest reservations

Tracked risks, roughly by severity.

- **The RVE-vs-learned-tier handoff in the violent regime** *(load-bearing engineering risk).* Voigt–Reuss bounds are linear-elasticity theorems; in large-deformation, actively-fracturing, dying-creature regimes there is *no* cheap analytic bound. At runtime you must decide: pay for a true fine RVE solve, or trust an estimated-uncertainty surrogate. Get it wrong → the system stalls (always solving) or lies (always trusting). Mitigant: the violent regime is *already* refined (wide bound), so it rarely needs homogenizing — but the explicit handoff rule is unbuilt and everything elegant rests on it. **The death cascade lives here too.**
- **Percolation.** Volume fraction is blind to connectivity; a thin connected crack/char seam destroys stiffness out of proportion to volume. Axis-aligned seams self-report (directional Reuss → 0); off-axis percolation needs a connectivity check or RVE solve as a hard refine trigger.
- **The organizer alphabet is authored, not derived.** Creativity is recombination of a finite human-seeded vocabulary. There is no body-plan-from-nothing.
- **Single-image inverse is underdetermined.** Topology, internals, back-faces, symmetry aren't fixed by one view. Needs priors (archetype library), multiple views, or human-in-the-loop (the vitruvian anchors *are* this supervision).
- **Emergence vs. art direction.** Growth gives *coherent* anatomy, not necessarily the *exact* drawn silhouette. Pinning reconciles, but a specific aesthetic cannot be fully *derived* from mechanism.
- **Regulator gain tuning / numerical stability.** Coupled nonlinear feedback can limit-cycle (looks like trembling — the model ringing). Passivity/energy formulation gives stability margins, but tuning is real work.
- **Geometric vs. physical smoothness misalignment.** A low-frequency geometric truncation can drop a high-impact physical feature (the char-layer problem). Co-design of bases is a goal, not free.
- **Biological parameters are largely unknown / invented.** Nebula is a *plausibility engine*, internally consistent and cross-comparable — **not** a clinical or surgical simulator without validated data. Do not oversell.
- **Fixed-topology spectral parts.** Branching/lacy/genus-changing morphology needs a **piecewise quilt** of patches stitched at the scaffold with `C¹` interface constraints; "one global continuous function" is aspirational.
- **Determinism vs. GPU float non-associativity.** Anything reproducible needs fixed reduction orders — a standing constraint on every kernel.

---

## Part VIII — Roadmap

**Phase 0 — Vertical slice (now): the tree, completely.** Heightfield + SDF + the hypergraph substrate; five growth/process operators; the fire operator set (combustion, conduction, pyrolysis, char-weakening) on the conserved-bus discipline; the restriction operator with Voigt–Reuss trust; the coarse-to-fine predicate; glTF export; deterministic seeding. Forces every architectural piece to become real. Built in Python + Taichi/Warp.

**Phase 1 — The coupling operator into authoring.** Extend the validated symmetry core into the full vitruvian pipeline: anchors → within-part lift (+ thickness knob) → symmetry-adapted GFT → coefficient tensor → grown geometry. Targets: one bilateral biped, then the six-fold seraph.

**Phase 2 — Living things.** Regulators + reserves + viability margin; the surgery scenario as the test (incision → processes → stitching, margin positive throughout); physiological LOD.

**Phase 3 — The package format & language surface.** `tree.pkg`, `seraph.pkg`, `dragon.pkg`, `xenomorph.pkg` — the declaration grammar spanning the simplest and strangest assets, host-language-agnostic by construction.

**Phase 4 — Surrogates & scale.** Physics-informed graph-net distillation; macro-surrogate far-field tier; the forest-fire scale test; the RVE-vs-surrogate handoff rule (Part VII risk #1).

**Later — the native shipping runtime.** C++/Rust (or C# for Unity), fed by AOT-emitted kernels.

---

## Part IX — Glossary

- **Asset** — a function of (time, resolution) sampled on demand, stored as its generative description.
- **Hypergraph substrate** — nodes (state) + typed hyperedges (law); storage, topology, and constraint graph in one.
- **Operator** — a constitutive/transfer law that reads shared state and writes only into conserved buses; phenomena emerge from operators composing.
- **Conserved bus** — a named conserved quantity (energy, mass, O₂, ...) into which operators stage additive contributions; reduced and audited each step.
- **Front** — a generative locus (growth) that senses fields and deposits matter; its history is an asset's layers.
- **Restriction / homogenization operator** — collapses a heterogeneous cell into one effective element + a trust bound (Voigt–Reuss gap + sub-cell variance).
- **Trust scalar / viability margin** — the single currency; drives refinement, conservation tolerance, surrogate trust, LOD, and (for life) the alive/dead predicate.
- **Law-domain** — a hyperedge carrying region-local governing equations (custom gravity/meteorology/physics).
- **Regulator** — a closed-loop controller spending a bounded conserved reserve; source of homeostasis and emergent mortality.
- **Viability envelope** — states a creature's regulators can recover from; shrinks as reserves deplete.
- **Morphogenetic scaffold** — axis/polarity + declared symmetry group + recursive organizer grammar; the origin of form.
- **Organizer** — a demand/attachment field inducing an organ/limb; discrete or distributed (surface-density).
- **Coupling operator** — the fiber-bundle transform (within-part shape basis → symmetry-adapted GFT) turning a reference + skeleton + thickness knob into a spectral form tensor.
- **Symmetry-adapted basis** — the canonical Laplacian basis chosen by the declared group's character projectors; resolves degeneracy, labels modes by irrep.
- **Metaphysical knob** — unified surface over reserve/material/law overrides; safe by the two guardrails.
- **The two guardrails** — laws as energies/potentials (not raw forces); active behavior spends bounded conserved reserves.
- **Gather → stage → reduce → commit** — the universal pattern that is composition, conservation audit, and GPU parallelism at once.

---

*End of record v0.1. Next artifact: the package declaration grammar (Phase 3), or the Phase 0 tree slice in Python + Taichi.*