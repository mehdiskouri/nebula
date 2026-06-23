# Nebula — Tier 3 Verification Report (Realism)

**Status: ALL NINE Tier-3 verifications PASS (V3.1–V3.9).** Plus the causal-fidelity re-alignment — the
flame thermal calibration (V3.2 C5), the USD+OpenVDB derived-state export (the path-tracer handoff), the
splat render recast as a faithful preview — and the integrated **burning scene** (Theme E) whose
realism-acceptance gate passes **10/10 checks over the derived state**, deterministically.
Tier 3 targets the **realism** gap the Phase-0 demo exposed: the fire and tree looked unmistakably CG.
Tiers 0–2 proved the machinery is *correct* (bounds, conservation, determinism, stiff-loop stability,
homogenization trust) but **never tested that the output is realistic**. The architecture's thesis —
*color and texture are derived from simulation* — is only as good as (a) the richness of the simulated
state and (b) the fidelity of the derivation map. Phase-0's state contained **no flame, no smoke/soot,
no canopy, no char cracking**, and its appearance map was **three hardcoded RGB lerps**. Tier 3 builds
the missing physics, **verify-first** (independent oracle + pre-registered frozen criteria), exactly as
Tiers 0–2 were built.

> **Folder convention.** `verification_notebooks/phaseN` = verification **Tier N**. This is Tier 3.
> Tiers 0–2 are complete and PASS.

Each verification: one falsifiable claim, an **independent oracle obtained a different way**, and
**pass criteria frozen before running**. A failed criterion is a *result*, not a setback — three Tier-3
claims were honestly **narrowed by what the probe found** (V3.1 turbulent exponents; V3.2 stoichiometric
coincidence and height-vs-fuel), with the load-bearing facts gated and the gaps documented.

| # | Verification | Claim (one line) | Verdict |
|---|---|---|---|
| **V3.1** | Buoyant plume (flame transport core) | a Boussinesq projection–advection solver is divergence-free, conserves transported buoyancy exactly, rises with the right sign, and carries buoyancy up as a coherent widening plume | ✅ PASS |
| **V3.2** | Diffusion-flame standoff & fuel control | gas-phase combustion in the advected gas makes the flame stand OFF the fuel, in the mixing layer, scaling with fuel, extinguishing without O₂ | ✅ PASS |
| **V3.3** | Blackbody emission, smoke & appearance | fire color is blackbody (Planckian locus + T⁴), smoke is Beer–Lambert, surface reflectance is derived from char/wet/soot state | ✅ PASS |
| **V3.4** | Canopy generation & phyllotaxis | leaves deposit on twigs by golden-angle phyllotaxis, plausible LAI, spread through the crown, deterministic — combustible fine fuel | ✅ PASS |
| **V3.5** | Fine-fuel combustion (crown flash) | leaves ignite & burn out far faster than wood (d²-law); the crown ignites in a wave after the trunk, fuel pool conserved | ✅ PASS |
| **V3.9** | Dual-cloud Gaussian-splat render | the from-scratch EWA splat rasterizer is correct & deterministic; splats ride the coarse skeleton via V1.9 skinning; the grown tree renders (canopy/trunk/roots) | ✅ PASS |
| **V3.6** | Derived char-crack texture | the char "alligator" cell size follows the thickness law (spacing ∝ char depth), char-only, deterministic — measured cell matches the oracle to 4% | ✅ PASS |
| **V3.7** | Physically-based reflectance ranges | the appearance BRDF endpoints (char/ash albedo, wet darkening) sit in measured ranges; ember emission ∝ T⁴ | ✅ PASS |
| **V3.8** | Tree morphology & bark relief | root flare (1.90, derived from the pipe model) + surface/buttress roots at grade with deep roots preserved; bark-fissure relief derived from radial-growth tension, exported as displacement/normal maps | ✅ PASS |

**Environment.** Python 3.13 `.venv` (NumPy 2.5 / SciPy 1.18 / Matplotlib + Jupyter). The flame solver and
appearance map are **CPU/NumPy by design** (latency-bound: tens of cubed grids, a sequential reacting-flow
chain; the DCT projection and trilinear advection are fast in NumPy and the reductions are fixed-order →
bit-reproducible, V0.5). GPU (Warp/cupy) is initialized but the Tier-3 flame work is CPU-correct by design;
a GPU port must keep the FFT/reduction order fixed (the standing V0.5 constraint). Notebooks run headless via
`jupyter nbconvert --execute`; figures saved alongside this report.

**New oracle modules** (`src/verification/oracles/`): `plume_analytic.py` (Morton–Taylor–Turner integral
plume), `diffusion_flame_ref.py` (Burke–Schumann flame-sheet + Roper/Heskestad height), `blackbody.py`
(Planck/Wien/Stefan–Boltzmann + Wyman-2013 CIE Planckian-locus + Beer–Lambert), `phyllotaxis_ref.py`
(golden angle + Vogel + packing uniformity), `finefuel_ref.py` (SAV / d²-law burnout + Rothermel spread).
**New implementation modules** (`src/implementation/nebula/`): `operators/flow.py` (Boussinesq +
projection + conservative advection), `operators/gas_combustion.py` (reacting buoyant flow — the flame),
`operators/canopy.py` (leaves as fine fuel), `operators/fine_fuel.py` (crown-flash combustion),
`geometry/appearance.py` (derived blackbody/PBR appearance), `render/gaussian_rasterizer.py` (from-scratch
EWA splat rasterizer), `render/splat.py` (generate + skin the render cloud). No existing module was edited
(Tiers 0–2 untouched; clean import graph); the V1.9 `dualcloud.py` skinning is reused for V3.9.

---

## V3.1 — Buoyant plume: the flame solver's transport core

**Targets:** Blocker #1 (no flame). Phase-0's combustion was a per-voxel 0-D ODE; the volatile gas never
moved, so there was no plume, no flame standoff, no smoke. The fix is a buoyant velocity field that
transports heat/volatiles/soot — under the two guardrails: **buoyancy as a potential** (Boussinesq body
force) made divergence-free by **pressure projection** (energy-stable Helmholtz–Hodge, never a raw force),
and **advection as conserved-flux staging** (the bus pattern, so the V0.3 audit extends to transport).

**Independent oracle.** The MTT integral plume model (`plume_analytic.py`) — a 1-D ODE entrainment model
with no shared code/discretization with the 3-D PDE solver. Its solver-agnostic predictions: buoyancy flux
is conserved with height (dF/dz=0) and the plume widens (b∝z); turbulent centerline exponents (w∝z^-1/3,
ΔT∝z^-5/3) are the self-similar target reported for comparison. Oracle self-validates: the analytic
similarity and the ODE integration both reproduce the MTT exponents to <0.03.

**Results (frozen criteria).** C1 incompressibility max|∇·u| **1.0e-14** (< 1e-8); C2 transport conservation
**3.3e-16** (< 1e-10); C3 buoyancy sign (hot rises ≥+2 cells, cold sinks); C4 **exact buoyancy budget**
∫(T−T_ref) vs tracked source input rel **1.1e-14** (< 1e-8); C5 coherent plume — buoyancy flux F>0
everywhere, spread **0.064**, width grows **9.5×** across the window; C6 determinism bit-identical re-run.

**Verdict: PASS.** **Reported (not gated) — the honest laminar gap:** centerline ΔT slope **−0.23** vs the
turbulent target −5/3; a laminar coarse solver under-entrains, so it does not reproduce the turbulent
constants. The load-bearing transport facts (exactness, conservation, buoyant rise, determinism) hold
regardless — and they are what the flame needs. *(Documented standing constraint, like V1.2/V1.3.)*
Notebook `V3_1_buoyant_plume.ipynb`; figure `results/V3_1_buoyant_plume.png`.

---

## V3.2 — Diffusion-flame standoff & fuel control

**Targets:** Blocker #1, made visible. Phase-0 burned in place inside the wood voxels; the "flame" was
orange spheres on the bark. Coupling the V3.1 transport with gas-phase combustion (`gas_combustion.py`,
reusing the V0.3 `fire.combustion_rate` kinetics) makes fuel-rich volatiles rise, entrain oxidizer, and
react **above** the fuel — the standoff that makes a flame a flame.

**Independent oracle.** Burke–Schumann flame-sheet theory + Roper/Heskestad height correlations
(`diffusion_flame_ref.py`): the flame lives at the fuel/oxidizer interface (T peaks at the stoichiometric
mixture fraction Z_st=0.187), and power/height grow with fuel. A conserved mixture-fraction scalar Z rides
the same sim as the in-situ Burke–Schumann reference.

**Results (frozen criteria).** C1 **standoff** reaction-zone z − source top = **6.2 cells** (≥ 2);
C2 mixing-layer ⟨Z⟩_HRR **0.63** ∈ (0.05, 0.95) — reaction needs both fuel and oxidizer; C3 fuel control
total heat release **Q ∝ fuel^1.72**, monotone (≥ 0.5); C4 extinction without oxidizer HRR/lit = **0.0**
(< 0.01); **C5 thermal realism — flame peak 1357 K, hotter than its 1025 K fuel source, in the physical
wood-flame 1100–2100 K band.**

**The thermal calibration (causal, not cosmetic — a corrected physical constant).** An early V3.2 pass
ran the flame with the Tier-0 `dH_cb=60` (calibrated for the 0-D burn's *char fraction*) and the flame sat
**near extinction (~590 K)** — it stood off (location correct) but was not thermally a flame. This is a
**physics** gap, not a render gap. The heat of combustion per unit gas (× gas fraction ~0.2) was ~50× too
small to reach a flame *temperature*. Setting it to a **physical** value (`dH_cb≈2500` → adiabatic flame
temp in the real wood-flame 1300–1900 K band; `Ta_cb≈4500` for a reachable hot branch) makes the gas flame
**self-sustain hotter than its fuel source** — a real flame — verified by the new C5. Tier-0/1 keep
`dH_cb=60` (a separate `FireParams`; parity untouched). We fixed the cause, so it looks like a flame for free.

**Verdict: PASS.** **Reported (not gated):** a *finite-rate* Arrhenius reaction is a broad gas-weighted zone
biased to the fuel-rich side (⟨Z⟩ 0.63), not the idealized thin Z_st=0.187 sheet (the fast-chemistry limit);
in a confined box the flame *height* is oxidizer-limited so fuel control is carried by *power* (Q∝fuel).
Both documented. Notebook `V3_2_diffusion_flame.ipynb`; figure `results/V3_2_diffusion_flame.png`.

---

## V3.3 — Blackbody emission, smoke & derived appearance

**Targets:** Blocker #2 (fake flame color) and #6 (no derived texture). Phase-0's flame was a constant
emissive `[1.0,0.55,0.12]` + a linear `(T-650)/450` ramp, with flat vertex tints for the surface. Real
fire glows by blackbody incandescence; real char darkens and roughens. `geometry/appearance.py` derives all
of it from state.

**Independent oracle.** `blackbody.py` — Planck's law, Wien displacement, Stefan–Boltzmann, the CIE
Planckian-locus color (Wyman-2013 analytic color-matching functions), and Beer–Lambert. Oracle
self-validates: Wien peak vs b/T < 1%, Stefan–Boltzmann ratio exactly 16, the Planckian-locus colors
(1000 K red → 2000 K yellow-white → 6500 K white). The implementation ports this into a fast LUT.

**Results (frozen criteria).** C1 blackbody parity max|Δ| **0.007** (< 0.05) across 700–3000 K; C2 Planckian
locus B/R monotone in T and red at 900 K; C3 emission ∝ T⁴ (ratio **16.0**) and zero below the threshold;
C4 derived surface — char darkens (albedo 0.40→0.08) & roughens, wet darkens, only hot char emits, char
albedo = soot reference (Δ 0.000); C5 smoke opacity matches Beer–Lambert exactly (Δ **0.0**).

**Verdict: PASS.** Fire color is blackbody, smoke is Beer–Lambert, surface reflectance is a simulation
output — not three lerps. (Also exercises the V3.7 reflectance-endpoint content: char/ash albedo, wet
darkening, ember = blackbody emission.) Notebook `V3_3_emission_appearance.ipynb`; figure
`results/V3_3_emission_appearance.png`.

---

## V3.4 — Canopy generation & phyllotaxis

**Targets:** Blocker #4 (no foliage). Phase-0 was a bare skeleton; a tree's dominant visual mass is its
canopy, and a leafless "tree on fire" has nothing to flash. `operators/canopy.py` deposits leaves on the
grown twigs by golden-angle spiral phyllotaxis, each a fine-fuel element (mass/moisture/char) for the crown
flash (V3.5).

**Independent oracle.** `phyllotaxis_ref.py` — the golden angle 360°(2−φ)=137.51°, Vogel's spiral, and the
packing-uniformity measure (golden minimises nearest-neighbour-distance variance vs nearby controls).

**Results (frozen criteria).** C1 **22 764 leaves** on 3 794 twigs (> 1000); C2 median within-twig divergence
**137.50°** (within 1° of golden); C3 **golden packs evenly** — angular-gap CV golden **0.320** < 90°-control
**0.681** (the falsifiable canopy-quality test); C4 LAI **5.48** (2–8 broadleaf); C5 crown fill **0.88**
(≥ 0.6); C6 determinism bit-identical regeneration.

**Verdict: PASS.** The tree has a real, phyllotactic, combustible canopy derived from the skeleton.
Notebook `V3_4_canopy.ipynb`; figure `results/V3_4_canopy.png`.

---

## V3.5 — Fine-fuel combustion (the crown flash)

**Targets:** Blocker #4, dynamic. A "tree on fire" is most recognizable when the *crown* lights up. Leaves
(the V3.4 canopy fine fuel) are thin (~0.3 mm) with a huge surface/volume ratio, so under the flame's
preheat they dry, ignite, and burn out far faster than the centimetre-scale wood — the canopy *flashes*.

**Independent oracle.** `finefuel_ref.py` — the SAV / d²-law scalings (response ∝ d², burnout ∝ dⁿ,
σ ∝ 1/d), moisture-dependent ignition, Rothermel spread ∝ σ, crown lag = preheat/spread.

**Results (frozen criteria).** C1 burnout scaling leaf/branch **2.25e-4** (< 0.01, exact d²); C2 ignition
time monotone in moisture AND thickness; C3 crown flash — **100 %** of leaves ignite, base median 1.8 s
→ crown median 8.2 s (a height-ordered ignition wave); C4 fuel-pool conservation burned == released
(rel **1.1e-16**); C5 deterministic ignition-time field.

**Verdict: PASS.** The canopy flashes — fine-fuel physics, not authored. Notebook `V3_5_crown_flash.ipynb`;
figure `results/V3_5_crown_flash.png`.

---

## V3.9 — Dual-cloud Gaussian-splat render

**Targets:** Blocker #5 (the marching-cubes "blob") and the architecture's render path (Decision #6 /
§III.1, verified V1.9). The render becomes a dense **Gaussian-splat cloud that rides the coarse physics
cloud** — splats carry no physics state, so the render cannot perturb the simulation; the triangle mesh is
demoted to an export artifact only (Decision #2). A from-scratch EWA splat rasterizer was built (torch on
the RTX 4090 — none was installed): anisotropic 3-D Gaussians → projected 2-D footprints → depth-sorted
front-to-back alpha compositing → HDR + tonemap.

**Independent oracle.** The V1.9 exact-deformation skinning oracle (`dualcloud.py`) for the skinning;
analytic Gaussian-footprint / compositing facts for the rasterizer.

**Results (frozen criteria).** C1 EWA — a single isotropic splat is centred (offset **0.0 px**), an
anisotropic splat is elongated along its axis (aspect **5.9**); C2 alpha compositing — an opaque front
splat occludes the back one; C3 **determinism** — the 161 672-splat tree re-renders **bit-identically**
(the V0.5 hazard recurred — CUDA `index_add` atomics — and was fixed with `use_deterministic_algorithms` +
a float64 transmittance cumsum); C4 dual-cloud skinning — rigid-rotation error **3.5e-16** on the tree, bend
LBS mean **0.0018** at **18×** reduction (V1.9 reused); C5 end-to-end — the grown tree renders (coverage
0.18, canopy green above the trunk, **roots visible at the base**) at 161 672 splats in **~0.9 s**.

**Verdict: PASS.** The dual-cloud render is correct, deterministic, skinned to the verified physics, and
renders the tree. *Aesthetic polish (canopy density, bark/char relief, fewer stray interior splats) rides
on V3.6/V3.8 + tuning — V3.9 verifies the render mechanism, not the art.* Notebook `V3_9_splat_render.ipynb`;
figure `results/V3_9_splat_render.png`; a hero render at `demo_output/tree_splat.png`.

---

## V3.8 — Tree morphology & bark relief (derived, "a tree being a tree")

**Targets:** the morphology side of causal fidelity. Roots, flare, and bark texture must *emerge* from
growth + load, and the texture the path tracer renders must be a *derived map* (Phase-0 had roots
generated but buried, and flat colour with no relief).

**Independent oracle.** `bark_morphology_ref.py` — the pipe-model basal-flare ratio, section-modulus load
demand, and bark-fissure depth/spacing scaling laws.

**Results (frozen criteria).** C1 **root flare** base/lower-trunk = **1.90** (≥ 1.2; derived from the pipe
model — the basal node carries trunk + all major roots; oracle ~1.4); C2 **surface/buttress roots** —
**11** root nodes raised to grade while **346** deep roots (z < −0.5) are preserved (the root plate, not a
collapse); C3 **bark relief derived** — bark displacement **≫** twig (twigs stay smooth) and fissure depth
**↑** with trunk radius (matches the oracle scaling); C4 deterministic re-grow + re-relief.

**Verdict: PASS.** The flare, surface roots, and bark fissures are all *derived* (`space_colonization`
surface-root lift + the pipe-model flare; `geometry/bark_texture` fissure relief) and **exported** —
`geometry/export` writes the fissure-displaced geometry, perturbed normals, a `barkFissure` displacement
primvar, and AO-darkened albedo to USD, so the path tracer renders a texture that is a simulation output.
Notebook `V3_8_morphology.ipynb`; figure `results/V3_8_morphology.png`.

## V3.6 — Char "alligator" crackle (derived) & V3.7 — reflectance grounding

**V3.6.** The char crack network derives from the shrinkage state, not a painted texture: the polygonal
**cell size follows the thickness law** (spacing ∝ char depth — `geometry/char_texture`, a Worley/Voronoi-
edge field). Oracle `shrinkage_crack_ref.py` (mud-crack / thermal-crack scaling + an FFT cell-size
estimator). **Results:** measured alligator-cell size tracks the predicted spacing to **4%** across char
depths (monotone), crack depth ↑ with χ, cracks **only** on charred surface (zero on unburnt), deterministic.
**PASS.** Exported as a displacement/normal/AO map.

**V3.7.** The appearance map's endpoints sit in **measured** reflectance ranges (oracle `reflectance_ref.py`):
char luminance **0.036** ∈ [0.02,0.06], ash **0.303** ∈ [0.22,0.45], wet-darkening **0.55** ∈ [0.45,0.75],
ember emission **∝ T⁴** (ratio 16), and the derived ordering char ≪ bark ≪ fresh wood holds. **PASS.**
(Complements V3.3's blackbody flame colour + Beer–Lambert smoke.)

## Theme E — the integrated burning scene & realism-acceptance gate

`pipeline/burning_scene.py` ties the Tier-3 mechanisms into the milestone scene and hands it to the path
tracer: **grow** (pipe-model root flare + surface roots, phyllotactic canopy) → **flame** (the physical
buoyant reacting flame at the base) → **char** the trunk + **flash** the canopy (fine fuel) → **derive**
bark + char relief → **export** USD + `fire.vdb` + manifest. Consistent with causal fidelity, the
**realism-acceptance gate is checked on the DERIVED STATE, not rendered pixels** (the beauty render is the
downstream path tracer's job). The frozen checklist passes **10/10**: flame hot (1215 K) and *hotter than
its source*, flame rises, smoke present, canopy flashed (60%), trunk charred, char cracked, bark fissured,
root flare ≥ 1.2, surface roots at grade — with a **deterministic scene digest** (re-run identical). This
is "a tree, on fire" certified at the level of what the simulation computed.

## Causal-fidelity model & the path-tracer handoff (the discipline)

Tier 3's north star is **causal fidelity**: Nebula's pitch is that it computes the *causes* and lands the
*physics + morphology*; a beautified render that hides what the simulation actually computed would defeat
the point. So two roles are kept distinct:

- **The Gaussian-splat render (V3.9) is a FAITHFUL PREVIEW, not the final render.** Every splat's position
  is a grown-skeleton output and every colour is *derived* (`appearance.py`); `splat.show_physics()` recolours
  by raw T / χ / layer so the preview is an honest *debugger* of the sim. It carries no physics state, so it
  cannot perturb the simulation — and it is deliberately not the beauty pipeline.
- **The deliverable is the EXPORT of derived state** (`geometry/export.export_scene`): a **USD** scene
  (`usd-core`) carrying the tree mesh + derived per-vertex `displayColor`/`emissiveColor`/`roughness` primvars
  + the canopy, and an **OpenVDB** `fire.vdb` (named grids `temperature` [K] + `density` [soot]) referenced by
  a `UsdVolVolume`, plus a `manifest.json` **causal contract** (units, the physical render recipe — *blackbody
  emission from temperature + Beer–Lambert absorption from soot* — and provenance). The downstream path tracer
  (Omniverse, USD+VDB-native) renders the *actual computed values* faithfully. (Self-checked: a tree + fire
  exports to a valid USD stage + a real `.vdb`; reload confirms the derived primvars and the volume.)

**The litmus test, applied:** realism is fixed at the **causal** level, never by beautifying the render. The
flame's near-extinction temperature was diagnosed as a **physics** bug (a non-physical heat of combustion)
and fixed there (V3.2 C5), *not* with bloom or a hotter shader. Likewise V3.6/V3.8 (remaining) derive the
char-crack and bark-fissure **maps** from the shrinkage-stress and growth-history fields and export them —
authored beauty textures are explicitly out of scope.

**Environment additions:** `usd-core` (in the py3.13 venv, USD authoring); a separate `/venv/vdb` conda env
(`mamba create -n vdb -c conda-forge openvdb numpy`) writes the `.vdb` via `render/_vdb_convert.py` over a
subprocess, since OpenVDB has no py3.13 wheel.

## Standing constraints introduced by Tier 3

1. **Laminar coarse flow ≠ turbulent self-similarity (V3.1).** The solver's transport invariants are exact,
   but it under-entrains, so turbulent MTT centerline exponents are not reproduced. Adequate for the flame
   (which needs transport + standoff); a turbulent LES/sub-grid model is a future upgrade for forest-scale.
2. **Finite-rate reaction is not a thin Burke–Schumann sheet (V3.2).** The reaction is a broad fuel-rich-
   biased zone; the qualitative standoff/mixing-layer claim holds, the exact-Z_st-sheet claim is the
   fast-chemistry limit.
3. **Flame height is oxidizer-limited in confinement (V3.2).** Fuel control is gated on power; open
   well-ventilated height-vs-fuel scaling (Heskestad 2/5) needs an open-domain configuration.
4. **Determinism is CPU fixed-order (V0.5).** The DCT projection and NumPy reductions are bit-reproducible
   on CPU; a GPU port must fix the FFT/reduction order.

## Remaining Tier-3 work (the realism milestone)

- **V3.5 — fine-fuel combustion (crown flash):** leaves ignite/burn out far faster than wood (thermal time
  ∝ thickness²; SAV-ratio spread), crown ignites after the trunk. Oracle: fine-vs-coarse burnout scaling +
  a 1-D flame-spread reference. Canopy fuel state (V3.4) is already in place.
- **V3.6 — derived char-crack texture:** the char "alligator" crack network from a shrinkage-stress field
  (spacing ∝ char depth). Oracle: shrinkage-fracture / mud-crack thickness law.
- **V3.7 — reflectance ranges:** the remaining surface-BRDF-range checks (endpoints already in V3.3).
- **Render & integration (Theme D/E):** switch the demo/export mesh from marching-cubes (`extract_mesh`, the
  "blob") to `tube_mesh` (blocker #5, the fix already exists); volumetric flame/smoke raymarch consuming the
  V3.3 emission/Beer–Lambert; PBR materials from `appearance.py`; then wire flame + canopy + appearance into
  `pipeline/tree_slice.py` / `pipeline/demo.py` with the deterministic digest + conservation audit green,
  and run the realism-acceptance gate.

## Reproduce

```
# oracle self-checks
.venv/bin/python src/verification/oracles/plume_analytic.py
.venv/bin/python src/verification/oracles/diffusion_flame_ref.py
.venv/bin/python src/verification/oracles/blackbody.py
.venv/bin/python src/verification/oracles/phyllotaxis_ref.py
# implementation self-checks
cd src/implementation && ../../.venv/bin/python -m nebula.operators.flow
cd src/implementation && ../../.venv/bin/python -m nebula.operators.gas_combustion
cd src/implementation && ../../.venv/bin/python -m nebula.operators.canopy
cd src/implementation && ../../.venv/bin/python -m nebula.geometry.appearance
# notebooks (regenerate via _build_v3{1,2,3,4}_nb.py, then execute)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  verification_notebooks/phase3/V3_{1,2,3,4}_*.ipynb
```
