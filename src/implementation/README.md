# Nebula — Phase 0 implementation (the tree, completely)

The Phase-0 **vertical slice** from `AGENT/ARCHITECTURE.md` Part VIII, built on the
verified mechanisms (`AGENT/verification_report.md`). It assembles the production-path
modules — **ported** from the frozen verification oracles in
`../verification/oracles/` — into a clean package, plus the pieces the verification
work did not need: the typed hypergraph substrate, SDF/heightfield, the adaptive
coarse-to-fine predicate, glTF export, and a minimal XPBD solve.

One deterministic scenario ties it together: **grow a tree → ignite it → simulate the
coupled burn with restriction + adaptive refinement → fracture the charred branches →
export the charred result to glTF — bit-reproducibly.**

## Run

```bash
export PYTHONPATH=src/implementation

# the end-to-end slice (grow → burn → restrict/refine → fracture → glTF) + determinism check
python -m nebula.pipeline.tree_slice

# the CLI
python -m nebula.cli grow-and-burn --seed 7 --age 22 --out tree.glb

# per-module self-checks (each reproduces its verified numbers)
python -m nebula.core.determinism
python -m nebula.core.hypergraph
python -m nebula.core.buses
python -m nebula.operators.fire
python -m nebula.operators.integrators
python -m nebula.operators.fire_taichi
python -m nebula.operators.growth
python -m nebula.fields.sdf
python -m nebula.fields.heightfield
python -m nebula.restriction.restriction      # + .homogenization .jensen .percolation
python -m nebula.adaptive.refine              # + .octree .octree_gpu
python -m nebula.mechanics.xpbd
python -m nebula.geometry.mesh_export

# regression: the ported package reproduces the frozen oracles bit-for-bit
PYTHONPATH=src/implementation:src/verification/oracles python -m regression.test_parity
```

## Package layout

| Module | Role | Provenance |
|---|---|---|
| `core/determinism` | fixed-order/integer reductions, stable blake2b hashing | port (V0.5) |
| `core/hypergraph` | Nodes (state SoA) + typed Hyperedges + categorize→color→flatten | **new** (Part II) |
| `core/schema` | the operator declaration schema (contributions / transitions) | **new** (§III.3) |
| `core/buses` | field-agnostic conserved-bus runtime + conservation audit | port+generalize (V0.3/V1.1) |
| `fields/sdf` | tree SDF (tapered-capsule union) + per-voxel layer classification | **new** |
| `fields/heightfield` | deterministic terrain | **new** |
| `operators/fire` | the 4 fire transfer laws + char-weakening, as a `Domain` | port (V0.3/V1.2/V1.3) |
| `operators/fire_taichi` | Taichi kernel for the conduction+Arrhenius field update (AOT bridge) | **new** (Part V) |
| `operators/integrators` | rate sub-stepping + semi-implicit IMEX (the stiff loop) | port (V1.2) |
| `operators/growth` | growth trace/memo/write-back + the **five growth/process operators** | port+extend (V1.8) |
| `restriction/homogenization` | Voigt/Reuss + directional estimate | port (V0.1) |
| `restriction/jensen` | sub-cell variance ε (the Jensen correction) | port (V1.3) |
| `restriction/percolation` | g_perc directional conductance residual + 26-conn backstop | port (V2.2) |
| `restriction/restriction` | the keystone: cell → trust scalar {gap, ε, g_perc} + lod_trust | **new** (§III.4) |
| `adaptive/octree`,`octree_gpu` | the one Morton octree (LOD/multigrid/far-field), CPU + Warp | port (V0.4) |
| `adaptive/refine` | the coarse-to-fine predicate: D + hysteresis + 2:1 + interface edge | **new** (Decision #10) |
| `mechanics/xpbd` | Constraint hyperedges solved by the color pass; char→fracture | **new** (added scope) |
| `geometry/mesh_export` | marching cubes → glTF, colour derived from χ/T | **new** (Decision #2) |
| `pipeline/tree_slice`,`cli` | the end-to-end deterministic scenario | **new** |

## The five growth/process operators (`operators/growth`)
1. **apical extension** — the meristem front advances the skeleton (the L-system walk)
2. **branching** — L-system + light-biased child spawning
3. **cambium rings** — secondary growth: per-season radius → taper + concentric ring layers
4. **reaction wood** — stress-driven (load × lever) asymmetric thickening; Wolff's law,
   the fully-stressed *multiplicative* update (V1.7)
5. **heartwood transition** — xylem older than `heartwood_age` → sapwood→heartwood

## Disciplines honoured (verification report §3) / ceilings budgeted (§5)
- fixed reduction order + hashed RNG + write-back in memo keys (determinism)
- two homogenization errors — the V-R **gap** *and* the variance **ε** — plus folded **g_perc**
- hysteresis + 2:1 balance + Interface (hanging-node) hyperedge on the adaptive octree
- cascade priority for transitions; the stiff fire loop runs on **IMEX / sub-stepping**, never
  the plain explicit path
- always-refine on the 26-connectivity defect span (the off-axis thin-connected tail)
- XPBD as energy/potential (gravity = a potential), guardrail #1

## Notes
- Backends: NumPy for the fire fields/restriction, **Taichi** (`cuda`) for the fire kernel,
  **Warp** for octree traversal + bus reductions, **cupy** for the conductance/RVE solves.
- The frozen oracles in `../verification/oracles/` are left untouched and kept as the
  regression ground truth (`regression/test_parity.py`).
- Biological/material parameters are representative plausibility-engine values, not clinical
  data (ARCHITECTURE Part VII) — Nebula is a plausibility engine.
