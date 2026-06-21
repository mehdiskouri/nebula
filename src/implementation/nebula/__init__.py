"""
Nebula — a simulation-first 3D creation language (Phase 0: the tree slice).

Assets are not modeled; they are grown, simulated, and derived. A Nebula program
describes the *causes* (materials, processes, fields, laws, developmental programs);
the runtime solves for what those causes *imply*, at whatever (time, resolution) it
is asked for. See AGENT/ARCHITECTURE.md.

This package is the Phase-0 vertical slice — "the tree, completely" (ARCHITECTURE
Part VIII). It assembles the verified production-path mechanisms (ported from the
frozen verification oracles in src/verification/oracles/) into a clean library, plus
the new pieces the verification work did not need: the typed hypergraph substrate,
SDF/heightfield, the adaptive coarse-to-fine predicate, glTF export, and a minimal
XPBD solve.

Subpackages
-----------
- core        : the substrate — determinism, the typed hypergraph, the operator
                schema, and the field-agnostic conserved-bus runtime.
- fields      : implicit representations — signed distance field, heightfield.
- operators   : constitutive/transfer + growth operators (fire, integrators, growth).
- restriction : the keystone restriction/homogenization operator + the trust scalar.
- adaptive    : the one octree + the coarse-to-fine refinement predicate.
- mechanics   : minimal XPBD over Constraint hyperedges.
- geometry    : mesh extraction + glTF export (the only place a mesh appears).
- pipeline    : the end-to-end deterministic tree slice.
"""

__version__ = "0.1.0"
