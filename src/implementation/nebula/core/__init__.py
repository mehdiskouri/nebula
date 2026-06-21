"""
Nebula core substrate (ARCHITECTURE Parts II–IV).

The single data structure under everything (the typed hypergraph), the single
compute pattern (gather -> stage into buses -> reduce -> commit), and the single
discipline that makes the program *be* the asset (determinism via fixed reduction
order + stable hashing).

- determinism : fixed-order / integer-exact reductions + stable blake2b hashing.
- hypergraph  : Nodes (state, SoA) + typed Hyperedges (law) + categorize/color/flatten.
- schema      : the operator declaration schema (the contract the bus runtime dispatches).
- buses       : the field-agnostic conserved-bus runtime + conservation audit.
"""
