"""
The restriction / homogenization operator -- the keystone (ARCHITECTURE §III.4; Decision #15).
Verified by V0.1 (bound validity, the keystone), V0.2 (criticality coincidence), V1.3 (Jensen
variance), V2.2 (percolation), V2.3 (geometric vs physical LOD).

Every other subsystem reads the single scalar this operator produces. It collapses a
heterogeneous cell into one effective element plus a TRUST bound that simultaneously gates
refinement, conservation tolerance, surrogate trust, and LOD ("one scalar, four jobs").

The bound carries the errors the verification proved are real and necessary:
  - homogenization (Voigt-Reuss directional gap, constitutive RESPONSES)        [homogenization]
  - nonlinear-rate sub-cell variance epsilon (the Jensen term)                   [jensen]
  - connectivity (the directional scalar-conductance residual g_perc)           [percolation]
  - the physics-weighted refine-vs-truncate gate lod_trust = gap x (1 + g_perc)  [restriction]
"""
