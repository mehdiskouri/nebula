"""
Nebula mechanics -- minimal XPBD (ARCHITECTURE Part II: "a hyperedge IS an XPBD constraint";
Decision #5: categorize -> color -> flatten for race-free parallel solve; Part IV guardrail #1:
laws as energies/potentials, not raw forces).

A Constraint hyperedge is literally an XPBD constraint. The solve projects each color group
(node-disjoint by the hypergraph coloring) in parallel without write conflicts -- the same
race-free pattern as the conserved-bus reduce. The char-weakening transition S = S0(1-chi)
drives per-constraint compliance, so a burnt branch loses stiffness and fractures/falls: the
growth -> fire -> mechanics coupling, closed.
"""
