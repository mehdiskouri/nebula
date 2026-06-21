"""
The Phase-0 vertical slice (ARCHITECTURE Part VIII): the tree, completely.

tree_slice wires every subsystem into one deterministic scenario -- grow a tree (5 growth
operators) -> build its SDF/heightfield -> ignite and burn on the conserved bus (stiff-loop
integrator) -> restrict each cell to the single trust scalar and adapt resolution (the
coarse-to-fine predicate) -> fracture charred branches (XPBD) -> export a glTF whose colour is
derived from the simulation. Running it twice yields a bit-identical scene digest ("the program
IS the asset").
"""
