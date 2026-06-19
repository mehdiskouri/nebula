"""
Nebula verification — shared reference oracles (throwaway scaffolding, protocol §7).

These modules exist *only* to ground the verification notebooks: they are the
slow, simple, unshippable ground truth against which the cheap production path
is judged. They are reused across notebooks (V0.1, V0.2, V1.3, V2.2, V2.3, V2.1).

Contents
--------
- homogenization : Voigt/Reuss bounds + the directional proxy under test
- analytic       : closed-form laminate effective stiffness (validates the DNS oracle)
- dns_elasticity_3d : the DNS micro-solver (3D voxel periodic homogenization)
- cells          : deterministic microstructure generators (layers, char wedge)
- regulator      : minimal regulated cardiovascular loop + bounded reserve; the
                   phase-space / basin-of-attraction oracle for V1.5 (mortality)
- regulator_stability : inertial-actuator regulator + linear-stability / limit-cycle
                   oracle for V1.6 (passivity vs a naïve force controller)
"""
