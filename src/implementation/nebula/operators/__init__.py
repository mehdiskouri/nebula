"""
Nebula operators (ARCHITECTURE §III.1, §III.3).

Constitutive/transfer operators (fire) and generative operators (growth), all expressed
against the conserved-bus discipline / hypergraph substrate. "Package the transfer
operator, not the phenomenon" (Decision #13): fire is the fixed point of combustion +
conduction + pyrolysis + char-weakening composing on shared buses.

- fire        : the four fire transfer laws + char-weakening transition, wired as a Domain.
- fire_taichi : Taichi GPU kernels for the conduction + Arrhenius field update (AOT bridge).
- integrators : rate-driven sub-stepping (multirate) + semi-implicit IMEX for the stiff loop.
- growth      : the field-biased growth front + the five growth/process operators.
"""
