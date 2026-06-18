# Shared verification oracles

Throwaway reference scaffolding for the Nebula verification protocol (`AGENT/verification_protocol.md` §7).
These modules are the **independent ground truth** the notebooks judge the cheap
production proxies against — deliberately slow, simple, and unshippable. They are
*not* product code.

| Module | Role | Used by |
|---|---|---|
| `dns_elasticity_3d.py` | 3D voxel periodic-homogenization solver → true effective 6×6 stiffness | V0.1, V0.2, V1.3, V2.2, V2.3 |
| `analytic.py` | closed-form Backus laminate stiffness (validates the DNS solver) | V0.1 |
| `homogenization.py` | Voigt/Reuss bounds, the directional proxy, gap + containment checks (the proxy *under test*) | V0.1, V0.2 |
| `cells.py` | deterministic microstructure generators (layered shells, char wedge) | V0.1, V0.2, V2.2 |

## Conventions
- Pure `numpy` / `scipy`; Voigt notation `[11,22,33,23,13,12]`, engineering shear.
- Flat imports (run as scripts or import after putting this dir on `sys.path`).
- Each module has a `__main__` smoke test: `PYTHONPATH=. python <module>.py`.
- Deterministic: fixed assembly/reduction order, seeded generators.

## Status
The DNS solver reproduces the homogeneous identity and the analytic laminate to
~1e-14 and is bit-reproducible on repeat runs.
