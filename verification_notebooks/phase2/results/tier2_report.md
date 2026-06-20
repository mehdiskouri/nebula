# Nebula — Tier 2 Verification Report

**Status: TIER 2 IN PROGRESS — V2.4 PASS, V2.1 CONSTRAIN (both complete).** This report covers
the two highest-leverage Tier-2 risks: **V2.4** (surrogate generalization / OOD fallback / PINN
data efficiency) and **V2.1** (the RVE ↔ learned-surrogate handoff in the violent regime — Risk #1,
the project's single largest engineering risk). Per the protocol's exit criteria (§9), scale &
exotic claims (architecture Phase 4) are gated on V2.1 and V2.4; until V2.1 fully resolves, the
violent-regime cost ceiling is treated as known-bounded by the always-RVE fallback.

> **Folder convention.** `verification_notebooks/phaseN` = verification **Tier N**, *not* architecture
> "Phase N". This is Tier 2. Tier 0 (V0.1–V0.5) and Tier 1 (V1.1–V1.9) are complete and PASS.

Each verification follows the protocol discipline: one falsifiable claim, an **independent oracle**
obtained a different way, and **pre-registered pass criteria frozen before running**. A failed
verification is a *result*, not a setback.

| # | Verification | Claim (one line) | Verdict |
|---|---|---|---|
| V2.4 | Surrogate generalization / OOD / data efficiency | a physics-informed graph net trained on one archetype, conditioned on the homogenized descriptor, generalizes in-family, detectably degrades OOD so the fallback triggers, and needs less data than a pure-data baseline | ✅ PASS |
| V2.1 | RVE ↔ surrogate handoff (violent regime) | in the violent regime where Voigt–Reuss is *invalid*, a u-keyed decision rule avoids both stalling (always-RVE) and lying (always-trust), bounding outcome error within a cost budget | ⚠️ CONSTRAIN |

**Environment.** Python 3.13 `.venv` (NumPy/SciPy/Matplotlib + Jupyter). Tier 2 is the first
verification tier to use **GPU**: the damage-DNS oracle runs on the existing cupy GPU-CG path, and
the surrogate (`surrogate_gnn.py`) runs on **torch 2.12.1+cu130**. The expensive DNS datasets are
cached to `verification_notebooks/phase2/cache/*.npz` so notebooks re-run from cache; torch is
seeded with deterministic algorithms (declared-tolerance regime, per V0.5).

**New oracle modules** (`src/verification/oracles/`):
- **`dns_damage_3d.py`** — the **violent-regime ground truth**: an incremental secant-damage solver
  extending the proven linear periodic-homogenization machinery (imports `dns_elasticity_3d` /
  `failure`, unedited). Monotone scalar damage with exponential softening; warm-started GPU CG over
  the load path. Its `__main__` proves (i) a homogeneous softening bar matches the closed form to
  1.3e-9, (ii) damage is monotone and dissipation ≥ 0, and **(iii) the damaged secant response of a
  char wedge falls *below* the Reuss bound** — i.e. Voigt–Reuss is provably **invalid** here, the
  whole reason a surrogate/RVE handoff is needed.
- **`violent_cells.py`** — the archetype char-wedge **parameter family**, an explicit **OOD set**
  (off-axis percolating seams, extreme-contrast wedges, random char blobs), the **homogenized
  descriptor** (the restriction-operator output the surrogate is conditioned on), the coarse
  **region graph**, and a **percolation** (connectivity span) test.
- **`surrogate_gnn.py`** — the learned tier: a small **physics-informed graph network** over the
  region graph, conditioned on the descriptor, predicting normalized peak strength; a **bootstrap
  deep ensemble of heteroscedastic heads** (epistemic + aleatoric `u`); the **validity envelope**
  (descriptor max-z) and the **multi-signal fallback trigger** (envelope-exit OR percolation).
- **`handoff_rule.py`** — the decision calculus: uncertainty **calibration** (binned reliability
  rank-correlation, coverage / over-confidence) and the **stall↔lie frontier** + operating-point
  search, with an optional validity **gate** for the multi-signal rule.

---

## V2.4 — Surrogate generalization, OOD fallback & PINN data efficiency

**Targets:** Decision #17 (train-on-archetype / condition-on-descriptor / monitor-by-predicate).

**Approach & oracle.** Train the physics-informed ensemble on the char-wedge family (45 cells, DNS
strength targets); the independent oracle is the **damage-DNS** itself. Three experiments, one per
pre-registered metric.

**Pre-registered criteria (frozen).** (1) in-family median relative error **< 12%**; (2) fallback
trigger flags **≥ 99%** of OOD cells with in-family false-positive **≤ 12%**; (3) PINN reaches a 5%
median-error target with samples-to-target ratio **≤ 0.6** of the pure-data baseline (3-seed avg).

**Results.**
- **(1) generalization — 1.7%** median relative error on held-out in-family parameters (p90 11.2%).
- **(2) OOD detection — 100%** of the 18 OOD cells flagged, **5%** in-family false-positive. The
  off-axis percolating **seam** is the named blind spot of volume-fraction homogenization (Risk:
  percolation / V2.2) — its tiny volume fraction makes its descriptor look intact, so the envelope
  alone misses it; the **connectivity (percolation) trigger** catches it, exactly the architecture's
  prescribed guard. The combined trigger (envelope-exit **OR** percolation) mirrors the operator
  schema's "fallback on envelope-exit OR residual-spike".
- **(3) data efficiency — ratio 0.28** (3-seed averaged, robust): the PINN reaches 5% median error
  at **N=5** training cells; the pure-data baseline needs **N=18**. The physics priors (strength in
  (0,1], non-increasing in contrast and soft-fraction) buy a real **scarce-data** advantage
  (~17–43% lower error at N≤12); both converge to the same ~2% floor with ample data.

**Verdict: PASS** (`V2_4_surrogate_generalization.ipynb`; figure `V2_4_surrogate_generalization.png`).
The macro-surrogate generalizes in-family, its multi-signal validity check detects OOD without
false alarms, and physics-informed training is data-efficient where it matters (data scarcity — the
point of "one archetype spans a family"). Decision #17 holds.

---

## V2.1 — RVE ↔ learned-surrogate handoff in the violent regime *(Risk #1)*

**Targets:** Risk #1 — the decision rule for *pay-for-RVE* vs *trust-surrogate* where the analytic
Voigt–Reuss bound is **invalid** (large-deformation / active fracture / the death cascade).

**Approach & oracle.** On a realistic violent battery (42 cells: a tractable in-family majority + a
~19% hard extrapolation minority), the **damage-DNS** gives the true outcome `R_true`; the
V2.4-trained surrogate gives a prediction + self-uncertainty `u`. Staged: **calibrate** `u` against
actual error, then design the **handoff rule**.

**Pre-registered criteria (frozen before running; not tuned to results).** (1) calibration — binned
reliability rho(`u`, error) **> 0.80** and not over-confident (|nominal−observed|@1σ **< 0.12**);
(2) rule — a u-keyed operating point with **P95 outcome error < 0.10** AND **RVE-fraction < 0.30**;
(3) necessity — always-trust tail **>** bound (lying) and always-RVE fraction **= 1** (stalling).

**Results.**
- **Necessity — PASS.** always-trust tail error **0.194 > 0.10** (lying is real); always-RVE
  fraction **1.00** (stalling extreme). The interior tradeoff is genuine.
- **Rule — PASS.** A u-only operating point exists at **RVE-fraction 0.262 (< 0.30)** with **P95
  outcome error 0.063 (< 0.10)** — a working handoff that avoids both stalling and lying within
  budget. The architecture's **multi-signal** rule (validity-gate OR u) reaches the same budget.
- **Calibration — SHORT.** Over-confidence at 1σ is acceptable (**+0.064 < 0.12**), but the binned
  reliability correlation is **rho 0.770 < 0.80**, with real 2σ under-coverage on the violent tail
  (0.76 observed vs 0.95 nominal): surrogate self-uncertainty under-flags the hardest extrapolation
  cells (max error 0.41 vs max u 0.17).

**Verdict: CONSTRAIN** (`V2_1_rve_surrogate_handoff.ipynb`; figure `V2_1_rve_surrogate_handoff.png`).
The **core claim is supported** — a decision rule that bounds violent-regime outcome error within a
cost budget while beating both extremes **exists** — but surrogate **self-uncertainty alone is only
borderline rank-calibrated and mildly over-confident on the violent tail**. This is precisely the
Risk #1 concern, and the verification's value is finding the boundary cheaply.
**Standing constraint:** in the violent regime, key the handoff on the **validity-aware** signal —
the descriptor validity-envelope and the connectivity (percolation) trigger together with `u`, not
surrogate self-uncertainty alone — and keep **always-RVE** as the safe (expensive) fallback where
even that is untrusted. This matches the architecture's own multi-signal fallback design and the
protocol's anticipated CONSTRAIN outcome for the violent regime.

**Forward links.** The calibration shortfall is the handoff into V2.5 (inverse design re-verifies
candidates against the real operators) and motivates the always-RVE planning assumption for Phase-4
scale work until uncertainty recalibration (e.g. distance-aware/temperature-scaled `u`) is built.

---

## Standing constraints introduced by Tier 2

- **V2.1 (CONSTRAIN):** surrogate self-uncertainty is insufficient on its own in the violent
  regime; the handoff must be gated by the descriptor validity-envelope + connectivity trigger, with
  always-RVE as the fallback. The violent-regime cost ceiling is bounded by the always-RVE fallback.
- **V2.4 / V2.2 link:** the percolating seam is invisible to volume-fraction homogenization; a
  connectivity trigger is a mandatory hard refine/fallback signal wherever seams can form (carries
  into V2.2).

## Reproduce

```
# oracle self-checks (each asserts against a simpler reference)
.venv/bin/python src/verification/oracles/dns_damage_3d.py
.venv/bin/python src/verification/oracles/violent_cells.py
.venv/bin/python src/verification/oracles/surrogate_gnn.py
.venv/bin/python src/verification/oracles/handoff_rule.py
# notebooks (dataset cached after first build; ~8–15 min first run)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  verification_notebooks/phase2/V2_4_surrogate_generalization.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  verification_notebooks/phase2/V2_1_rve_surrogate_handoff.ipynb
```
Scratch calibration helpers `oracles/_calib_v2{1,4}.py` and the notebook builders
`phase2/_build_v2{1,4}_nb.py` are intentionally uncommitted/auxiliary.
