"""SCRATCH (uncommitted) calibration for V2.1 — freezes the numbers for the notebook (A+B+C fix).

Mirrors what V2_1_rve_surrogate_handoff.ipynb computes:
  (C) RPF surrogate (randomized-prior ensemble) -> bare `u` is RANK-reliable (binned-rho gate);
  (B) rank-preserving TEMPERATURE recalibration on a HELD-OUT split -> point coverage;
  (A)+(B) FeatureDensity-distance-keyed MONDRIAN CONFORMAL -> distribution-free coverage guarantee;
  the u-only handoff frontier + necessity, against the FROZEN budget.
All DNS is cached (v24_train / v21_battery / v21_calib), so this runs in ~2 min.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np

import violent_cells as vc
from surrogate_gnn import Ensemble, TrainCfg, build_dataset, DATA_PARAMS
import handoff_rule as hr

N = 12
CACHE = str(pathlib.Path(__file__).parents[3] / "verification_notebooks" / "phase2" / "cache")
RHO_MIN, OC_MAX, ERR_BOUND, RVE_BUDGET = 0.80, 0.12, 0.10, 0.30      # FROZEN

# RPF surrogate, trained on the exact V2.4 training set
rng = np.random.default_rng(2024)
train = vc.family_battery(N, rng, 45)
d_tr = build_dataset(train, DATA_PARAMS, cache=f"{CACHE}/v24_train.npz")
ens = Ensemble.train(train, d_tr["y"], TrainCfg(epochs=400, physics=True, beta=1.0), M=5, base_seed=0)

# scored battery (seed 777, reproduces the original -> cache valid) + held-out calibration split
battery = vc.violent_battery(N, np.random.default_rng(777))
d_b = build_dataset(battery, DATA_PARAMS, cache=f"{CACHE}/v21_battery.npz")
t = time.time()
calib_samples = vc.violent_battery(N, np.random.default_rng(20259), n_easy=30, n_hard=10)
d_c = build_dataset(calib_samples, DATA_PARAMS, cache=f"{CACHE}/v21_calib.npz")
print(f"battery {len(battery)} cells; calib {len(calib_samples)} cells ({time.time()-t:.0f}s)")

pred_b, pred_c = ens.predict(battery), ens.predict(calib_samples)
y_b, y_c = d_b["y"], d_c["y"]
rel_b = np.abs(pred_b["mean"] - y_b) / np.abs(y_b)
ae_b, ae_c = np.abs(pred_b["mean"] - y_b), np.abs(pred_c["mean"] - y_c)
u = pred_b["u"]
print(f"rel_err: median={np.median(rel_b):.3f} p95={np.quantile(rel_b,0.95):.3f} max={rel_b.max():.3f}")

# (B-gate) bare RPF u calibration — the frozen criteria
RHO = hr.binned_rank_correlation(u, rel_b)
OC = hr.overconfidence(rel_b, u, k=1.0)
c1, c2 = hr.coverage(rel_b, u)[1.0], hr.coverage(rel_b, u)[2.0]
CALIB_PASS = (RHO > RHO_MIN) and (abs(OC) < OC_MAX)
print(f"\n[calib] binned rho={RHO:.3f} (>{RHO_MIN}); oc@1s={OC:+.3f} (|.|<{OC_MAX}); "
      f"cov@1s={c1[0]:.2f}/{c1[1]:.2f} cov@2s={c2[0]:.2f}/{c2[1]:.2f}  -> {'PASS' if CALIB_PASS else 'SHORT'}")

# (B) rank-preserving temperature point-coverage + (A+B) distance-keyed conformal guarantee
temp = hr.Calibrator.fit(pred_c, ae_c, k=2.0)
c2t = hr.coverage(rel_b, temp.u(pred_b))[2.0]
hw = hr.mondrian_conformal(pred_c["dist"], ae_c, pred_b["dist"], alpha=0.1, n_bins=3)
print(f"[recal] temperature s={temp.s:.2f}: cov@2s {c2[0]:.2f}->{c2t[0]:.2f} (nominal {c2t[1]:.2f}); "
      f"mondrian-conformal 90% coverage={np.mean(ae_b<=hw):.2f}")

# (rule) u-only handoff + necessity, frozen budget
fr = hr.frontier(u, rel_b)
op = hr.operating_point(fr, err_bound=ERR_BOUND, rve_budget=RVE_BUDGET)
RULE_PASS = bool(op["exists"] and op["within_budget"])
NEC_PASS = bool(fr.tail_err[-1] > ERR_BOUND and fr.rve_frac[0] == 1.0 and 0.0 < op["rve_frac"] < 1.0)
print(f"\n[rule] u-only: rve_frac={op['rve_frac']:.3f} (<{RVE_BUDGET}) tail={op['tail_err']:.3f} (<{ERR_BOUND}) "
      f"-> {'PASS' if RULE_PASS else 'FAIL'}")
print(f"[necessity] always-trust tail={fr.tail_err[-1]:.3f} (>{ERR_BOUND}); always-RVE frac={fr.rve_frac[0]:.2f} "
      f"-> {'PASS' if NEC_PASS else 'FAIL'}")
print(f"\nV2.1 -> {'PASS' if (CALIB_PASS and RULE_PASS and NEC_PASS) else 'CONSTRAIN'}")
