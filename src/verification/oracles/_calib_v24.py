"""SCRATCH (uncommitted) calibration for V2.4 — measures the three metrics to freeze thresholds.

Run: .venv/bin/python src/verification/oracles/_calib_v24.py
Prints in-family generalization error, OOD detection / false-positive, and the PINN-vs-baseline
data-efficiency ratio. Thresholds for the notebook are then set BELOW the achieved margins.
"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np

import violent_cells as vc
from surrogate_gnn import (Ensemble, EnvelopeDetector, TrainCfg, build_dataset, fallback_flags,
                           DATA_PARAMS)

N = 12
CACHE = str(pathlib.Path(__file__).parents[3] / "verification_notebooks" / "phase2" / "cache")
rng = np.random.default_rng(2024)

train = vc.family_battery(N, rng, 45)
test = vc.family_battery(N, rng, 20)
ood = vc.ood_battery(N, rng, 6)              # 18 cells: 6 seam + 6 extreme-contrast + 6 blob

t = time.time()
d_tr = build_dataset(train, DATA_PARAMS, cache=f"{CACHE}/v24_train.npz")
d_te = build_dataset(test, DATA_PARAMS, cache=f"{CACHE}/v24_test.npz")
d_oo = build_dataset(ood, DATA_PARAMS, cache=f"{CACHE}/v24_ood.npz")
print(f"dataset ready ({time.time()-t:.0f}s)  y range train={d_tr['y'].min():.3f}..{d_tr['y'].max():.3f}")

# 1) in-family generalization (PINN)
ens = Ensemble.train(train, d_tr["y"], TrainCfg(epochs=400, physics=True), M=5, base_seed=0)
p_te = ens.predict(test)
rel = np.abs(p_te["mean"] - d_te["y"]) / np.abs(d_te["y"])
print(f"[1] in-family median rel err = {np.median(rel):.4f}  (p90={np.quantile(rel,0.9):.4f})")

# 2) OOD detection via fallback trigger
env = EnvelopeDetector.fit(train)
thr = float(np.quantile(env.score(test), 0.95))
f_oo, _ = fallback_flags(ood, env, thr)
f_te, _ = fallback_flags(test, env, thr)
print(f"[2] OOD detected = {f_oo.mean()*100:.1f}% ; in-family false-positive = {f_te.mean()*100:.1f}%")

# 3) data efficiency: PINN vs pure-data, test error vs #train samples
sizes = [8, 16, 25, 35, 45]
def curve(physics):
    errs = []
    for k in sizes:
        e = Ensemble.train(train[:k], d_tr["y"][:k], TrainCfg(epochs=400, physics=physics),
                           M=5, base_seed=10)
        rk = np.abs(e.predict(test)["mean"] - d_te["y"]) / np.abs(d_te["y"])
        errs.append(float(np.median(rk)))
    return np.array(errs)
e_pinn = curve(True)
e_base = curve(False)
print("[3] sizes      :", sizes)
print("    PINN  med  :", np.round(e_pinn, 4))
print("    base  med  :", np.round(e_base, 4))
target = max(e_base.min(), e_pinn.min()) * 1.2     # a target accuracy both can plausibly hit
def samples_to(errs):
    hit = [s for s, e in zip(sizes, errs) if e <= target]
    return hit[0] if hit else np.inf
sp, sb = samples_to(e_pinn), samples_to(e_base)
print(f"    target err = {target:.4f}; samples-to-target PINN={sp} base={sb}; ratio={sp/sb if sb else float('nan'):.2f}")
