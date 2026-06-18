"""
Determinism under GPU float non-associativity (ARCHITECTURE Decision #3; Part VII).

The conserved-bus REDUCE step (V0.3's gather -> stage -> reduce -> commit) is a
scatter-add of many contributions into buses. On a GPU, an atomic scatter-add
accumulates in scheduler-determined order, and since (a+b)+c != a+(b+c) in floating
point, the result changes bit-for-bit run to run -- silently breaking the "program IS
the asset" / memoization promise. This module isolates that reduction and offers two
deterministic fixes whose bit-reproducibility V0.5 verifies:

  - fixed-order float : sum each bus's contributions in a CANONICAL order (sort by
    (key, value)); bit-identical across runs/configs (and, for pure summation with no
    FMA, across CPU<->GPU too -- measured, not assumed).
  - integer-exact     : quantize to int64 and accumulate; integer addition is exactly
    associative, so ANY order (even nondeterministic atomics) gives the identical
    integer -> bit-exact across runs, configs, AND devices. The bit-exact path for
    memoization keys.

CPU (numpy) is the genuine second device; GPU launch configs are varied via input
permutations (which change atomic accumulation order). Warp is the substrate.
"""
import math

import numpy as np

try:
    import warp as wp
    wp.init()
    _HAS_WARP = wp.get_cuda_device_count() > 0
except Exception:
    _HAS_WARP = False


# ---------------- problem + CPU references (the oracle) ----------------

def make_problem(M=4_000_000, K=64, seed=0):
    """M contributions -> K buses, with a WIDE dynamic range so summation order matters."""
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, K, M).astype(np.int32)
    big = rng.random(M) < 0.5
    vals = (rng.random(M) * np.where(big, 1e6, 1e-3)).astype(np.float64)
    return keys, vals


def canonical_sort(keys, vals):
    """Canonical (key, value) order -> permutation-invariant fixed reduction order."""
    order = np.lexsort((vals, keys))           # sort by key, then value
    ks = keys[order]; vs = vals[order]
    seg_start = np.searchsorted(ks, np.arange(int(keys.max()) + 1), side="left").astype(np.int32)
    seg_end = np.searchsorted(ks, np.arange(int(keys.max()) + 1), side="right").astype(np.int32)
    return vs, seg_start, seg_end


def cpu_fixed_order(vs, seg_start, seg_end, K):
    """Per-bus SEQUENTIAL float64 sum in canonical order (matches the GPU kernel exactly)."""
    lengths = seg_end - seg_start
    maxlen = int(lengths.max())
    mat = np.zeros((K, maxlen))
    for k in range(K):
        L = lengths[k]
        mat[k, :L] = vs[seg_start[k]:seg_end[k]]
    acc = np.zeros(K)
    for t in range(maxlen):                    # sequential left-to-right accumulation
        acc += mat[:, t]
    return acc


def cpu_integer_exact(keys, vals, K, scale):
    """Quantize to int64 and exact-accumulate (order-independent). Returns (int_bus, float_bus)."""
    q = np.rint(vals * scale).astype(np.int64)
    ibus = np.zeros(K, dtype=np.int64)
    np.add.at(ibus, keys, q)                   # exact integer accumulation
    return ibus, ibus / scale


def high_precision(vs, seg_start, seg_end, K):
    """Exact-as-possible per-bus reference via math.fsum over canonical segments."""
    out = np.zeros(K)
    for k in range(K):
        out[k] = math.fsum(vs[seg_start[k]:seg_end[k]])
    return out


# ---------------- divergence helpers ----------------

def bitwise_equal(a, b):
    """True iff a and b are identical IEEE-754 bit patterns."""
    return np.array_equal(np.asarray(a, np.float64).view(np.uint64),
                          np.asarray(b, np.float64).view(np.uint64))


def rel_diff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-300)))


# ---------------- GPU kernels (Warp) ----------------

if _HAS_WARP:
    @wp.kernel
    def _atomic_f64(keys: wp.array(dtype=wp.int32), vals: wp.array(dtype=wp.float64),
                    bus: wp.array(dtype=wp.float64)):
        i = wp.tid()
        wp.atomic_add(bus, keys[i], vals[i])

    @wp.kernel
    def _fixed_order_f64(seg_start: wp.array(dtype=wp.int32), seg_end: wp.array(dtype=wp.int32),
                         vs: wp.array(dtype=wp.float64), bus: wp.array(dtype=wp.float64)):
        k = wp.tid()
        acc = wp.float64(0.0)
        j = seg_start[k]
        while j < seg_end[k]:                  # sequential, canonical order, config-independent
            acc = acc + vs[j]
            j += 1
        bus[k] = acc

    @wp.kernel
    def _atomic_i64(keys: wp.array(dtype=wp.int32), q: wp.array(dtype=wp.int64),
                    bus: wp.array(dtype=wp.int64)):
        i = wp.tid()
        wp.atomic_add(bus, keys[i], q[i])


def gpu_atomic_float(keys, vals, K):
    kd = wp.array(keys, dtype=wp.int32, device="cuda")
    vd = wp.array(vals, dtype=wp.float64, device="cuda")
    bus = wp.zeros(K, dtype=wp.float64, device="cuda")
    wp.launch(_atomic_f64, dim=len(keys), inputs=[kd, vd, bus], device="cuda")
    wp.synchronize()
    return bus.numpy()


def gpu_fixed_order(vs, seg_start, seg_end, K):
    vd = wp.array(vs, dtype=wp.float64, device="cuda")
    sd = wp.array(seg_start, dtype=wp.int32, device="cuda")
    ed = wp.array(seg_end, dtype=wp.int32, device="cuda")
    bus = wp.zeros(K, dtype=wp.float64, device="cuda")
    wp.launch(_fixed_order_f64, dim=K, inputs=[sd, ed, vd, bus], device="cuda")
    wp.synchronize()
    return bus.numpy()


def gpu_integer_exact(keys, vals, K, scale):
    q = np.rint(vals * scale).astype(np.int64)
    kd = wp.array(keys, dtype=wp.int32, device="cuda")
    qd = wp.array(q, dtype=wp.int64, device="cuda")
    bus = wp.zeros(K, dtype=wp.int64, device="cuda")
    wp.launch(_atomic_i64, dim=len(keys), inputs=[kd, qd, bus], device="cuda")
    wp.synchronize()
    ibus = bus.numpy()
    return ibus, ibus / scale


if __name__ == "__main__":
    print(f"Warp GPU available: {_HAS_WARP}")
    K = 64; SCALE = 1e3
    keys, vals = make_problem(M=4_000_000, K=K, seed=0)
    vs, ss, se = canonical_sort(keys, vals)

    # CPU references
    cpu_fix = cpu_fixed_order(vs, ss, se, K)
    cpu_ibus, cpu_ifloat = cpu_integer_exact(keys, vals, K, SCALE)

    if not _HAS_WARP:
        print("no GPU; CPU references only"); raise SystemExit

    # 1) ATOMIC: run-to-run + permutation divergence (the hazard)
    a0 = gpu_atomic_float(keys, vals, K)
    a1 = gpu_atomic_float(keys, vals, K)                       # same input, repeated
    perm = np.random.default_rng(1).permutation(len(keys))
    a2 = gpu_atomic_float(keys[perm], vals[perm], K)           # permuted "config"
    print("\n1) atomic float64 scatter-add:")
    print(f"   run0==run1 bitwise: {bitwise_equal(a0, a1)}   (rel diff {rel_diff(a0, a1):.2e})")
    print(f"   run0==perm bitwise: {bitwise_equal(a0, a2)}   (rel diff {rel_diff(a0, a2):.2e})")
    print("   => hazard real if any False / rel diff > 0")

    # 2) FIXED-ORDER float64: reproducible across runs/perms; vs CPU
    f0 = gpu_fixed_order(vs, ss, se, K)
    f1 = gpu_fixed_order(vs, ss, se, K)
    vs_p, ss_p, se_p = canonical_sort(keys[perm], vals[perm])  # canonical sort is perm-invariant
    f2 = gpu_fixed_order(vs_p, ss_p, se_p, K)
    print("2) fixed-order float64:")
    print(f"   GPU run0==run1: {bitwise_equal(f0, f1)}   GPU run0==perm: {bitwise_equal(f0, f2)}")
    print(f"   CPU==GPU bitwise: {bitwise_equal(cpu_fix, f0)}   (rel diff {rel_diff(cpu_fix, f0):.2e})")

    # 3) INTEGER-EXACT: exact across runs/perms AND CPU<->GPU
    g_ibus, _ = gpu_integer_exact(keys, vals, K, SCALE)
    g_ibus_p, _ = gpu_integer_exact(keys[perm], vals[perm], K, SCALE)
    print("3) integer-exact (int64):")
    print(f"   GPU run==perm exact: {np.array_equal(g_ibus, g_ibus_p)}   "
          f"CPU==GPU exact: {np.array_equal(cpu_ibus, g_ibus)}")
    hp = high_precision(vs, ss, se, K)
    print(f"   quantization rel err vs fsum (scale={SCALE:g}): {rel_diff(cpu_ifloat, hp):.2e}")
