"""
Determinism under GPU float non-associativity (ARCHITECTURE Decision #3; Part VII).
Verified by V0.5 (PASS): fixed reduction order is bit-reproducible, atomic order is not.

The conserved-bus REDUCE step (gather -> stage -> reduce -> commit) is a scatter-add of
many contributions into buses. On a GPU an atomic scatter-add accumulates in
scheduler-determined order, and since (a+b)+c != a+(b+c) in floating point the result
changes bit-for-bit run to run -- silently breaking the "program IS the asset" /
memoization promise (V0.5 measured 12 distinct bit-patterns from identical work). This
module isolates that reduction and offers the deterministic fixes:

  - fixed-order float : sum each bus's contributions in a CANONICAL order (sort by
    (key, value)); bit-identical across runs/configs (and, for pure summation with no
    FMA, across CPU<->GPU too -- measured, not assumed).
  - integer-exact     : quantize to int64 and accumulate; integer addition is exactly
    associative, so ANY order (even nondeterministic atomics) gives the identical
    integer -> bit-exact across runs, configs, AND devices. The bit-exact path for
    MEMOIZATION KEYS.

It also exposes the stable hashing (blake2b) used to derive every per-element RNG stream
from a stable lineage key rather than the global draw order -- the requirement that makes
per-element memoization sound (V1.8). NEVER use Python's salted builtin hash() for this.

Ported verbatim-in-behaviour from src/verification/oracles/determinism.py (frozen oracle).
"""
import hashlib
import math

import numpy as np

try:
    import warp as wp
    wp.init()
    _HAS_WARP = wp.get_cuda_device_count() > 0
except Exception:
    _HAS_WARP = False


# ---------------- stable hashing (per-element sub-seeds + memo keys) ----------------

def stable_hash(*parts) -> int:
    """Deterministic 63-bit hash of the parts (blake2b over a canonical repr).

    Unlike builtin hash(), this is identical across processes/runs -- the requirement for
    bit-reproducible per-branch sub-seeds (growth) and for write-back-state cache keys.
    """
    h = hashlib.blake2b(repr(parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2 ** 63)


def rng_from_key(*parts):
    """A numpy Generator seeded by the stable hash of the parts (env-independent stream)."""
    return np.random.default_rng(stable_hash(*parts))


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


def fixed_order_sum(arrays):
    """Sum a list of arrays in a CANONICAL (caller-fixed) order, left-to-right.

    The host-side analogue of the GPU fixed-order reduce: when several operators each
    stage a full-field contribution, summing them in a fixed order makes the committed
    state bit-reproducible (V1.1: 120 orderings bit-identical under fixed-order reduce).
    Callers pass `arrays` already in canonical order (e.g. operators sorted by name).
    """
    if not arrays:
        return None
    acc = np.array(arrays[0], dtype=np.float64, copy=True)
    for a in arrays[1:]:
        acc += a
    return acc


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
    # stable hashing is process-independent
    assert stable_hash("a", 1, (2, 3)) == stable_hash("a", 1, (2, 3))
    print("stable_hash deterministic: OK")

    K = 64; SCALE = 1e3
    keys, vals = make_problem(M=4_000_000, K=K, seed=0)
    vs, ss, se = canonical_sort(keys, vals)
    cpu_fix = cpu_fixed_order(vs, ss, se, K)
    cpu_ibus, cpu_ifloat = cpu_integer_exact(keys, vals, K, SCALE)

    if not _HAS_WARP:
        print("no GPU; CPU references only"); raise SystemExit

    a0 = gpu_atomic_float(keys, vals, K)
    perm = np.random.default_rng(1).permutation(len(keys))
    a2 = gpu_atomic_float(keys[perm], vals[perm], K)
    print(f"1) atomic float: run0==perm bitwise = {bitwise_equal(a0, a2)} "
          f"(rel diff {rel_diff(a0, a2):.2e}) -> hazard real if False")

    f0 = gpu_fixed_order(vs, ss, se, K)
    vs_p, ss_p, se_p = canonical_sort(keys[perm], vals[perm])
    f2 = gpu_fixed_order(vs_p, ss_p, se_p, K)
    print(f"2) fixed-order float: GPU run0==perm {bitwise_equal(f0, f2)}  "
          f"CPU==GPU {bitwise_equal(cpu_fix, f0)}")

    g_ibus, _ = gpu_integer_exact(keys, vals, K, SCALE)
    g_ibus_p, _ = gpu_integer_exact(keys[perm], vals[perm], K, SCALE)
    print(f"3) integer-exact: GPU run==perm {np.array_equal(g_ibus, g_ibus_p)}  "
          f"CPU==GPU {np.array_equal(cpu_ibus, g_ibus)}")
