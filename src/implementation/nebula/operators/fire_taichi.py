"""
Taichi GPU kernel for the fire field update (ARCHITECTURE Part V: the Taichi+Warp GPU
kernel substrate; Decision #27). Per-element physics never runs in the host language --
it runs in compiled GPU kernels. This is the throughput half of the fire law-domain: the
7-point conduction flux with k(chi) char insulation + the Arrhenius reaction update + the
boundary heat-loss / O2 influx, as one explicit forward-Euler step matching
operators.fire (the NumPy reference) to float tolerance.

It is written as a Taichi kernel over ti.ndarray buffers precisely because Taichi
**ahead-of-time (AOT) compilation** is the bridge to the eventual native runtime (the
kernel validated here is emitted for C++/Rust later; the port is the orchestration layer
only). f64 throughout to keep the Arrhenius exponentials faithful to the NumPy path.

The stiff-loop discipline (V1.2) is unchanged: call this explicit step inside
operators.integrators.step_substep (rate-driven sub-stepping) or use the IMEX path -- the
kernel is the per-sub-step field update, not a license to take the corrupted big step.
"""
import numpy as np

try:
    import taichi as ti
    ti.init(arch=ti.cuda, default_fp=ti.f64, random_seed=0)
    _HAS_TAICHI = True
    _ARCH = "cuda"
except Exception:
    try:
        import taichi as ti
        ti.init(arch=ti.cpu, default_fp=ti.f64, random_seed=0)
        _HAS_TAICHI = True
        _ARCH = "cpu"
    except Exception:
        _HAS_TAICHI = False
        _ARCH = None


if _HAS_TAICHI:
    @ti.func
    def _kchi(char, m_s, k_wood, k_char):
        chi = char / (char + m_s + 1e-12)
        return k_wood * (1.0 - chi) + k_char * chi

    @ti.kernel
    def _fire_explicit(
        T: ti.types.ndarray(), m_s: ti.types.ndarray(), gas: ti.types.ndarray(),
        o2: ti.types.ndarray(), char: ti.types.ndarray(), q: ti.types.ndarray(),
        To: ti.types.ndarray(), m_so: ti.types.ndarray(), gaso: ti.types.ndarray(),
        o2o: ti.types.ndarray(), charo: ti.types.ndarray(), qo: ti.types.ndarray(),
        A_py: ti.f64, Ta_py: ti.f64, nu_g: ti.f64, nu_c: ti.f64, dH_py: ti.f64,
        A_cb: ti.f64, Ta_cb: ti.f64, s_o2: ti.f64, dH_cb: ti.f64,
        k_wood: ti.f64, k_char: ti.f64, h_loss: ti.f64, T_amb: ti.f64,
        o2_influx: ti.f64, o2_amb: ti.f64, lambda_q: ti.f64, C_V: ti.f64,
        inv_dx2: ti.f64, dt: ti.f64, N: ti.i32):
        for i, j, k in ti.ndrange(N, N, N):
            Tc = T[i, j, k]
            kc = _kchi(char[i, j, k], m_s[i, j, k], k_wood, k_char)
            # --- conduction: 7-point Fourier flux with face-averaged k(chi) ---
            dE = 0.0
            for a in ti.static(range(3)):
                for s in ti.static((-1, 1)):
                    ii = i + (s if a == 0 else 0)
                    jj = j + (s if a == 1 else 0)
                    kk = k + (s if a == 2 else 0)
                    if 0 <= ii < N and 0 <= jj < N and 0 <= kk < N:
                        knb = _kchi(char[ii, jj, kk], m_s[ii, jj, kk], k_wood, k_char)
                        kf = 0.5 * (kc + knb)
                        dE += kf * (T[ii, jj, kk] - Tc) * inv_dx2
            # --- boundary heat loss + O2 influx on every outer face this cell touches ---
            o2_src = 0.0
            for a in ti.static(range(3)):
                on_lo = (i == 0) if a == 0 else ((j == 0) if a == 1 else (k == 0))
                on_hi = (i == N - 1) if a == 0 else ((j == N - 1) if a == 1 else (k == N - 1))
                if on_lo:
                    dE += h_loss * inv_dx2 * (T_amb - Tc)
                    o2_src += o2_influx * (o2_amb - o2[i, j, k])
                if on_hi:
                    dE += h_loss * inv_dx2 * (T_amb - Tc)
                    o2_src += o2_influx * (o2_amb - o2[i, j, k])
            # --- Arrhenius reaction (rates at old state) ---
            r_py = A_py * ti.exp(-Ta_py / ti.max(Tc, 1.0)) * ti.max(m_s[i, j, k], 0.0)
            r_cb = (A_cb * ti.exp(-Ta_cb / ti.max(Tc, 1.0))
                    * ti.max(gas[i, j, k], 0.0) * ti.max(o2[i, j, k], 0.0))
            # --- forward-Euler commit with non-negativity clamps (the limiter) ---
            To[i, j, k] = Tc + ((dH_cb * r_cb - dH_py * r_py) / C_V + dE / C_V) * dt
            m_so[i, j, k] = ti.max(m_s[i, j, k] - r_py * dt, 0.0)
            gaso[i, j, k] = ti.max(gas[i, j, k] + (nu_g * r_py - r_cb) * dt, 0.0)
            charo[i, j, k] = ti.max(char[i, j, k] + nu_c * r_py * dt, 0.0)
            o2o[i, j, k] = ti.max(o2[i, j, k] + (-s_o2 * r_cb + o2_src) * dt, 0.0)
            qo[i, j, k] = ti.max(q[i, j, k] - lambda_q * q[i, j, k] * dt, 0.0)


def step_taichi(st, p, dt):
    """One explicit fire field-update step on the GPU (Taichi). Returns a new state dict.

    Matches operators.fire / core.buses one-step forward Euler to float tolerance; raises if
    Taichi is unavailable (the NumPy path in operators.fire is the fallback).
    """
    if not _HAS_TAICHI:
        raise RuntimeError("Taichi unavailable; use the NumPy path in operators.fire")
    N = st["T"].shape[0]

    def nd(a):
        arr = ti.ndarray(dtype=ti.f64, shape=(N, N, N))
        arr.from_numpy(np.ascontiguousarray(a, dtype=np.float64))
        return arr

    T, m_s, gas, o2, char, q = (nd(st[f]) for f in ("T", "m_s", "gas", "o2", "char", "q"))
    out = [ti.ndarray(dtype=ti.f64, shape=(N, N, N)) for _ in range(6)]
    To, m_so, gaso, o2o, charo, qo = out
    _fire_explicit(
        T, m_s, gas, o2, char, q, To, m_so, gaso, o2o, charo, qo,
        p.A_py, p.Ta_py, p.nu_g, p.nu_c, p.dH_py, p.A_cb, p.Ta_cb, p.s_o2, p.dH_cb,
        p.k_wood, p.k_char, p.h_loss, p.T_amb, p.o2_influx, p.o2_amb, p.lambda_q, p.C_V,
        1.0 / (p.dx * p.dx), float(dt), N)
    ti.sync()
    new = {f: arr.to_numpy() for f, arr in
           zip(("T", "m_s", "gas", "o2", "char", "q"), (To, m_so, gaso, o2o, charo, qo))}
    return new


if __name__ == "__main__":
    from ..core import buses
    from . import fire as fo
    print(f"Taichi available: {_HAS_TAICHI}  arch: {_ARCH}")
    if not _HAS_TAICHI:
        raise SystemExit

    dom = fo.fire_domain()
    p = dom.params
    N = 16
    # an active scene (hot core) so reaction + conduction both matter
    st = fo.make_state(N, T0=550.0, gas0=0.1, o2=0.2)
    c = slice(N // 2 - 2, N // 2 + 2)
    st["T"][c, c, c] = 850.0

    dt = 1e-4
    ref, *_ = buses.step(dom, st, dt, op_order=fo.ORACLE_OP_ORDER)   # NumPy reference
    gpu = step_taichi(st, p, dt)                                     # Taichi kernel

    print("Taichi vs NumPy one explicit step (rel diff per field):")
    worst = 0.0
    for f in ("T", "m_s", "gas", "o2", "char"):
        denom = np.abs(ref[f]).max() + 1e-30
        rd = float(np.abs(gpu[f] - ref[f]).max() / denom)
        worst = max(worst, rd)
        print(f"   {f:4}: max rel diff = {rd:.2e}")
    assert worst < 1e-9, f"Taichi kernel diverged from NumPy reference ({worst:.2e})"
    print(f"\nfire Taichi kernel matches the NumPy reference (worst {worst:.1e}).")
