"""
Buoyant incompressible flow — the gas-phase law-domain that makes a FLAME a flame
(ARCHITECTURE §III.3 "package the transfer operator"; Part IV guardrail #1).

Phase-0's fire was a per-voxel 0-D ODE: combustion burned in place and the volatile gas
never moved, so there was no plume, no flame standoff, no smoke — nothing that reads as
fire. This module adds the missing transport: the volatiles/heat/soot ride a buoyancy-
driven incompressible velocity field, so combustion happens ABOVE the fuel in a rising
flame and the products rise as smoke.

Discipline (the two guardrails + the one compute pattern):
  - BUOYANCY IS A POTENTIAL, NOT A RAW FORCE (guardrail #1). A Boussinesq body force
    f_z = beta*g*(T - T_ref) is added, then a PRESSURE PROJECTION removes the divergent
    part — the energy-stable Helmholtz–Hodge decomposition. No energy is injected; the
    flow stays divergence-free and stable. (Raw force fields inject energy and blow up.)
  - ADVECTION = CONSERVED-FLUX STAGING ON THE BUS (gather→stage→reduce→commit). Scalars
    move by a flux-form (finite-volume) upwind update on the MAC faces: a quantity leaving
    one cell enters its neighbour exactly, so ∫scalar is conserved to machine precision and
    the V0.3 conservation audit extends to transport for free. Walls carry no normal flux.
  - DETERMINISM (Decision #3 / V0.5). A staggered MAC grid + DCT Poisson solve uses only
    fixed-order CPU reductions (numpy/scipy.fft), so a re-run is bit-identical. A GPU port
    must keep the FFT/reduction order fixed (the standing V0.5 constraint).

MAC layout: pressure & scalars at cell centres (nx,ny,nz); u,v,w on the x/y/z faces
((nx+1,ny,nz) etc.). div(faces→centre) and grad(centre→faces) compose to exactly the
3-point Neumann Laplacian the DCT inverts, so projection removes divergence exactly and
there is no collocated-grid checkerboard.
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import dctn, idctn


@dataclass
class FlowParams:
    dx: float = 1.0               # uniform cell size
    g: float = 9.81              # gravity magnitude (acts on -z; buoyancy lifts hot fluid +z)
    beta: float = 1.0 / 300.0    # thermal expansion coeff (Boussinesq); buoyancy = beta*g*ΔT
    T_ref: float = 300.0         # ambient/reference temperature
    nu: float = 0.0              # kinematic viscosity (0 = inviscid stable-fluids)
    cfl: float = 0.8             # max advective Courant number (caps the scalar sub-step)


# --------------------------------------------------------------------------- grid helpers
def zero_velocity(shape):
    """Face-centred velocity arrays (u,v,w) for a centre grid of `shape`, all walls no-flow."""
    nx, ny, nz = shape
    return (np.zeros((nx + 1, ny, nz)), np.zeros((nx, ny + 1, nz)), np.zeros((nx, ny, nz + 1)))


def divergence(u, v, w, dx):
    """Cell-centred divergence of a MAC velocity (1/dx * sum of net face outflow)."""
    return ((u[1:, :, :] - u[:-1, :, :]) + (v[:, 1:, :] - v[:, :-1, :])
            + (w[:, :, 1:] - w[:, :, :-1])) / dx


def max_divergence(u, v, w, dx):
    return float(np.abs(divergence(u, v, w, dx)).max())


# --------------------------------------------------------------- DCT Neumann Poisson solve
def _neumann_eigs(n, dx):
    """Eigenvalues of the 1-D 3-point Neumann Laplacian under DCT-II."""
    k = np.arange(n)
    return -(2.0 - 2.0 * np.cos(np.pi * k / n)) / (dx * dx)


def project(u, v, w, dx):
    """Make (u,v,w) divergence-free: solve ∇²φ = div(u*), then u ← u* − ∇φ (interior faces).

    Returns (u, v, w, div_before, div_after). Wall (boundary) normal velocities are left at
    zero (no penetration); pressure uses homogeneous Neumann BCs (DCT-II), consistent with
    the closed/no-flux box.
    """
    rhs = divergence(u, v, w, dx)
    div_before = float(np.abs(rhs).max())
    nx, ny, nz = rhs.shape
    lx = _neumann_eigs(nx, dx)[:, None, None]
    ly = _neumann_eigs(ny, dx)[None, :, None]
    lz = _neumann_eigs(nz, dx)[None, None, :]
    denom = lx + ly + lz
    denom[0, 0, 0] = 1.0                                   # gauge: φ mean is arbitrary
    rhs_hat = dctn(rhs, type=2, norm="ortho")
    phi_hat = rhs_hat / denom
    phi_hat[0, 0, 0] = 0.0
    phi = idctn(phi_hat, type=2, norm="ortho")
    # correct interior faces by the pressure gradient (walls keep zero normal velocity)
    u[1:-1, :, :] -= (phi[1:, :, :] - phi[:-1, :, :]) / dx
    v[:, 1:-1, :] -= (phi[:, 1:, :] - phi[:, :-1, :]) / dx
    w[:, :, 1:-1] -= (phi[:, :, 1:] - phi[:, :, :-1]) / dx
    div_after = max_divergence(u, v, w, dx)
    return u, v, w, div_before, div_after


# ------------------------------------------------------------------- velocity interpolation
def _centre_velocity(u, v, w):
    """Velocity averaged to cell centres (for buoyancy sampling / diagnostics / backtrace)."""
    uc = 0.5 * (u[1:, :, :] + u[:-1, :, :])
    vc = 0.5 * (v[:, 1:, :] + v[:, :-1, :])
    wc = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
    return uc, vc, wc


def _sample_trilinear(field, pts):
    """Trilinear sample of a 3-D `field` at integer-indexed coords `pts` (..., 3), clamped."""
    s = np.asarray(field.shape) - 1
    p = np.clip(pts, 0.0, s.astype(float))
    i0 = np.floor(p).astype(int)
    i1 = np.minimum(i0 + 1, s)
    f = p - i0
    out = np.zeros(p.shape[:-1])
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                wgt = ((f[..., 0] if cx else 1 - f[..., 0])
                       * (f[..., 1] if cy else 1 - f[..., 1])
                       * (f[..., 2] if cz else 1 - f[..., 2]))
                ix = i1[..., 0] if cx else i0[..., 0]
                iy = i1[..., 1] if cy else i0[..., 1]
                iz = i1[..., 2] if cz else i0[..., 2]
                out += wgt * field[ix, iy, iz]
    return out


def advect_velocity(u, v, w, dt, dx):
    """Unconditionally-stable semi-Lagrangian self-advection of the MAC velocity.

    Each face component is traced back along the full velocity sampled at that face (its own
    component directly, the other two averaged from centres onto the face) and resampled.
    Boundary normal faces are reset to 0 afterward (no penetration).
    """
    uc, vc, wc = _centre_velocity(u, v, w)
    velx = np.stack([u, _pad_avg(vc, 0, u.shape[0]), _pad_avg(wc, 0, u.shape[0])], -1)
    vely = np.stack([_pad_avg(uc, 1, v.shape[1]), v, _pad_avg(wc, 1, v.shape[1])], -1)
    velz = np.stack([_pad_avg(uc, 2, w.shape[2]), _pad_avg(vc, 2, w.shape[2]), w], -1)
    un = _trace_component(u, velx, dt, dx)
    vn = _trace_component(v, vely, dt, dx)
    wn = _trace_component(w, velz, dt, dx)
    un[0, :, :] = un[-1, :, :] = 0.0
    vn[:, 0, :] = vn[:, -1, :] = 0.0
    wn[:, :, 0] = wn[:, :, -1] = 0.0
    return un, vn, wn


def _pad_avg(centre, axis, n):
    """Average a centre field to a face array of length n along `axis` (edge-replicated)."""
    lo = np.take(centre, 0, axis=axis)
    hi = np.take(centre, centre.shape[axis] - 1, axis=axis)
    padded = np.concatenate([np.expand_dims(lo, axis), centre, np.expand_dims(hi, axis)], axis=axis)
    a = np.take(padded, range(0, n), axis=axis)
    b = np.take(padded, range(1, n + 1), axis=axis)
    return 0.5 * (a + b)


def _trace_component(comp, vel_at_faces, dt, dx):
    """Semi-Lagrangian resample of `comp` at its own face positions, in comp's index frame."""
    sh = comp.shape
    gi, gj, gk = np.meshgrid(np.arange(sh[0]), np.arange(sh[1]), np.arange(sh[2]), indexing="ij")
    pos = np.stack([gi, gj, gk], -1).astype(float)
    dep = pos - (dt / dx) * vel_at_faces
    return _sample_trilinear(comp, dep)


# ----------------------------------------------------------- conservative scalar advection
def advect_scalar(s, u, v, w, dt, dx):
    """Flux-form first-order upwind advection of a centre scalar by the MAC velocity.

    Conservative by construction: the face flux leaving cell i enters cell i+1 unchanged, so
    Σ s changes only through boundary faces — and walls carry u=0, so a closed domain
    conserves Σ s to machine precision (the transport analogue of the conserved bus).
    """
    out = s.copy()
    # x-faces (interior): flux = u * upwind(s)
    fu = np.zeros_like(u)
    up = np.where(u[1:-1, :, :] > 0, s[:-1, :, :], s[1:, :, :])
    fu[1:-1, :, :] = u[1:-1, :, :] * up
    fv = np.zeros_like(v)
    up = np.where(v[:, 1:-1, :] > 0, s[:, :-1, :], s[:, 1:, :])
    fv[:, 1:-1, :] = v[:, 1:-1, :] * up
    fw = np.zeros_like(w)
    up = np.where(w[:, :, 1:-1] > 0, s[:, :, :-1], s[:, :, 1:])
    fw[:, :, 1:-1] = w[:, :, 1:-1] * up
    div_flux = ((fu[1:, :, :] - fu[:-1, :, :]) + (fv[:, 1:, :] - fv[:, :-1, :])
                + (fw[:, :, 1:] - fw[:, :, :-1])) / dx
    out -= dt * div_flux
    return out


def add_buoyancy(w, T, p, dt):
    """Boussinesq body force on the vertical (z) faces: f_z = beta*g*(T_face − T_ref)."""
    T_face = 0.5 * (T[:, :, 1:] + T[:, :, :-1])
    w[:, :, 1:-1] += dt * p.beta * p.g * (T_face - p.T_ref)
    return w


def max_speed(u, v, w):
    uc, vc, wc = _centre_velocity(u, v, w)
    return float(np.sqrt(uc * uc + vc * vc + wc * wc).max())


def stable_dt(u, v, w, p, dt_max):
    """Largest advective sub-step under the CFL cap (semi-Lagrangian velocity is uncond. stable,
    so the binding limit is the conservative upwind scalar update)."""
    smax = max_speed(u, v, w)
    if smax < 1e-12:
        return dt_max
    return min(dt_max, p.cfl * p.dx / smax)


def step(u, v, w, scalars, p, dt, sub=True):
    """Advance the buoyant flow + advected scalars by `dt`.

    scalars: dict of centre fields to transport (e.g. {"T":..., "gas":..., "soot":...}); the
    field named "T" drives buoyancy. Returns (u,v,w, scalars, info). Internally sub-cycles to
    respect the scalar CFL so transport stays conservative & stable.
    """
    done = 0.0
    info = {"substeps": 0, "div_before": 0.0, "div_after": 0.0}
    while done < dt - 1e-15:
        h = stable_dt(u, v, w, p, dt - done) if sub else (dt - done)
        u, v, w = advect_velocity(u, v, w, h, p.dx)
        if "T" in scalars:
            w = add_buoyancy(w, scalars["T"], p, h)
        u, v, w, d0, d1 = project(u, v, w, p.dx)
        for name in scalars:
            scalars[name] = advect_scalar(scalars[name], u, v, w, h, p.dx)
        info["substeps"] += 1
        info["div_before"] = max(info["div_before"], d0)
        info["div_after"] = max(info["div_after"], d1)
        done += h
    return u, v, w, scalars, info


# ---------------------------------------------------------------------- plume diagnostics
def centerline_profiles(scalars, u, v, w, p, axis_xy=None):
    """Per-height centreline ΔT, vertical velocity W, and buoyancy flux F(z)=Σ_A w·ΔT dA.

    Returns dict z, dT (centre-column T−T_ref), W (centre-column wc), F (per-height integral).
    """
    T = scalars["T"]
    nx, ny, nz = T.shape
    cx, cy = (nx // 2, ny // 2) if axis_xy is None else axis_xy
    _, _, wc = _centre_velocity(u, v, w)
    dT = T[cx, cy, :] - p.T_ref
    W = wc[cx, cy, :]
    dArea = p.dx * p.dx
    F = (wc * (T - p.T_ref)).sum(axis=(0, 1)) * dArea   # buoyancy flux per height
    z = (np.arange(nz) + 0.5) * p.dx
    return {"z": z, "dT": dT, "W": W, "F": F}


if __name__ == "__main__":
    np.seterr(all="ignore")
    p = FlowParams(dx=1.0, beta=1.0 / 300.0, T_ref=300.0)
    shape = (24, 24, 24)

    # 1) projection removes divergence to machine precision (random divergent field).
    rng = np.random.default_rng(0)
    u, v, w = zero_velocity(shape)
    u += rng.standard_normal(u.shape); v += rng.standard_normal(v.shape); w += rng.standard_normal(w.shape)
    u[0] = u[-1] = 0; v[:, 0] = v[:, -1] = 0; w[:, :, 0] = w[:, :, -1] = 0
    u, v, w, d0, d1 = project(u, v, w, p.dx)
    print(f"1) projection: max|div| {d0:.3e} -> {d1:.3e}")
    assert d1 < 1e-9

    # 2) conservative advection conserves Σ scalar (closed box, prescribed solenoidal flow).
    s = np.zeros(shape); s[8:16, 8:16, 8:16] = 1.0
    u, v, w = zero_velocity(shape)
    u += 0.5; v += 0.3                          # uniform flow (divergence-free, walls reset)
    u[0] = u[-1] = 0; v[:, 0] = v[:, -1] = 0; w[:, :, 0] = w[:, :, -1] = 0
    u, v, w, *_ = project(u, v, w, p.dx)
    tot0 = s.sum()
    for _ in range(40):
        s = advect_scalar(s, u, v, w, 0.4, p.dx)
    print(f"2) scalar conservation over 40 steps: Σs {tot0:.6f} -> {s.sum():.6f} "
          f"(rel {abs(s.sum()-tot0)/tot0:.2e})")
    assert abs(s.sum() - tot0) / tot0 < 1e-10

    # 3) a hot bottom patch RISES (buoyancy sign) and stays divergence-free.
    T = np.full(shape, 300.0)
    sc = {"T": T}
    u, v, w = zero_velocity(shape)
    def hot(sc):
        sc["T"][10:14, 10:14, 0:2] = 700.0     # heated plate at the base centre
    com0 = None
    for n in range(60):
        hot(sc)
        u, v, w, sc, info = step(u, v, w, sc, p, 0.5)
        dT = np.clip(sc["T"] - 300.0, 0, None)
        z = (np.arange(shape[2]) + 0.5)
        com = float((dT.sum(axis=(0, 1)) * z).sum() / (dT.sum() + 1e-30))
        if n == 5:
            com0 = com
    print(f"3) hot-patch plume: heat center-of-mass z {com0:.2f} -> {com:.2f} "
          f"(rises), final max|div| {info['div_after']:.2e}")
    assert com > com0 + 1.0 and info["div_after"] < 1e-8

    # 4) determinism: re-run bit-identical (fixed-order CPU reductions).
    def run():
        uu, vv, ww = zero_velocity(shape); TT = np.full(shape, 300.0); s2 = {"T": TT}
        for _ in range(8):
            s2["T"][10:14, 10:14, 0:2] = 700.0
            uu, vv, ww, s2, _ = step(uu, vv, ww, s2, p, 0.5)
        return s2["T"]
    a, b = run(), run()
    print(f"4) determinism: re-run bitwise-identical = {np.array_equal(a.view(np.uint64), b.view(np.uint64))}")
    assert np.array_equal(a.view(np.uint64), b.view(np.uint64))
    print("\nflow solver self-checks passed.")
