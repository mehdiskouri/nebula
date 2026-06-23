"""
Gaussian splat rasterizer — the dual-cloud render path (ARCHITECTURE §III.1; Decision #6; V3.9).

The architecture renders the dense appearance cloud as **Gaussian splats** that ride the coarse
physics cloud (V1.9). This is a from-scratch, headless **EWA splatting** forward rasterizer in
torch (we have a 4090; no splat rasterizer was installed): anisotropic 3-D Gaussians → projected
2-D footprints → depth-sorted front-to-back alpha compositing → HDR image → tonemap + bloom (the
fire glow). It carries NO physics — it only consumes positions/covariances/colors, so it cannot
perturb the simulation; the mesh stays an export artifact (Decision #2).

Determinism (V0.5): a STABLE depth sort + fixed-order segmented compositing, so a re-render is
bit-reproducible on the same device.

Compositing is data-parallel (not a per-Gaussian python loop): every (pixel, gaussian) overlap is
a pair; pairs are sorted by (pixel, depth); a segmented exclusive cumprod of (1−α) gives the
front-to-back transmittance T; weight = α·T; scatter-add weight·color → image. Correct "over"
alpha compositing, fully vectorized.
"""
from dataclasses import dataclass

import numpy as np

try:
    import torch
    _DEV = "cuda" if torch.cuda.is_available() else "cpu"
    # V0.5 discipline for the render: force deterministic GPU reductions (index_add etc.) so a
    # re-render is bit-reproducible — the same float-reduction-order hazard the conserved bus has.
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:                                   # pragma: no cover
    torch = None
    _DEV = "cpu"


@dataclass
class Camera:
    eye: tuple
    target: tuple
    up: tuple = (0.0, 0.0, 1.0)
    fov_deg: float = 45.0
    W: int = 800
    H: int = 600

    def view(self):
        eye = np.asarray(self.eye, float); tgt = np.asarray(self.target, float)
        up = np.asarray(self.up, float)
        f = tgt - eye; f /= np.linalg.norm(f)
        s = np.cross(f, up); s /= np.linalg.norm(s)
        u = np.cross(s, f)
        R = np.stack([s, u, -f], 0)                 # world→camera rotation (rows)
        return R, eye

    def focal(self):
        return 0.5 * self.H / np.tan(0.5 * np.deg2rad(self.fov_deg))


def _t(a, dtype=None):
    return torch.as_tensor(np.asarray(a), dtype=dtype or torch.float32, device=_DEV)


def cov_from_scale_rot(scales, quats):
    """3×3 covariance Σ = R diag(s²) Rᵀ from per-splat scales (N,3) and unit quaternions (N,4 wxyz)."""
    s = _t(scales); q = _t(quats)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=1).reshape(-1, 3, 3)
    S = torch.zeros(len(s), 3, 3, device=_DEV)
    S[:, 0, 0] = s[:, 0] ** 2; S[:, 1, 1] = s[:, 1] ** 2; S[:, 2, 2] = s[:, 2] ** 2
    return R @ S @ R.transpose(1, 2)


def render(means, cov3d, colors, opacity, cam: Camera, bg=(0.03, 0.035, 0.05),
           max_radius_px=28, alpha_min=1.0 / 255):
    """Rasterize 3-D Gaussians to an HDR image (H,W,3) + alpha (H,W). Inputs are (N,3),(N,3,3),(N,3),(N,).

    `max_radius_px` caps each splat's screen footprint (bounds the pixel-gaussian pair count and so
    GPU memory). The accumulation is chunked + deterministic (index_add under
    use_deterministic_algorithms), staying bit-reproducible without materializing all pairs at once.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    means = _t(means); cov3d = _t(cov3d); colors = _t(colors); opacity = _t(opacity)
    Rv, eye = cam.view()
    Rv_t = _t(Rv); eye_t = _t(eye)
    foc = cam.focal(); cx, cy = cam.W / 2.0, cam.H / 2.0

    # to camera space
    pc = (means - eye_t) @ Rv_t.T                   # (N,3); camera looks down −z
    z = -pc[:, 2]
    front = z > 1e-4
    # projected screen coords (pinhole)
    sx = foc * pc[:, 0] / z.clamp_min(1e-6) + cx
    sy = -foc * pc[:, 1] / z.clamp_min(1e-6) + cy
    # EWA: project 3-D cov to 2-D via the affine Jacobian J of the projection, in camera frame
    J = torch.zeros(len(means), 2, 3, device=_DEV)
    J[:, 0, 0] = foc / z.clamp_min(1e-6)
    J[:, 0, 2] = foc * pc[:, 0] / z.clamp_min(1e-6) ** 2
    J[:, 1, 1] = -foc / z.clamp_min(1e-6)
    J[:, 1, 2] = -(-foc) * pc[:, 1] / z.clamp_min(1e-6) ** 2
    W = Rv_t.unsqueeze(0)                            # camera rotation
    JW = J @ W
    cov2d = JW @ cov3d @ JW.transpose(1, 2)         # (N,2,2)
    cov2d[:, 0, 0] += 0.3; cov2d[:, 1, 1] += 0.3    # low-pass dilation (anti-alias)
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    ok = front & (det > 1e-9) & (sx > -max_radius_px) & (sx < cam.W + max_radius_px) \
        & (sy > -max_radius_px) & (sy < cam.H + max_radius_px)
    idx = torch.nonzero(ok, as_tuple=False).squeeze(1)
    if len(idx) == 0:
        img = torch.tensor(bg, device=_DEV).expand(cam.H, cam.W, 3).clone()
        return img.cpu().numpy(), np.zeros((cam.H, cam.W), np.float32)
    sx, sy, z = sx[idx], sy[idx], z[idx]
    cov2d, colors, opacity = cov2d[idx], colors[idx], opacity[idx]
    inv = torch.linalg.inv(cov2d)                   # (M,2,2)
    # footprint radius (3σ) from the larger eigenvalue
    tr = cov2d[:, 0, 0] + cov2d[:, 1, 1]
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    lam = 0.5 * tr + torch.sqrt((0.5 * tr) ** 2 - det).clamp_min(0)
    rad = torch.clamp(3.0 * torch.sqrt(lam.clamp_min(1e-6)), 1.0, max_radius_px).ceil().long()

    # enumerate (pixel, gaussian) pairs over each footprint bbox
    pix_id, g_id, du, dv = [], [], [], []
    for r in torch.unique(rad):
        sel = torch.nonzero(rad == r, as_tuple=False).squeeze(1)
        rr = int(r.item())
        oy, ox = torch.meshgrid(torch.arange(-rr, rr + 1, device=_DEV),
                                torch.arange(-rr, rr + 1, device=_DEV), indexing="ij")
        ox = ox.reshape(-1); oy = oy.reshape(-1)
        px = (sx[sel].round().long()[:, None] + ox[None, :])      # (k, P)
        py = (sy[sel].round().long()[:, None] + oy[None, :])
        inb = (px >= 0) & (px < cam.W) & (py >= 0) & (py < cam.H)
        gg = sel[:, None].expand_as(px)
        dxp = px.float() - sx[sel][:, None]; dyp = py.float() - sy[sel][:, None]
        pix_id.append((py * cam.W + px)[inb]); g_id.append(gg[inb])
        du.append(dxp[inb]); dv.append(dyp[inb])
    pix_id = torch.cat(pix_id); g_id = torch.cat(g_id)
    du = torch.cat(du); dv = torch.cat(dv)

    # per-pair alpha = opacity * exp(-1/2 dᵀ Σ⁻¹ d)
    a = inv[g_id, 0, 0] * du * du + 2 * inv[g_id, 0, 1] * du * dv + inv[g_id, 1, 1] * dv * dv
    al = (opacity[g_id] * torch.exp(-0.5 * a)).clamp(0, 0.999)
    keep = al > alpha_min
    pix_id, g_id, al = pix_id[keep], g_id[keep], al[keep]
    zc = z[g_id]
    # STABLE lexicographic sort by (pixel, depth): front-to-back within each pixel. A float key
    # pix_id*K+depth would lose depth precision in the mantissa, so sort in two stable passes.
    o1 = torch.argsort(zc, stable=True)                 # secondary: depth ascending (front first)
    o2 = torch.argsort(pix_id[o1], stable=True)         # primary: pixel (stable keeps depth order)
    order = o1[o2]
    pix_id, g_id, al, zc = pix_id[order], g_id[order], al[order], zc[order]
    # segmented exclusive cumprod of (1−α) → transmittance T (front-to-back). The cumsum is in
    # FLOAT64: the within-segment value is a small difference of large global prefix sums, so float32
    # would lose it to catastrophic cancellation.
    log1m = torch.log((1 - al).clamp_min(1e-8)).double()
    cum = torch.cumsum(log1m, 0)
    seg_start = torch.cat([torch.tensor([0], device=_DEV),
                           torch.nonzero(pix_id[1:] != pix_id[:-1], as_tuple=False).squeeze(1) + 1])
    base_vals = cum[seg_start] - log1m[seg_start]              # exclusive offset at each segment start
    seg_id = torch.zeros(len(pix_id), dtype=torch.long, device=_DEV)
    seg_id[seg_start] = 1; seg_id = torch.cumsum(seg_id, 0) - 1
    T = torch.exp(cum - log1m - base_vals[seg_id]).float()     # exclusive within segment
    weight = al * T

    # accumulate (precise, no cancellation) via index_add_; deterministic under
    # use_deterministic_algorithms (PyTorch sorts the indices), the V0.5 fixed-order reduction.
    img = torch.zeros(cam.H * cam.W, 3, device=_DEV)
    acc_a = torch.zeros(cam.H * cam.W, device=_DEV)
    CH = 20_000_000                                            # chunk to bound peak memory
    for s0 in range(0, len(pix_id), CH):
        sl = slice(s0, s0 + CH)
        img.index_add_(0, pix_id[sl], weight[sl, None] * colors[g_id[sl]])
        acc_a.index_add_(0, pix_id[sl], weight[sl])
    bg_t = torch.tensor(bg, device=_DEV)
    img = img + (1 - acc_a)[:, None] * bg_t                    # composite background
    return img.reshape(cam.H, cam.W, 3).cpu().numpy(), acc_a.reshape(cam.H, cam.W).cpu().numpy()


def tonemap(hdr, exposure=1.0, bloom=0.0):
    """ACES-ish tonemap of an HDR image to 0..1 sRGB, with optional bloom on bright (fire) pixels."""
    x = np.asarray(hdr, float) * exposure
    if bloom > 0:
        from scipy.ndimage import gaussian_filter
        lum = x.sum(-1, keepdims=True)
        bright = np.clip(x * (lum > 1.0), 0, None)
        x = x + bloom * np.stack([gaussian_filter(bright[..., c], 6) for c in range(3)], -1)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return np.clip(x ** (1 / 2.2), 0, 1)


if __name__ == "__main__":
    assert torch is not None, "torch required"
    print(f"device: {_DEV}")
    cam = Camera(eye=(0, -6, 0), target=(0, 0, 0), W=128, H=128, fov_deg=45)

    # 1) a single isotropic Gaussian renders a centered, symmetric 2-D blob.
    means = np.array([[0, 0, 0.0]]); cov = np.eye(3)[None] * 0.04
    col = np.array([[1.0, 0.4, 0.1]]); op = np.array([0.9])
    img, al = render(means, cov, col, op, cam)
    ys, xs = np.where(al > 0.05)
    cyx = (ys.mean(), xs.mean())
    print(f"1) single splat: alpha peak at ({cyx[0]:.1f},{cyx[1]:.1f}) ~ image centre (64,64); "
          f"max alpha {al.max():.2f}")
    assert abs(cyx[0] - 64) < 4 and abs(cyx[1] - 64) < 4 and al.max() > 0.5

    # 2) determinism: re-render is bit-identical (stable sort + fixed-order compositing).
    img2, _ = render(means, cov, col, op, cam)
    print(f"2) determinism: re-render identical = {np.array_equal(img, img2)}")
    assert np.array_equal(img, img2)

    # 3) alpha compositing: a near (opaque, blue) splat occludes a far (red) one behind it.
    means = np.array([[0, 0, 0.0], [0, 2.0, 0.0]])          # blue nearer the camera (−y side)
    cov = np.eye(3)[None].repeat(2, 0) * 0.05
    col = np.array([[0, 0, 1.0], [1.0, 0, 0]]); op = np.array([0.98, 0.98])
    img, _ = render(means, cov, col, op, cam, bg=(0, 0, 0))
    centre = img[64, 64]
    print(f"3) occlusion: centre pixel RGB {np.round(centre,2)} (blue front should dominate red back)")
    assert centre[2] > centre[0]

    # 4) tonemap maps HDR (>1) into [0,1].
    out = tonemap(img * 5.0, bloom=0.3)
    print(f"4) tonemap range [{out.min():.2f},{out.max():.2f}] within [0,1]")
    assert out.min() >= 0 and out.max() <= 1
    print("\ngaussian_rasterizer self-checks passed.")
