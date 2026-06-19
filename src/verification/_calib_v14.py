"""Scratch calibration for V1.4 (B truncation, C macro/micro, D symmetry-lock). NOT committed."""
import numpy as np
import coupling_pipeline as cp

KAPPA = 0.6


def pipeline(sk, C0):
    basis, eig, lab = cp.make_basis(sk)
    ref, bnd = cp.silhouette(cp.all_points(sk, C0, KAPPA))
    Cl = cp.lift_from_silhouette(sk, ref, bnd)
    Chat = cp.gft_forward(Cl, basis)
    return basis, eig, lab, ref, bnd, Cl, Chat


sk = cp.biped_skeleton(); C0 = cp.biped_target()
basis, eig, lab, ref, bnd, Cl, Chat = pipeline(sk, C0)
full = cp.gft_inverse(Chat, basis)
mask_full, _ = cp.silhouette(cp.all_points(sk, full, KAPPA), bounds=bnd)

# ---- (B) graceful truncation: keep n lowest graph modes (all channels) ----
print("=== (B) truncation LOD (biped): n_graph -> coeff-error (monotone) + IoU (no spike) ===")
order = np.argsort(eig)
cerr, ious = [], []
for ng in range(1, sk["N"] + 1):
    Ct = cp.truncate(Chat, eig, ng, cp.K)
    cerr.append(float(np.linalg.norm(Ct - Chat)))                       # coeff-space (Parseval) error
    m, _ = cp.silhouette(cp.all_points(sk, cp.gft_inverse(Ct, basis), KAPPA), bounds=bnd)
    ious.append(cp.iou(m, mask_full))
coeff_mono = all(cerr[i + 1] <= cerr[i] + 1e-12 for i in range(len(cerr) - 1))
max_drop = max([ious[i] - ious[i + 1] for i in range(len(ious) - 1)] + [0.0])
print("  coeff error:", [f"{x:.3f}" for x in cerr])
print("  IoU vs full:", [f"{x:.3f}" for x in ious])
print(f"  coeff-error monotone non-increasing: {coeff_mono}   max IoU single-step drop = {max_drop:.3f}")

# ---- (C) macro vs micro: perturb a low (m,k) vs a high (m,k) coefficient ----
print("\n=== (C) macro vs micro (biped) ===")
area0 = mask_full.sum()
m_lo, k_lo = order[0], 0          # lowest graph mode, constant-radius channel = global bulk
m_hi, k_hi = order[-1], cp.K - 1  # highest graph mode, theta-harmonic channel = micro detail
delta = 0.5


def area_after(mm, kk, d):
    Cp = Chat.copy(); Cp[mm, kk] += d
    m, _ = cp.silhouette(cp.all_points(sk, cp.gft_inverse(Cp, basis), KAPPA), bounds=bnd)
    return m.sum(), m


dA_lo = abs(area_after(m_lo, k_lo, delta)[0] - area0) / area0
dA_hi = abs(area_after(m_hi, k_hi, delta)[0] - area0) / area0
print(f"  Δarea low (m,k)=({m_lo},{k_lo}): {dA_lo:.3f}   high ({m_hi},{k_hi}): {dA_hi:.4f}")
print(f"  macro/micro ratio = {dA_lo / (dA_hi + 1e-9):.1f}  (need >=10)")

# ---- (D) symmetry lock: lock the symmetry-breaking irreps -> EXACT coeff-space symmetry ----
# Exact, raster-free metric: sym_residual(C) = ||C - permute(C, group)|| / ||C||.
def perm_apply(C, perm):
    out = np.zeros_like(C)
    for i, pi in enumerate(perm):
        out[pi] = C[i]
    return out

def sym_residual(C, perm):
    Cp = perm_apply(C, perm)
    return float(np.linalg.norm(C - Cp) / (np.linalg.norm(C) + 1e-12))

print("\n=== (D) symmetry lock (biped Z2) — coeff-space residual (exact) ===")
C_asym = cp.biped_target()
C_asym[6, 0] += 0.10; C_asym[7, 0] += 0.08    # thicken LEFT arm only (break symmetry)
_, _, char = cp.make_basis(sk)
refa, bnda = cp.silhouette(cp.all_points(sk, C_asym, KAPPA))
Cla = cp.lift_from_silhouette(sk, refa, bnda)
Chata = cp.gft_forward(Cla, basis)
C_recon = cp.gft_inverse(Chata, basis)
Chat_lock = Chata.copy(); Chat_lock[char < 0] = 0.0      # lock ANTI (symmetry-breaking) modes to 0
C_locked = cp.gft_inverse(Chat_lock, basis)
print(f"  asym recon  sym-residual = {sym_residual(C_recon, sk['perm']):.4f}")
print(f"  ANTI-locked sym-residual = {sym_residual(C_locked, sk['perm']):.2e}  (need ~0)")
# geometric confirmation (raster, x-centered bounds so fliplr is the exact mirror)
def mirror_iou(C):
    pts = cp.all_points(sk, C, KAPPA)
    half = np.abs(pts[:, :2]).max() + 0.2
    bb = (np.array([-half, -half]), np.array([half, half]))
    m, _ = cp.silhouette(pts, bounds=bb); return cp.iou(m, m[:, ::-1])
print(f"  mirror-IoU: symmetric target {mirror_iou(cp.biped_target()):.3f}  "
      f"asym {mirror_iou(C_recon):.3f}  locked {mirror_iou(C_locked):.3f}")

print("\n=== (D) symmetry lock (seraph C6) — coeff-space residual (exact) ===")
sk6 = cp.seraph_skeleton(); C6 = cp.seraph_target()
C6[2, 0] += 0.12; C6[3, 0] += 0.10              # fatten ONE wing (break 6-fold)
b6, e6, m6 = cp.make_basis(sk6)
r6, bn6 = cp.silhouette(cp.all_points(sk6, C6, KAPPA))
Cl6 = cp.lift_from_silhouette(sk6, r6, bn6)
Ch6 = cp.gft_forward(Cl6, b6)
C6_recon = cp.gft_inverse(Ch6, b6)
keep = np.isin(m6, [0, 3])                        # keep only m=0 (and m=3 sign motif)
Ch6_lock = Ch6.copy(); Ch6_lock[~keep] = 0.0
C6_locked = cp.gft_inverse(Ch6_lock, b6)
print(f"  asym recon  sym-residual = {sym_residual(C6_recon, sk6['perm']):.4f}")
print(f"  m∈{{0,3}}-locked sym-residual = {sym_residual(C6_locked, sk6['perm']):.2e}  (need ~0)")
