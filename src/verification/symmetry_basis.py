"""
Symmetry-adapted graph-Fourier basis — the reusable core of the coupling operator
(ARCHITECTURE §III.8; Decision #24). Factored out of the proven scripts
`coupling_operator_core.py` (Z2 biped) and `coupling_operator_c6.py` (C6 seraph), which
remain the untouched §8 baseline. V1.4 builds the full geometric pipeline on these functions.

The claim these encode (already validated): a symmetric skeleton forces Laplacian
eigenvalue degeneracy, so a naive graph-Fourier transform is ill-posed on the degenerate
eigenspaces; diagonalising the symmetry-group action WITHIN each Laplacian eigenspace (the
character projector) resolves it canonically and labels every mode by irrep —
  * Z2:  pure +1 (symmetric) / -1 (antisymmetric) modes;
  * Cn:  complex angular-momentum modes with definite m (e^{i*2pi*m/n} phase under rotation).

Pure numpy. `__main__` reproduces the two proven scripts' numbers (the regression guard).
"""
import numpy as np

norm = np.linalg.norm


# ---------------- graph builders ----------------

def laplacian(N, edges):
    A = np.zeros((N, N))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return np.diag(A.sum(1)) - A


def permutation_matrix(perm):
    N = len(perm)
    P = np.zeros((N, N))
    for i, pi in enumerate(perm):
        P[pi, i] = 1.0
    return P


def clusters(w, tol=1e-6):
    """Contiguous near-equal eigenvalue groups (the symmetry-forced degeneracies)."""
    out, i = [], 0
    while i < len(w):
        j = i
        while j + 1 < len(w) and w[j + 1] - w[i] < tol:
            j += 1
        out.append((i, j)); i = j + 1
    return out


# ---------------- Z2 (real, 1-D irreps): SYM / ANTI ----------------

def adapted_basis_real(L, P, tol=1e-6):
    """Symmetry-adapted real eigenbasis for an involution P (P^2 = I).

    Returns (w, Vad, char): eigenvalues, adapted eigenvectors (still exact L-eigenvectors),
    and char in {+1 (symmetric), -1 (antisymmetric)} per mode."""
    w, V = np.linalg.eigh(L)
    Vad = np.zeros_like(V); char = np.zeros(len(w))
    for a, b in clusters(w, tol):
        blk = V[:, a:b + 1]
        pe, pv = np.linalg.eigh(blk.T @ P @ blk)      # eigenvalues are +-1
        Vad[:, a:b + 1] = blk @ pv
        char[a:b + 1] = pe
    return w, Vad, char


def irrep_energy_real(signal, Vad, char):
    """(symmetric_energy, antisymmetric_energy) of a node signal in the adapted basis."""
    c = Vad.T @ np.asarray(signal, float)
    return float(np.sum(c[char > 0] ** 2)), float(np.sum(c[char < 0] ** 2))


# ---------------- Cn (genuine 2-D irreps): angular momentum m ----------------

def adapted_basis_cyclic(L, R, order, tol=1e-6):
    """Symmetry-adapted complex eigenbasis for an order-n rotation generator R.

    Returns (w, U, m): eigenvalues, complex adapted modes (columns), and integer angular
    momentum m in [0, order) per mode. In a degenerate eigenspace R acts as a pure phase
    e^{i*2pi*m/order} on each adapted mode (definite m); non-degenerate modes get m by the
    same phase. L-exact to solver tolerance."""
    w, V = np.linalg.eigh(L)
    N = L.shape[0]
    U = np.zeros((N, N), complex); m = np.zeros(N, int)
    for a, b in clusters(w, tol):
        B = V[:, a:b + 1]
        ev, evec = np.linalg.eig(B.T @ R @ B)         # unit-modulus eigenvalues e^{i*2pi*m/order}
        for k in range(ev.shape[0]):
            U[:, a + k] = B @ evec[:, k]
            m[a + k] = int(round(np.angle(ev[k]) / (2 * np.pi / order))) % order
    return w, U, m


def angular_momentum_energy(signal, R, order):
    """Energy of a node signal at each angular momentum m=0..order-1 (group projector).

    Independent of any basis: P_m = (1/n) Σ_k ω^{-mk} R^k, ω = e^{i2π/order}."""
    N = R.shape[0]; f = np.asarray(signal, complex); omega = np.exp(2j * np.pi / order)
    out = []
    for mm in range(order):
        M, Rk = np.zeros((N, N), complex), np.eye(N)
        for k in range(order):
            M += omega ** (-mm * k) * Rk; Rk = R @ Rk
        M /= order
        out.append(float(np.real(np.vdot(M @ f, M @ f))))
    return np.array(out)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    # ---- Z2 biped (reproduces coupling_operator_core.py) ----
    N = 17
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 7), (2, 8), (8, 9), (9, 10),
             (0, 11), (11, 12), (12, 13), (0, 14), (14, 15), (15, 16)]
    L = laplacian(N, edges)
    swap = {5: 8, 6: 9, 7: 10, 11: 14, 12: 15, 13: 16}
    perm = list(range(N))
    for a, b in swap.items():
        perm[a], perm[b] = b, a
    P = permutation_matrix(perm)
    w, Vad, char = adapted_basis_real(L, P)
    comm = norm(L @ P - P @ L)
    res = max(norm(L @ Vad[:, k] - w[k] * Vad[:, k]) for k in range(N))
    f_both = np.zeros(N); f_both[[5, 6, 7, 8, 9, 10]] = 1.0
    f_left = np.zeros(N); f_left[[5, 6, 7]] = 1.0
    print("Z2 biped:")
    print(f"  ||LP-PL|| = {comm:.1e}   max adapted L-residual = {res:.1e}")
    print(f"  both-arms edit  SYM/ANTI = {irrep_energy_real(f_both, Vad, char)}")
    print(f"  left-arm edit   SYM/ANTI = {irrep_energy_real(f_left, Vad, char)}")
    assert comm < 1e-10 and res < 1e-10
    assert irrep_energy_real(f_both, Vad, char)[1] < 1e-9          # symmetric edit -> 0 ANTI

    # ---- C6 seraph (reproduces coupling_operator_c6.py) ----
    Nc = 19; ce = []
    for wg in range(6):
        root, mid, tip = 1 + 3 * wg, 2 + 3 * wg, 3 + 3 * wg
        ce += [(0, root), (root, mid), (mid, tip), (root, 1 + 3 * ((wg + 1) % 6))]
    Lc = laplacian(Nc, ce)
    pc = list(range(Nc))
    for wg in range(6):
        nw = (wg + 1) % 6
        pc[1 + 3 * wg], pc[2 + 3 * wg], pc[3 + 3 * wg] = 1 + 3 * nw, 2 + 3 * nw, 3 + 3 * nw
    Rc = permutation_matrix(pc)
    wc, Uc, mc = adapted_basis_cyclic(Lc, Rc, 6)
    commc = norm(Lc @ Rc - Rc @ Lc)
    resc = max(norm(Lc @ Uc[:, k] - wc[k] * Uc[:, k]) for k in range(Nc))
    tips = [3 + 3 * wg for wg in range(6)]
    f_all = np.zeros(Nc); f_all[tips] = 1.0
    f_one = np.zeros(Nc); f_one[tips[0]] = 1.0
    e_all = angular_momentum_energy(f_all, Rc, 6)
    e_one = angular_momentum_energy(f_one, Rc, 6)
    print("\nC6 seraph:")
    print(f"  ||LR-RL|| = {commc:.1e}   max adapted L-residual = {resc:.1e}")
    print(f"  m-labels present: {sorted(set(mc))}")
    print(f"  all-wings edit  energy by m = {e_all}")
    print(f"  single-wing edit energy by m = {e_one}")
    assert commc < 1e-10 and resc < 1e-10
    assert e_all[0] > 1e-6 and np.sum(e_all[1:]) < 1e-9            # all-equal -> pure m=0
    assert np.count_nonzero(e_one > 1e-6) >= 4                     # single wing spreads over m
    print("\nOK — symmetry_basis reproduces the proven Z2 and C6 core results (§8 regression).")
