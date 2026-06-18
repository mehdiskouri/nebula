"""
C6 stress test of the coupling operator's symmetry core.

Bilateral (Z2) under-tests it: all Z2 irreps are 1-D, so raw eigenvectors come
out symmetry-pure by accident. C6 has genuine 2-D irreps (angular momentum
m = +-1, +-2 around the 6-fold axis). In those eigenspaces the raw eigensolver
returns an ARBITRARY rotated basis with no definite m -- and the character
projector is what assigns a canonical angular-momentum label. This proves it.
"""
import numpy as np
norm = np.linalg.norm
np.set_printoptions(precision=3, suppress=True)

# ---- C6 seraph scaffold: core + 6 identical wings + halo ring ----
# node 0 = core (fixed by rotation); wing w: root=1+3w, mid=2+3w, tip=3+3w
N = 19
edges = []
for w in range(6):
    root, mid, tip = 1+3*w, 2+3*w, 3+3*w
    edges += [(0, root), (root, mid), (mid, tip)]        # core -> wing chain
    edges.append((root, 1 + 3*((w+1) % 6)))              # halo ring (couples wings)
A = np.zeros((N, N))
for i, j in edges: A[i, j] = A[j, i] = 1.0
L = np.diag(A.sum(1)) - A

# ---- C6 generator rho: rotate wing w -> w+1, core fixed ----
perm = list(range(N))
for w in range(6):
    nw = (w+1) % 6
    perm[1+3*w], perm[2+3*w], perm[3+3*w] = 1+3*nw, 2+3*nw, 3+3*nw
R = np.zeros((N, N))
for i, pi in enumerate(perm): R[pi, i] = 1.0

R6 = np.linalg.matrix_power(R, 6)
print(f"1) symmetry:  ||LR-RL||={norm(L@R-R@L):.1e}   ||R^6 - I||={norm(R6-np.eye(N)):.1e}")
print("   (both ~0 => rho is an order-6 symmetry of the operator)\n")

# ---- spectrum, annotated by angular-momentum irrep of each eigenspace ----
w_, V = np.linalg.eigh(L)
def clusters(w, tol=1e-6):
    out, i = [], 0
    while i < len(w):
        j = i
        while j+1 < len(w) and w[j+1]-w[i] < tol: j += 1
        out.append((i, j)); i = j+1
    return out
cl = clusters(w_)

def irrep_of(block):                      # which m's does rho act as on this eigenspace?
    ev = np.linalg.eigvals(block.T @ R @ block)
    ms = sorted({int(round(np.angle(e)/(np.pi/3))) % 6 for e in ev})
    return ms

print("2) spectrum organized by angular momentum m around the 6-fold axis:")
two_d_cluster = None
for a, b in cl:
    ms = irrep_of(V[:, a:b+1])
    lbl = {(0,):"m=0", (3,):"m=3", (1,5):"m=+-1", (2,4):"m=+-2"}.get(tuple(ms), f"m in {ms}")
    print(f"     lambda={w_[a]:.4f}  mult={b-a+1:d}  irrep {lbl}")
    if b > a and tuple(ms) in [(1,5),(2,4)] and two_d_cluster is None:
        two_d_cluster = (a, b, lbl)
print("   => m=0 and m=3 are non-degenerate; m=+-1, m=+-2 come as DEGENERATE PAIRS")
print("      (the 2-D irreps the bilateral biped never produced)\n")

# ---- the load-bearing contrast: raw eigenvectors in a 2-D irrep have NO definite m ----
a, b, lbl = two_d_cluster
v0 = V[:, a]
Rv0 = R @ v0
overlap = float(v0 @ Rv0)
perp = norm(Rv0 - overlap*v0)
print(f"3) RAW eigenvector in the {lbl} eigenspace (lambda={w_[a]:.4f}):")
print(f"     <v, R v>={overlap:+.3f}   ||R v - <v,Rv> v||={perp:.3f}")
print(f"     rho ROTATES v into its partner (perp>>0) -> v has no definite m.")
print(f"     [contrast: bilateral case always gave R v = +-v exactly, perp=0]\n")

# ---- symmetry-adapted: complex modes with definite angular momentum ----
B = V[:, a:b+1]
ev, evec = np.linalg.eig(B.T @ R @ B)       # unit-modulus eigenvalues e^{i*60*m}
print("4) SYMMETRY-ADAPTED modes: rho acts as a pure phase (definite m), L exact:")
for k in range(len(ev)):
    u = B @ evec[:, k]
    m = int(round(np.angle(ev[k])/(np.pi/3))) % 6
    r_res = norm(R@u - ev[k]*u)             # rho u = e^{i theta} u
    l_res = norm(L@u - w_[a]*u)
    print(f"     m={m}  rho-phase={np.angle(ev[k])/np.pi*180:+6.1f} deg  "
          f"rho-residual={r_res:.1e}  L-residual={l_res:.1e}")
print()

# ---- authoring payoff: where do edits live, by angular momentum? ----
omega = np.exp(2j*np.pi/6)
def Pm(m):                                   # projector onto angular-momentum m
    M, Rk = np.zeros((N, N), complex), np.eye(N)
    for k in range(6):
        M += omega**(-m*k) * Rk; Rk = R @ Rk
    return M/6
def spectrum_over_m(f):
    return [float(np.real(np.vdot(Pm(m)@f, Pm(m)@f))) for m in range(6)]

tips = [3+3*w for w in range(6)]
f_all = np.zeros(N); f_all[tips] = 1.0                     # all six wings, equal
f_alt = np.zeros(N); f_alt[[tips[w] for w in (0,2,4)]] = 1 # alternating wings
f_one = np.zeros(N); f_one[tips[0]] = 1.0                  # a single wing

print("5) authoring test -- energy of an edit by angular momentum m=[0,1,2,3,4,5]:")
for label, f in [("all wings equal (6-fold symmetric)", f_all),
                 ("alternating wings (3-fold motif) ", f_alt),
                 ("single wing  (symmetry-breaking)  ", f_one)]:
    e = spectrum_over_m(f)
    print(f"     {label}: {np.array(e)}")
print("\n   all-equal  -> pure m=0  (author ONE wing, m=0 replicates it to six)")
print("   alternating-> m=0 + m=3 (a real motif from the sign irrep)")
print("   single wing-> spread across ALL m, incl. the 2-D m=+-1,+-2 irreps")
print("   => breaking 6-fold symmetry is unrepresentable without resolving them.")