"""
Coupling operator — load-bearing core: the SYMMETRY-ADAPTED graph Fourier basis.

Claim under test: a symmetric skeleton forces Laplacian eigenvalue degeneracy,
which makes a naive graph Fourier transform ill-posed; the declared symmetry
group resolves it canonically and splits modes into symmetric (symmetry-
preserving) vs antisymmetric (symmetry-breaking). This file proves that on a
bilateral biped using only numpy.
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# ---- biped skeleton graph (nodes = joints, edges = bones) ----
names = ['pelvis','spine','chest','neck','head',
         'L_sho','L_elb','L_hand','R_sho','R_elb','R_hand',
         'L_hip','L_knee','L_foot','R_hip','R_knee','R_foot']
N = len(names)
edges = [(0,1),(1,2),(2,3),(3,4),
         (2,5),(5,6),(6,7),  (2,8),(8,9),(9,10),
         (0,11),(11,12),(12,13),  (0,14),(14,15),(15,16)]
A = np.zeros((N,N))
for i,j in edges: A[i,j]=A[j,i]=1.0
L = np.diag(A.sum(1)) - A

# ---- bilateral symmetry sigma: swap Left <-> Right limbs ----
swap = {5:8,6:9,7:10, 11:14,12:15,13:16}
perm = list(range(N))
for a,b in swap.items(): perm[a],perm[b]=b,a
P = np.zeros((N,N))
for i,pi in enumerate(perm): P[pi,i]=1.0

print("1) symmetry check:  ||L P - P L|| =", np.linalg.norm(L@P - P@L),
      " (==0 => L respects the symmetry)\n")

# ---- naive eigendecomposition ----
w, V = np.linalg.eigh(L)

def clusters(w, tol=1e-6):
    out=[]; i=0
    while i<len(w):
        j=i
        while j+1<len(w) and w[j+1]-w[i]<tol: j+=1
        out.append((i,j)); i=j+1
    return out
cl = clusters(w)

print("2) Laplacian spectrum (degeneracies are forced by the symmetry):")
for a,b in cl:
    print(f"     lambda={w[a]:.4f}   multiplicity={b-a+1}")
print()

print("3) RAW eigenvectors inside degenerate clusters are symmetry-MIXED")
print("   (<v,Pv> is neither +1 nor -1, so 'symmetric vs antisymmetric' is undefined):")
for a,b in cl:
    if b>a:
        for k in range(a,b+1):
            print(f"     lambda={w[k]:.4f}   <v,Pv>={V[:,k]@(P@V[:,k]):+.3f}")
print()

# ---- symmetry-adapted basis: diagonalize P within each L-eigenspace ----
Vad = np.zeros_like(V); char = np.zeros(N)
for a,b in cl:
    blk = V[:,a:b+1]
    pe, pv = np.linalg.eigh(blk.T @ P @ blk)   # eigenvalues are +-1
    Vad[:,a:b+1] = blk @ pv
    char[a:b+1] = pe

print("4) SYMMETRY-ADAPTED basis: every mode is now PURE (+1 sym / -1 antisym)")
print("   AND still an exact Laplacian eigenvector (residual ~ 0):")
for k in range(N):
    res = np.linalg.norm(L@Vad[:,k] - w[k]*Vad[:,k])
    print(f"     lambda={w[k]:.4f}   P-eig={char[k]:+.2f} "
          f"[{'SYM ' if char[k]>0 else 'ANTI'}]   L-residual={res:.1e}")
print()

# ---- authoring payoff: where do edits live in the spectrum? ----
def split_energy(f):
    c = Vad.T @ f
    return float(np.sum(c[char>0]**2)), float(np.sum(c[char<0]**2))

f_both = np.zeros(N); f_both[[5,6,7,8,9,10]] = 1.0   # lengthen BOTH arms (symmetric)
f_left = np.zeros(N); f_left[[5,6,7]]        = 1.0   # lengthen LEFT arm only (asymmetric)

print("5) authoring test — project an edit onto the adapted basis:")
for label,f in [("lengthen both arms (symmetry-preserving)", f_both),
                ("lengthen left arm only (symmetry-breaking)", f_left)]:
    es,ea = split_energy(f)
    print(f"     {label:42s}  SYM={es:.3f}  ANTI={ea:.3f}")
print("\n   => symmetric edits live ENTIRELY in SYM coefficients; breaking symmetry")
print("      REQUIRES ANTI coefficients. Locking ANTI=0 guarantees bilateral symmetry.")