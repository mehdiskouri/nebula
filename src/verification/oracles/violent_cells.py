"""
Violent-regime cell generators + homogenized descriptor / coarse graph (V2.4, V2.1).

Supplies the battery the surrogate trains on and the handoff rule is judged over:
  - an ARCHETYPE PARAMETER FAMILY (char-wedge cells over depth x contrast) — one archetype
    spanning a parameter family, per Decision #17 ("train on the archetype, condition on the
    descriptor");
  - an explicit OUT-OF-DISTRIBUTION set (diagonal percolating seams, extreme contrast, random
    char blobs) — different topology/contrast than the family, so a faithful surrogate's
    uncertainty must rise and the predicate must trigger fallback (V2.4 metric 2).

Also provides the two things the surrogate consumes, both derived the SAME way the runtime
restriction operator derives them so "coarsening feature = surrogate input = validity state"
(ARCHITECTURE §III.5):
  - `descriptor(...)` — the homogenized feature vector (Voigt/Reuss diagonals + V-R gap +
    fractions + contrast) computed on the UNDAMAGED cell;
  - `region_graph(...)` — a coarse region-adjacency graph (node features + 6-neighbour edges)
    for the physics-informed graph net.

Imports `cells.py` and `homogenization.py`; does not edit them (regression discipline).
Pure numpy; deterministic given a seeded numpy Generator.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import cells
from homogenization import isotropic_stiffness, voigt_bound, reuss_bound, relative_gap

E_WOOD = 10.0
NU = 0.3


@dataclass
class Sample:
    """A violent-regime cell plus the generative parameters that produced it."""
    grid: np.ndarray
    materials: list
    kind: str
    theta: np.ndarray              # generative parameters (family coordinate)
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- generators
def wedge_sample(n, depth, contrast):
    """Archetype family member: a char wedge (reuses cells.char_wedge_cell)."""
    c = cells.char_wedge_cell(n=n, depth=depth, contrast=contrast, E_wood=E_WOOD, nu=NU)
    return Sample(grid=c.grid, materials=c.materials, kind="char_wedge",
                  theta=np.array([depth, contrast], float),
                  meta={"char_fraction": c.meta["char_fraction"], "contrast": contrast})


def seam_cell(n, angle_deg, thickness=2, contrast=60.0):
    """Thin CONNECTED low-stiffness seam crossing the cell at `angle_deg` in the x-z plane.

    A percolating seam destroys stiffness out of proportion to its volume fraction; off-axis
    (angle != 0/90) it is the named blind spot of volume-fraction homogenization — solidly OOD
    relative to the axis-aligned wedge family.
    """
    materials = [(E_WOOD, NU), (E_WOOD / contrast, NU)]
    grid = np.zeros((n, n, n), dtype=np.int64)
    xi = (np.arange(n) + 0.5)
    zi = (np.arange(n) + 0.5)
    th = np.deg2rad(angle_deg)
    nx, nz = np.sin(th), -np.cos(th)            # seam normal in x-z
    X, Z = np.meshgrid(xi, zi, indexing="ij")
    signed = (X - n / 2.0) * nx + (Z - n / 2.0) * nz
    mask_xz = np.abs(signed) < (thickness / 2.0)   # (nx,nz)
    grid[mask_xz[:, None, :].repeat(n, axis=1)] = 1
    return grid, materials


def seam_sample(n, angle_deg, contrast):
    grid, materials = seam_cell(n, angle_deg, thickness=2, contrast=contrast)
    return Sample(grid=grid, materials=materials, kind="seam",
                  theta=np.array([angle_deg, contrast], float),
                  meta={"angle_deg": angle_deg, "contrast": contrast})


def blob_sample(n, n_blobs, contrast, rng):
    """Random scattered char blobs — a topology the wedge family never shows (OOD)."""
    materials = [(E_WOOD, NU), (E_WOOD / contrast, NU)]
    grid = np.zeros((n, n, n), dtype=np.int64)
    for _ in range(n_blobs):
        c = rng.integers(2, n - 2, size=3)
        r = rng.integers(1, max(2, n // 5))
        ii, jj, kk = np.ogrid[:n, :n, :n]
        grid[((ii - c[0]) ** 2 + (jj - c[1]) ** 2 + (kk - c[2]) ** 2) <= r * r] = 1
    return Sample(grid=grid, materials=materials, kind="blob",
                  theta=np.array([n_blobs, contrast], float),
                  meta={"n_blobs": n_blobs, "contrast": contrast})


# ----------------------------------------------------------------------------- batteries
def family_battery(n, rng, n_samples, depth_rng=(0.2, 0.85), contrast_rng=(20.0, 80.0)):
    """In-distribution archetype family: char wedges over (depth, contrast)."""
    out = []
    for _ in range(n_samples):
        depth = float(rng.uniform(*depth_rng))
        contrast = float(rng.uniform(*contrast_rng))
        out.append(wedge_sample(n, depth, contrast))
    return out


def ood_battery(n, rng, n_per=8):
    """Out-of-distribution set: off-axis percolating seams, extreme contrast, random blobs."""
    out = []
    for _ in range(n_per):                                   # off-axis percolating seams
        out.append(seam_sample(n, float(rng.uniform(20.0, 70.0)), float(rng.uniform(40.0, 90.0))))
    for _ in range(n_per):                                   # extreme contrast wedges
        out.append(wedge_sample(n, float(rng.uniform(0.3, 0.8)), float(rng.uniform(300.0, 800.0))))
    for _ in range(n_per):                                   # random blob topology
        out.append(blob_sample(n, int(rng.integers(3, 8)), float(rng.uniform(40.0, 90.0)), rng))
    return out


# ----------------------------------------------------------------------------- features
def descriptor(grid, materials):
    """Homogenized descriptor of the UNDAMAGED cell (the restriction-operator output).

    [ Voigt diag(6) | Reuss diag(6) | V-R gap(6) | soft_fraction | log10(contrast) ] -> 20-vec.
    """
    phases = np.asarray(grid).ravel()
    P = len(materials)
    frac = np.bincount(phases, minlength=P) / phases.size
    C_phases = [isotropic_stiffness(E, nu) for (E, nu) in materials]
    Cv = voigt_bound(frac, C_phases)
    Cr = reuss_bound(frac, C_phases)
    gap = relative_gap(Cv, Cr)
    Es = np.array([E for (E, _) in materials], float)
    contrast = float(Es.max() / Es.min())
    soft_frac = float(frac[int(np.argmin(Es))]) if P > 1 else 0.0
    return np.concatenate([np.diag(Cv), np.diag(Cr), gap,
                           [soft_frac, np.log10(contrast)]]).astype(np.float64)


def percolates(grid, materials, axis=None):
    """Does the SOFT phase form a connected cluster spanning two opposite faces?

    Volume-fraction homogenization (hence the descriptor envelope) is blind to connectivity:
    a thin connected seam destroys stiffness out of proportion to its volume (ARCHITECTURE
    Risk: percolation; V2.2). This 6-connectivity span test is the architecture's prescribed
    hard guard for that blind spot — it catches the percolating seam the envelope misses, while
    leaving non-spanning wedges/blobs untouched.
    """
    Es = np.array([E for (E, _) in materials], float)
    soft = int(np.argmin(Es))
    mask = (np.asarray(grid) == soft)
    if not mask.any():
        return False
    lab, _ = ndimage.label(mask)
    axes = [axis] if axis is not None else range(3)
    for ax in axes:
        lo = set(np.unique(lab.take(0, axis=ax))) - {0}
        hi = set(np.unique(lab.take(-1, axis=ax))) - {0}
        if lo & hi:
            return True
    return False


def region_graph(grid, materials, R=4):
    """Coarse RxRxR region-adjacency graph for the physics-informed graph net.

    Node features per super-voxel: [soft-phase fraction, mean E (normalized by E_wood)].
    Edges: 6-neighbour connectivity (undirected, stored both directions). Returns
    (node_feats (R^3, 2), edge_index (2, n_edges) int64).
    """
    grid = np.asarray(grid)
    n = grid.shape[0]
    Es = np.array([E for (E, _) in materials], float)
    soft = int(np.argmin(Es))
    E_field = Es[grid]                                              # (n,n,n)
    bins = np.linspace(0, n, R + 1).astype(int)
    feats = np.zeros((R, R, R, 2))
    for a in range(R):
        for b in range(R):
            for c in range(R):
                sub = grid[bins[a]:bins[a + 1], bins[b]:bins[b + 1], bins[c]:bins[c + 1]]
                subE = E_field[bins[a]:bins[a + 1], bins[b]:bins[b + 1], bins[c]:bins[c + 1]]
                feats[a, b, c, 0] = float((sub == soft).mean())
                feats[a, b, c, 1] = float(subE.mean()) / E_WOOD
    node_feats = feats.reshape(R ** 3, 2)
    idx = np.arange(R ** 3).reshape(R, R, R)
    edges = []
    for axis in range(3):
        sl_a = [slice(None)] * 3; sl_b = [slice(None)] * 3
        sl_a[axis] = slice(0, R - 1); sl_b[axis] = slice(1, R)
        u = idx[tuple(sl_a)].ravel(); v = idx[tuple(sl_b)].ravel()
        edges.append(np.stack([u, v])); edges.append(np.stack([v, u]))
    edge_index = np.concatenate(edges, axis=1).astype(np.int64)
    return node_feats.astype(np.float64), edge_index


DESCRIPTOR_DIM = 20
NODE_FEAT_DIM = 2


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    rng = np.random.default_rng(0)
    n = 16

    fam = family_battery(n, rng, 6)
    ood = ood_battery(n, rng, 3)
    print(f"1) family battery: {len(fam)} char-wedge cells; OOD battery: {len(ood)} cells")
    print("   kinds in OOD:", sorted(set(s.kind for s in ood)))

    d = descriptor(fam[0].grid, fam[0].materials)
    print(f"2) descriptor dim = {d.size} (expect {DESCRIPTOR_DIM}); first cell:\n   {d}")
    assert d.size == DESCRIPTOR_DIM

    nf, ei = region_graph(fam[0].grid, fam[0].materials, R=4)
    print(f"3) region graph: nodes {nf.shape}, edges {ei.shape}")
    assert nf.shape == (64, NODE_FEAT_DIM)
    assert ei.shape[0] == 2 and ei.shape[1] == 2 * 3 * (4 - 1) * 4 * 4  # 2*directions count
    # node features bounded
    assert nf[:, 0].min() >= 0.0 and nf[:, 0].max() <= 1.0

    # descriptors of family vs OOD should be separable (OOD contrast/topology differs).
    Dfam = np.stack([descriptor(s.grid, s.materials) for s in family_battery(n, rng, 20)])
    Dood = np.stack([descriptor(s.grid, s.materials) for s in ood_battery(n, rng, 8)])
    mu, sd = Dfam.mean(0), Dfam.std(0) + 1e-9
    z_fam = np.abs((Dfam - mu) / sd).max(1)
    z_ood = np.abs((Dood - mu) / sd).max(1)
    print(f"4) descriptor separability: family max-z median={np.median(z_fam):.2f}, "
          f"OOD max-z median={np.median(z_ood):.2f}")
    assert np.median(z_ood) > np.median(z_fam), "OOD should sit further from the family mean"
    print("\nALL violent_cells self-checks PASSED")
