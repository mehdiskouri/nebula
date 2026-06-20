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


def violent_battery(n, rng, n_easy=34, n_hard=8,
                    easy_depth=(0.2, 0.85), easy_contrast=(20.0, 80.0),
                    hard_depth=(0.80, 0.95), hard_contrast=(95.0, 180.0)):
    """The V2.1 violent-regime DEPLOYMENT population: an in-family majority + a hard extrapolation
    tail (deeper, higher-contrast wedges that push the surrogate off its training box — the cells
    whose error its bare self-uncertainty saturates on).

    Centralizing it here is load-bearing for V2.1's calibration discipline: the SCORED battery and
    the HELD-OUT CALIBRATION split must be *exchangeable* draws of the same distribution (same
    generator, different seeds) for split-conformal / held-out recalibration to be valid. The draw
    order (family_battery then the hard loop) matches the original inline construction, so seeding
    with rng=default_rng(777) reproduces the exact pre-existing battery — keeping its DNS cache
    valid.
    """
    easy = family_battery(n, rng, n_easy, depth_rng=easy_depth, contrast_rng=easy_contrast)
    hard = [wedge_sample(n, float(rng.uniform(*hard_depth)), float(rng.uniform(*hard_contrast)))
            for _ in range(n_hard)]
    return easy + hard


# ----------------------------------------------------------------------------- features
def descriptor(grid, materials, connectivity=False):
    """Homogenized descriptor of the UNDAMAGED cell (the restriction-operator output).

    [ Voigt diag(6) | Reuss diag(6) | V-R gap(6) | soft_fraction | log10(contrast) ] -> 20-vec.

    `connectivity=False` (default) reproduces the original 20-vec BYTE-IDENTICALLY — so V2.4 (and the
    V2.1 RPF surrogate) and their cached DNS truth are unchanged. `connectivity=True` APPENDS the
    directional conductance residual (`percolation.connectivity_residual`, 3 channels) -> 23-vec, the
    V2.2 fix that folds connectivity INTO the descriptor/trust-scalar coordinate (so the validity
    envelope and `u` see it) instead of a parallel boolean gate. The conductance residual is the
    PHYSICS signal (it does not over-count dense scatter the way a topological span check does). Opt-in,
    additive, mirroring the surrogate's `TrainCfg.beta=0/1` discipline. The connectivity term is lazily
    imported to keep the `percolation -> violent_cells` dependency direction acyclic.
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
    base = np.concatenate([np.diag(Cv), np.diag(Cr), gap,
                           [soft_frac, np.log10(contrast)]]).astype(np.float64)
    if not connectivity:
        return base
    from percolation import connectivity_residual               # lazy: avoids an import cycle
    return np.concatenate([base, connectivity_residual(grid, materials)]).astype(np.float64)


def percolates(grid, materials, axis=None, connectivity=1):
    """Does the SOFT phase form a connected cluster spanning two opposite faces?

    Volume-fraction homogenization (hence the descriptor envelope) is blind to connectivity:
    a thin connected seam destroys stiffness out of proportion to its volume (ARCHITECTURE
    Risk: percolation; V2.2). This span test is the architecture's prescribed hard guard for that
    blind spot — it catches the percolating seam the envelope misses, while leaving non-spanning
    wedges/blobs untouched.

    `connectivity` selects the voxel adjacency rule (scipy `generate_binary_structure(3, c)`):
    `1` = 6-connectivity (faces only) — the DEFAULT, preserving the original V2.4/V2.1 behaviour
    byte-for-byte; `3` = 26-connectivity (the hardened V2.2 rule) — also links face/edge/corner
    diagonals so a thin DIAGONAL soft path that is only corner-connected (which the 6-rule misses,
    forcing the old `thickness=3` crutch) still registers.
    """
    Es = np.array([E for (E, _) in materials], float)
    soft = int(np.argmin(Es))
    mask = (np.asarray(grid) == soft)
    if not mask.any():
        return False
    struct = ndimage.generate_binary_structure(3, connectivity)
    lab, _ = ndimage.label(mask, structure=struct)
    axes = [axis] if axis is not None else range(3)
    for ax in axes:
        lo = set(np.unique(lab.take(0, axis=ax))) - {0}
        hi = set(np.unique(lab.take(-1, axis=ax))) - {0}
        if lo & hi:
            return True
    return False


def spanning_cluster_fraction(grid, materials, axis=None, connectivity=3):
    """Graded discrete companion to the boolean `percolates`: the fraction of the SOFT phase that
    lies in a face-spanning cluster (0 if nothing spans; -> 1 for a fully connected seam). Uses
    26-connectivity by default so thin diagonal soft paths register. A smooth proxy for "how
    connected" that complements the continuous conductance residual.
    """
    Es = np.array([E for (E, _) in materials], float)
    soft = int(np.argmin(Es))
    mask = (np.asarray(grid) == soft)
    if not mask.any():
        return 0.0
    struct = ndimage.generate_binary_structure(3, connectivity)
    lab, _ = ndimage.label(mask, structure=struct)
    axes = [axis] if axis is not None else range(3)
    spanning = set()
    for ax in axes:
        lo = set(np.unique(lab.take(0, axis=ax))) - {0}
        hi = set(np.unique(lab.take(-1, axis=ax))) - {0}
        spanning |= (lo & hi)
    if not spanning:
        return 0.0
    return float(np.isin(lab, list(spanning)).sum()) / float(mask.sum())


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


DESCRIPTOR_DIM = 20                 # fraction-only descriptor (connectivity=False)
DESCRIPTOR_DIM_CONNECTIVITY = 23    # + directional conductance residual g_perc(3) (connectivity=True)
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
    # connectivity=True APPENDS 3 channels and leaves the first 20 BYTE-IDENTICAL (reproduction).
    dc = descriptor(fam[0].grid, fam[0].materials, connectivity=True)
    assert dc.size == DESCRIPTOR_DIM_CONNECTIVITY and np.array_equal(dc[:DESCRIPTOR_DIM], d), \
        "connectivity=True must append channels without disturbing the original 20-vec"
    print(f"   connectivity=True -> dim {dc.size}; appended g_perc = {np.round(dc[DESCRIPTOR_DIM:],3)}")

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

    # 5) violent_battery centralizes the V2.1 battery — verify it reproduces the original inline
    #    construction byte-for-byte (so the cached DNS truth stays valid).
    b1 = violent_battery(n, np.random.default_rng(777))
    cr = np.random.default_rng(777)
    b0 = (family_battery(n, cr, 34)
          + [wedge_sample(n, float(cr.uniform(0.80, 0.95)), float(cr.uniform(95.0, 180.0)))
             for _ in range(8)])
    assert len(b1) == len(b0) and all(np.array_equal(a.grid, g.grid) for a, g in zip(b1, b0)), \
        "violent_battery(seed=777) must reproduce the original inline battery (DNS cache stays valid)"
    print(f"5) violent_battery reproduces the original {len(b1)}-cell battery: OK")
    print("\nALL violent_cells self-checks PASSED")
