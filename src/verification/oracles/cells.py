"""
Deterministic microstructure generators for verification cells (protocol §7).

Builds the heterogeneous unit cells V0.1 sweeps over — and that V0.2 (char-wedge
criticality) and V2.2 (percolating seam) will reuse. A cell is a voxel phase
grid plus the per-phase material constants; everything is seeded / parametric so
runs are bit-reproducible.

Material constants are *representative plausibility-engine values*, not measured
clinical data (ARCHITECTURE.md Part VII: "biological parameters are largely
unknown / invented"). Do not read physiological meaning into the numbers; the
verification only needs realistic *contrasts*.

Voigt notation and the 6x6 stiffness convention follow homogenization.py.
"""
from dataclasses import dataclass, field

import numpy as np

from homogenization import isotropic_stiffness

# Representative isotropic moduli (E in arbitrary consistent units, GPa-ish).
# Among the intact wood tissues the contrast is modest (a "quiescent" cell);
# char is ~50-100x softer (the high-contrast, structurally-critical case).
MATERIALS = {
    "bark":      (2.0, 0.30),
    "sapwood":   (9.0, 0.35),
    "heartwood": (12.0, 0.35),
    "char":      (0.15, 0.30),
}


@dataclass
class Cell:
    """A heterogeneous unit cell: integer phase grid + per-phase (E, nu)."""
    grid: np.ndarray                       # (nx, ny, nz) int phase ids
    materials: list                        # list of (E, nu) indexed by phase id
    kind: str                              # "layered" / "char_wedge" / "homogeneous"
    contrast: float                        # max/min phase stiffness ratio in the cell
    layer_axis: int = None                 # principal axis layers stack along (layered only)
    meta: dict = field(default_factory=dict)

    @property
    def C_phases(self):
        """Per-phase 6x6 stiffness tensors."""
        return [isotropic_stiffness(E, nu) for (E, nu) in self.materials]

    @property
    def fractions(self):
        """Volume fraction of each phase, aligned with `materials`."""
        counts = np.bincount(self.grid.ravel(), minlength=len(self.materials))
        return counts / counts.sum()

    @property
    def low_contrast(self, threshold=3.0):
        """Quiescent (low-contrast) cell? Used to gate the tightness criterion."""
        return self.contrast <= threshold


def _contrast_of(materials):
    Es = np.array([E for (E, _) in materials], float)
    return float(Es.max() / Es.min())


def homogeneous_cell(n=24, E=10.0, nu=0.3):
    """Single-phase cell — the trivial sanity case (C_eff must equal the phase)."""
    grid = np.zeros((n, n, n), dtype=np.int64)
    return Cell(grid=grid, materials=[(E, nu)], kind="homogeneous", contrast=1.0)


def layered_cell(n=24, fractions=(0.34, 0.33, 0.33),
                 moduli=None, nus=None, axis=2):
    """Clean planar layers stacked along `axis` (local model of concentric rings).

    `fractions` set each layer's thickness; `moduli`/`nus` its material. Defaults
    to the bark/sapwood/heartwood library. Returns a Cell whose principal-direction
    response should match the directional proxy exactly (layered-exactness claim).
    """
    fractions = np.asarray(fractions, float)
    fractions = fractions / fractions.sum()
    if moduli is None:
        moduli = [MATERIALS["bark"][0], MATERIALS["sapwood"][0], MATERIALS["heartwood"][0]]
    if nus is None:
        nus = [MATERIALS["bark"][1], MATERIALS["sapwood"][1], MATERIALS["heartwood"][1]]
    materials = list(zip(moduli, nus))

    # Partition the `axis` extent into contiguous layer slabs by fraction.
    edges = np.round(np.cumsum(fractions) * n).astype(int)
    edges[-1] = n
    grid = np.zeros((n, n, n), dtype=np.int64)
    start = 0
    idx = [slice(None)] * 3
    for phase, end in enumerate(edges):
        idx[axis] = slice(start, end)
        grid[tuple(idx)] = phase
        start = end
    return Cell(grid=grid, materials=materials, kind="layered",
                contrast=_contrast_of(materials), layer_axis=axis,
                meta={"fractions": fractions.tolist()})


def two_phase_layered(n=24, frac_stiff=0.5, contrast=2.0, axis=2,
                      E_stiff=10.0, nu=0.3):
    """Two-phase layered cell with an explicit stiffness `contrast` (for the sweep).

    Stiff phase E = E_stiff; soft phase E = E_stiff / contrast. Sweeping `contrast`
    from ~1.2 (quiescent) to ~100 (heavy char) spans V0.1's tightness population.
    """
    materials = [(E_stiff, nu), (E_stiff / contrast, nu)]
    grid = np.zeros((n, n, n), dtype=np.int64)
    edge = int(round(frac_stiff * n))
    idx = [slice(None)] * 3
    idx[axis] = slice(edge, n)
    grid[tuple(idx)] = 1
    return Cell(grid=grid, materials=materials, kind="layered",
                contrast=float(contrast), layer_axis=axis,
                meta={"frac_stiff": frac_stiff})


def char_wedge_cell(n=24, depth=0.5, contrast=60.0,
                    E_wood=10.0, nu=0.3):
    """Wood matrix invaded by a CHAR WEDGE (phase 1) deepening along z.

    `depth` in [0,1] sets how far the wedge tip penetrates (x-direction) at the
    far face. High `contrast` (char ~ 1/contrast as stiff) is the structurally
    critical, worst-for-homogenization case. Directly reused by V0.2.
    """
    materials = [(E_wood, nu), (E_wood / contrast, nu)]
    grid = np.zeros((n, n, n), dtype=np.int64)
    xi = (np.arange(n) + 0.5) / n
    zi = (np.arange(n) + 0.5) / n
    # char where x-depth is shallower than the wedge profile, which grows with z.
    profile = depth * zi  # (nz,)
    mask = xi[:, None] < profile[None, :]      # (nx, nz)
    grid[mask[:, None, :].repeat(n, axis=1)] = 1
    char_frac = grid.mean()
    return Cell(grid=grid, materials=materials, kind="char_wedge",
                contrast=float(contrast),
                meta={"depth": depth, "char_fraction": float(char_frac)})


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    print("1) library materials:", MATERIALS)

    c = layered_cell(n=12)
    print("\n2) layered tree cell: grid", c.grid.shape,
          "fractions", c.fractions, "contrast", round(c.contrast, 2),
          "low_contrast", c.low_contrast)

    c2 = two_phase_layered(n=12, contrast=1.5)
    print("3) two-phase contrast=1.5: fractions", c2.fractions,
          "low_contrast", c2.low_contrast)

    c3 = char_wedge_cell(n=16, depth=0.6)
    print("4) char wedge depth=0.6: char fraction",
          round(c3.meta["char_fraction"], 3), "contrast", c3.contrast,
          "low_contrast", c3.low_contrast)
