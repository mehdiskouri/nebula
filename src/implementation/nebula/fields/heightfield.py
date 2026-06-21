"""
Terrain heightfield (ARCHITECTURE Foundational thesis: heightfields for terrain).

The deterministic ground the tree is rooted in. Phase 0 keeps this minimal: a seeded,
band-limited value-noise surface h(x, y), with helpers to sample the height (so growth /
export sit the tree on the ground) and to emit a ground mesh for the exported scene. The
field is a pure function of (seed, params) -- deterministic, like everything in Nebula.
"""
from dataclasses import dataclass

import numpy as np

from ..core.determinism import rng_from_key


@dataclass
class Heightfield:
    origin: np.ndarray       # world (x0, y0) of corner
    spacing: float
    heights: np.ndarray      # (Nx, Ny) terrain elevation

    @property
    def shape(self):
        return self.heights.shape

    def height_at(self, x, y):
        """Bilinear terrain height at world (x, y) (scalars or arrays)."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        fx = np.clip((x - self.origin[0]) / self.spacing, 0, self.heights.shape[0] - 1.001)
        fy = np.clip((y - self.origin[1]) / self.spacing, 0, self.heights.shape[1] - 1.001)
        i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
        tx = fx - i0; ty = fy - j0
        h = self.heights
        return ((h[i0, j0] * (1 - tx) + h[i0 + 1, j0] * tx) * (1 - ty)
                + (h[i0, j0 + 1] * (1 - tx) + h[i0 + 1, j0 + 1] * tx) * ty)


def make_terrain(seed=0, size=6.0, n=48, amplitude=0.25, octaves=4):
    """Seeded band-limited value-noise terrain on an n x n grid spanning [-size/2, size/2]^2."""
    spacing = size / (n - 1)
    origin = np.array([-size / 2, -size / 2])
    heights = np.zeros((n, n))
    xs = origin[0] + spacing * np.arange(n)
    ys = origin[1] + spacing * np.arange(n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    amp = amplitude
    for o in range(octaves):
        freq = 2 ** o
        rng = rng_from_key("terrain", seed, o)
        # a small random low-frequency lattice, smoothly upsampled (deterministic)
        m = 4 * freq + 1
        lattice = rng.standard_normal((m, m))
        li = np.clip((gx - origin[0]) / size * (m - 1), 0, m - 1.001)
        lj = np.clip((gy - origin[1]) / size * (m - 1), 0, m - 1.001)
        i0 = np.floor(li).astype(int); j0 = np.floor(lj).astype(int)
        tx = li - i0; ty = lj - j0
        # smoothstep weights for C1 continuity
        sx = tx * tx * (3 - 2 * tx); sy = ty * ty * (3 - 2 * ty)
        val = ((lattice[i0, j0] * (1 - sx) + lattice[i0 + 1, j0] * sx) * (1 - sy)
               + (lattice[i0, j0 + 1] * (1 - sx) + lattice[i0 + 1, j0 + 1] * sx) * sy)
        heights += amp * val
        amp *= 0.5
    heights -= heights.mean()
    return Heightfield(origin, spacing, heights)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    t = make_terrain(seed=1)
    print(f"1) terrain {t.shape}  spacing={t.spacing:.3f}  "
          f"height range [{t.heights.min():.3f}, {t.heights.max():.3f}]")
    # determinism: same seed -> identical terrain
    t2 = make_terrain(seed=1)
    print(f"2) deterministic (same seed identical): {np.array_equal(t.heights, t2.heights)}")
    assert np.array_equal(t.heights, t2.heights)
    # height_at agrees with the grid at a node
    h00 = t.height_at(t.origin[0], t.origin[1])
    print(f"3) height_at corner = {float(h00):.3f} (grid {t.heights[0, 0]:.3f})")
    assert abs(float(h00) - t.heights[0, 0]) < 1e-9
    print("\nheightfield self-checks passed.")
