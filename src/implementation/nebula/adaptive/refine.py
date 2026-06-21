"""
The coarse-to-fine refinement predicate (ARCHITECTURE §III.2; Decision #10).

This is the piece the verification did NOT build (V0.4 measured the static octree's scaling;
the adaptive refine/coarsen policy is new). It refines exactly where the coarse proxy's error
exceeds tolerance and nowhere else, with the architecture's NON-NEGOTIABLE stability guardrails:

  refinement predicate  D = max over normalized criteria:
     - proximity   : a contact / ignition source within a margin sized to the contact scale
     - proxy-error : the restriction trust scalar lod_trust (gap folded with connectivity, V2.3)
     - rate        : the Jensen variance epsilon (|d state/dt| analogue; sets a finer step too, V1.3)
     - pin         : an authored override
  hysteresis        : split at T_hi; merge ONLY after D < T_lo for tau quiet steps (no thrash/popping)
  2:1 balance       : adjacent cells differ by <= 1 level
  Interface edge    : the hanging-node constraint binding fine cells to their coarse parent --
                      "the LOD seam is just another constraint category" (reuses the hypergraph).

Criteria are normalized by their own thresholds so D is dimensionless: D >= T_hi (=1) -> refine,
D < T_lo (=0.5) sustained -> coarsen. Operates on a per-cell level field (the graded-octree leaf
levels -- the LOD / multigrid hat of the one octree). Pure numpy + scipy.ndimage.
"""
from dataclasses import dataclass, field as _field

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..core.hypergraph import Hypergraph, Nodes


@dataclass
class RefineParams:
    max_level: int = 3
    T_hi: float = 1.0          # split when normalized D >= T_hi
    T_lo: float = 0.5          # merge when D < T_lo ...
    tau: int = 3               # ... sustained for tau quiet steps (hysteresis)
    proxy_thresh: float = 0.30  # lod_trust threshold (V0.1/V0.2 T_hi=0.30) -> normalizer
    eps_thresh: float = 0.5     # Jensen eps* (V1.3) -> normalizer
    contact_margin: float = 2.0  # proximity margin in cells


def compute_D(trust=None, eps=None, proximity=None, pin=None, rp=RefineParams()):
    """The refinement predicate D = max over normalized criteria (dimensionless; >=T_hi -> refine).

    trust: lod_trust per cell (proxy-error); eps: Jensen variance (rate); proximity: in [0,1]
    (1 at a contact/ignition source); pin: in [0,1] authored override. Any may be None.
    """
    crits = []
    shape = None
    for arr in (trust, eps, proximity, pin):
        if arr is not None:
            shape = np.asarray(arr).shape
            break
    if shape is None:
        raise ValueError("compute_D needs at least one criterion")
    if trust is not None:
        crits.append(np.asarray(trust, float) / rp.proxy_thresh)
    if eps is not None:
        crits.append(np.asarray(eps, float) / rp.eps_thresh)
    if proximity is not None:
        crits.append(np.asarray(proximity, float))
    if pin is not None:
        crits.append(np.asarray(pin, float))
    return np.max(np.stack(crits, axis=0), axis=0)


def proximity_field(shape, source_mask, spacing=1.0, margin=2.0):
    """Normalized proximity in [0,1]: 1 at/inside a source, decaying to 0 beyond `margin` cells."""
    source_mask = np.asarray(source_mask, bool)
    if not source_mask.any():
        return np.zeros(shape)
    dist = distance_transform_edt(~source_mask) * spacing
    return np.clip((margin * spacing - dist) / (margin * spacing), 0.0, 1.0)


@dataclass
class AdaptiveGrid:
    """A per-cell refinement level field with hysteresis + 2:1 balance (the graded octree leaves)."""
    level: np.ndarray            # (nx,ny,nz) int leaf level
    params: RefineParams = _field(default_factory=RefineParams)
    quiet: np.ndarray = None     # consecutive steps with D < T_lo (hysteresis counter)

    def __post_init__(self):
        self.level = np.asarray(self.level, np.int64)
        if self.quiet is None:
            self.quiet = np.zeros_like(self.level)

    @classmethod
    def coarse(cls, shape, params=None):
        return cls(np.zeros(shape, np.int64), params or RefineParams())

    def step(self, D):
        """One refine/coarsen step given the predicate field D. Returns the interface seam list.

        Split immediately where D >= T_hi (hysteresis up); coarsen only where D < T_lo has held
        for tau steps (hysteresis down); then enforce 2:1 balance. Mutates self.level/self.quiet.
        """
        rp = self.params
        D = np.asarray(D, float)
        self.quiet = np.where(D < rp.T_lo, self.quiet + 1, 0)
        new = self.level.copy()
        split = (D >= rp.T_hi) & (self.level < rp.max_level)
        new[split] = self.level[split] + 1
        self.quiet[split] = 0
        merge = (self.quiet >= rp.tau) & (self.level > 0) & (D < rp.T_lo)
        new[merge] = self.level[merge] - 1
        self.quiet[merge] = 0
        self.level = new
        self.balance()
        return self.interfaces()

    def balance(self):
        """Enforce the 2:1 condition: no face-adjacent cells differ by more than one level."""
        L = self.level
        changed = True
        while changed:
            changed = False
            nbmax = np.full_like(L, -1)
            for ax in range(3):
                if L.shape[ax] < 2:
                    continue
                lo = [slice(None)] * 3; hi = [slice(None)] * 3
                lo[ax] = slice(0, -1); hi[ax] = slice(1, None)
                nbmax[tuple(lo)] = np.maximum(nbmax[tuple(lo)], L[tuple(hi)])   # +neighbor
                nbmax[tuple(hi)] = np.maximum(nbmax[tuple(hi)], L[tuple(lo)])   # -neighbor
            need = nbmax - 1
            viol = need > L
            if viol.any():
                L = np.where(viol, need, L)
                changed = True
        self.level = L

    def interfaces(self):
        """Face-adjacent cell pairs at different levels -> hanging-node seams.

        Returns a list of (cellA_index, cellB_index, levelA, levelB) with flat cell indices
        (the coarse cell first). These become Interface hyperedges (interface_hyperedges)."""
        seams = []
        L = self.level
        flat = np.arange(L.size).reshape(L.shape)
        for ax in range(3):
            if L.shape[ax] < 2:
                continue
            lo = [slice(None)] * 3; hi = [slice(None)] * 3
            lo[ax] = slice(0, -1); hi[ax] = slice(1, None)
            La = L[tuple(lo)]; Lb = L[tuple(hi)]
            diff = La != Lb
            ia = flat[tuple(lo)][diff]; ib = flat[tuple(hi)][diff]
            la = La[diff]; lb = Lb[diff]
            for a, b, x, y in zip(ia, ib, la, lb):
                coarse, fine = (int(a), int(b)) if x < y else (int(b), int(a))
                seams.append((coarse, fine, int(min(x, y)), int(max(x, y))))
        return seams

    def stats(self):
        return {int(l): int((self.level == l).sum()) for l in np.unique(self.level)}


def interface_hyperedges(hg: Hypergraph, seams):
    """Add each LOD seam as an Interface hyperedge (the hanging-node constraint category).

    Reuses the typed-hypergraph substrate -- the seam is "just another constraint category",
    the same hanging-node fix that appears in physics, LOD, and geometry stitching.
    """
    for coarse, fine, lc, lf in seams:
        hg.add_edge("interface", [coarse, fine], law={"coarse_level": lc, "fine_level": lf})
    return hg


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    shape = (16, 16, 16)
    rp = RefineParams(max_level=3, tau=3)

    # 1) refine where the trust scalar is high (a char-wedge corner), coarse elsewhere.
    trust = np.zeros(shape)
    trust[:4, :4, :4] = 2.0                      # high lod_trust corner -> refine
    eps = np.zeros(shape)
    ag = AdaptiveGrid.coarse(shape, rp)
    seams = ag.step(compute_D(trust=trust, eps=eps, rp=rp))
    print(f"1) after one step: level histogram {ag.stats()}  interface seams={len(seams)}")
    assert ag.level.max() == 1 and ag.level[0, 0, 0] == 1 and ag.level[-1, -1, -1] == 0

    # drive it to max level over several steps (the corner keeps exceeding T_hi).
    for _ in range(4):
        ag.step(compute_D(trust=trust, eps=eps, rp=rp))
    print(f"   after 5 steps: level histogram {ag.stats()} (corner -> max_level {rp.max_level})")
    assert ag.level[0, 0, 0] == rp.max_level

    # 2) 2:1 balance: adjacent levels never differ by more than 1.
    maxjump = 0
    L = ag.level
    for axis in range(3):
        d = np.abs(np.diff(L, axis=axis))
        maxjump = max(maxjump, int(d.max()))
    print(f"2) max adjacent level jump = {maxjump} (2:1 balance => <= 1)")
    assert maxjump <= 1

    # 3) hysteresis: drop D to ~0; the corner must NOT merge until tau quiet steps elapse.
    Dlow = np.zeros(shape)
    levels_over_time = []
    for s in range(rp.tau + 2):
        ag.step(Dlow)
        levels_over_time.append(int(ag.level[0, 0, 0]))
    print(f"3) corner level after each quiet step: {levels_over_time} (no merge before tau={rp.tau})")
    assert levels_over_time[0] == rp.max_level, "merged too early (hysteresis violated)"
    assert levels_over_time[-1] < rp.max_level, "never merged after sustained quiet"

    # 4) interface hyperedges: the seam is just another constraint category.
    ag2 = AdaptiveGrid.coarse(shape, rp)
    seams = ag2.step(compute_D(trust=trust, rp=rp))
    hg = Hypergraph(Nodes(shape[0] * shape[1] * shape[2]))
    interface_hyperedges(hg, seams)
    print(f"4) interface hyperedges added: {hg.stats().get('interface', 0)} (hanging-node seams)")
    assert hg.stats().get("interface", 0) == len(seams) > 0
    print("\nrefinement predicate self-checks passed.")
