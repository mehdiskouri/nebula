"""
The typed hypergraph substrate (ARCHITECTURE Part II; Decisions #4, #5).

The single data structure underneath everything. NODES carry fast-changing per-element
STATE (position, velocity, stress, temperature, ...): small, per-element, mutable each
step. Typed HYPEREDGES carry shared, lawful structure (constitutive models, constants,
governing equations, interaction rules). An n-ary edge is essential because physical
coupling is rarely binary; law is shared and amortized (ten thousand sapwood nodes
reference one sapwood hyperedge).

The same object serves three roles at once: storage format, coupling topology, and the
solver's constraint graph (a hyperedge IS an XPBD constraint).

A naive hypergraph is GPU-hostile; the **categorize -> color -> flatten** discipline
rescues it (Decision #5): group edges by type, graph-color the independent edges within a
type so they run in parallel without write conflicts (race-free -- the same guarantee as
the conserved-bus reduce), and flatten each color into contiguous arrays for kernel launch.

Phase-0 edge taxonomy (the subset the tree slice needs):
  material  -- a node set + physical constants / constitutive model
  layer     -- a node set + material tag + deposition history (bark/sapwood/heartwood, rings)
  constraint-- nodes + a positional/energetic constraint (XPBD distance/volume/contact)
  lawdomain -- a region of nodes + governing equations/fields (the fire field; gravity)
  front     -- a generative locus (growth sensing + decision + deposition rule)
  interface -- a fine/coarse seam + the hanging-node constraint binding fine to coarse parent
(Regulator / Transient-contact are later arch phases.)

Pure numpy here (the host-side substrate). The flattened color arrays are what the Warp /
Taichi kernels launch over.
"""
from dataclasses import dataclass, field as _field
from typing import Any, Optional

import numpy as np

from . import determinism as det

# the typed-hyperedge taxonomy tags (ARCHITECTURE Part II table)
EDGE_TYPES = ("material", "layer", "constraint", "lawdomain", "front", "interface")


class Nodes:
    """Per-node STATE in struct-of-arrays form (GPU-coalesced, the size+speed win).

    Each field is a contiguous array whose first axis is the node index. Every node also
    carries a stable integer `key` (a hashed lineage identity) so its RNG stream and any
    memoization are a pure function of identity, not draw order (V0.5/V1.8).
    """

    def __init__(self, count: int, keys=None):
        self.n = int(count)
        self.fields: dict[str, np.ndarray] = {}
        if keys is None:
            self.keys = np.arange(self.n, dtype=np.int64)
        else:
            keys = np.asarray(keys, dtype=np.int64)
            assert keys.shape[0] == self.n, "keys length must match node count"
            self.keys = keys

    def add_field(self, name: str, init=0.0, shape=(), dtype=np.float64):
        """Declare a per-node field, shape (n, *shape), filled with `init`."""
        arr = np.full((self.n,) + tuple(shape), init, dtype=dtype)
        self.fields[name] = arr
        return arr

    def set_field(self, name: str, arr):
        arr = np.asarray(arr)
        assert arr.shape[0] == self.n, f"field {name} first axis must be n={self.n}"
        self.fields[name] = arr
        return arr

    def __getitem__(self, name):
        return self.fields[name]

    def __contains__(self, name):
        return name in self.fields

    def copy(self):
        nd = Nodes(self.n, self.keys.copy())
        nd.fields = {k: v.copy() for k, v in self.fields.items()}
        return nd


@dataclass
class Hyperedge:
    """A typed n-ary edge: a category tag, its member node indices, and its shared law.

    members : int node indices into the Nodes SoA (empty for a grid-backed law-domain whose
              state lives in a dense payload rather than the flat cloud).
    law     : the shared payload -- constitutive params, a Domain object, constraint data, etc.
    """
    etype: str
    members: np.ndarray
    law: Any = None
    meta: dict = _field(default_factory=dict)

    def __post_init__(self):
        if self.etype not in EDGE_TYPES:
            raise ValueError(f"unknown edge type {self.etype!r}; expected one of {EDGE_TYPES}")
        self.members = np.asarray(self.members, dtype=np.int64).ravel()


@dataclass
class FlatColor:
    """A flattened color group: CSR member layout for a race-free parallel kernel launch.

    All edges in one color are node-disjoint, so a kernel may process them concurrently
    with no write conflicts. `offsets` is length (n_edges+1); edge e's members are
    `members[offsets[e]:offsets[e+1]]`; `edge_ids` maps back to the original edge index.
    """
    offsets: np.ndarray
    members: np.ndarray
    edge_ids: np.ndarray

    @property
    def n_edges(self):
        return len(self.offsets) - 1


class Hypergraph:
    """Nodes + typed hyperedges, with the categorize -> color -> flatten discipline."""

    def __init__(self, nodes: Optional[Nodes] = None):
        self.nodes = nodes if nodes is not None else Nodes(0)
        self.edges: list[Hyperedge] = []

    # ---- construction ----
    def add_edge(self, etype, members, law=None, **meta) -> int:
        e = Hyperedge(etype, members, law, dict(meta))
        self.edges.append(e)
        return len(self.edges) - 1

    # ---- categorize ----
    def categorize(self) -> dict:
        """Group edge indices by type tag (uniform per-category processing, Decision #5)."""
        cats: dict[str, list[int]] = {t: [] for t in EDGE_TYPES}
        for i, e in enumerate(self.edges):
            cats[e.etype].append(i)
        return {t: idx for t, idx in cats.items() if idx}

    # ---- color ----
    def color(self, etype) -> list[list[int]]:
        """Greedy proper coloring of one category's edges by node-sharing conflict.

        Two edges conflict iff they share a node; same-color edges are therefore
        node-disjoint and safe to process in parallel (the race-free guarantee). Edges are
        visited in ascending index order and assigned the smallest admissible color, so the
        coloring is deterministic (a determinism requirement, not just a perf nicety).
        """
        idxs = [i for i, e in enumerate(self.edges) if e.etype == etype]
        node_colors: dict[int, set] = {}          # node -> set of colors already incident
        colors: dict[int, int] = {}               # edge idx -> color
        buckets: list[list[int]] = []
        for ei in idxs:
            members = self.edges[ei].members
            forbidden = set()
            for m in members:
                if m in node_colors:
                    forbidden |= node_colors[m]
            c = 0
            while c in forbidden:
                c += 1
            colors[ei] = c
            for m in members:
                node_colors.setdefault(int(m), set()).add(c)
            if c >= len(buckets):
                buckets.append([])
            buckets[c].append(ei)
        return buckets

    # ---- flatten ----
    def flatten(self, edge_ids) -> FlatColor:
        """Pack a list of edge indices into CSR member arrays for a kernel launch."""
        edge_ids = list(edge_ids)
        lengths = [len(self.edges[e].members) for e in edge_ids]
        offsets = np.zeros(len(edge_ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)
        members = (np.concatenate([self.edges[e].members for e in edge_ids])
                   if edge_ids else np.zeros(0, dtype=np.int64))
        return FlatColor(offsets, members.astype(np.int64), np.asarray(edge_ids, dtype=np.int64))

    def colored_flat(self, etype) -> list[FlatColor]:
        """categorize -> color -> flatten for one category: the launch-ready color groups."""
        return [self.flatten(b) for b in self.color(etype)]

    # ---- diagnostics ----
    def stats(self):
        cats = self.categorize()
        return {t: len(v) for t, v in cats.items()}


def conflict_free(hg: Hypergraph, etype) -> bool:
    """Verify every color group is node-disjoint (the coloring invariant)."""
    for fc in hg.colored_flat(etype):
        for e in range(fc.n_edges):
            seg = fc.members[fc.offsets[e]:fc.offsets[e + 1]]
            # within a color, no node may appear in two different edges
        seen = {}
        for e in range(fc.n_edges):
            for m in fc.members[fc.offsets[e]:fc.offsets[e + 1]]:
                if m in seen and seen[m] != e:
                    return False
                seen[int(m)] = e
    return True


if __name__ == "__main__":
    # Build a small cloud and a tangle of overlapping distance constraints; verify the
    # categorize->color->flatten discipline produces race-free color groups.
    rng = np.random.default_rng(0)
    n = 200
    nd = Nodes(n, keys=np.array([det.stable_hash("node", i) for i in range(n)]))
    nd.add_field("x", shape=(3,))
    nd.fields["x"][:] = rng.random((n, 3))

    hg = Hypergraph(nd)
    # a chain of distance constraints + some random extra edges (lots of node sharing)
    for i in range(n - 1):
        hg.add_edge("constraint", [i, i + 1], law={"rest": 1.0})
    for _ in range(150):
        a, b = rng.integers(0, n, 2)
        if a != b:
            hg.add_edge("constraint", [a, b], law={"rest": 1.0})
    # a couple of n-ary law-domain + front edges
    hg.add_edge("lawdomain", np.arange(n), law={"gravity": -9.81})
    hg.add_edge("front", [0], law={"kind": "apical"})

    print("1) category counts:", hg.stats())
    buckets = hg.color("constraint")
    total = sum(len(b) for b in buckets)
    n_constraints = hg.stats()["constraint"]
    print(f"2) colored {n_constraints} constraints into {len(buckets)} colors "
          f"(sizes {[len(b) for b in buckets][:8]}{'...' if len(buckets) > 8 else ''})")
    assert total == n_constraints, "every edge must get exactly one color"
    print(f"3) all color groups node-disjoint (race-free): {conflict_free(hg, 'constraint')}")
    assert conflict_free(hg, "constraint")

    flats = hg.colored_flat("constraint")
    print(f"4) flattened: {len(flats)} FlatColor groups; "
          f"first has {flats[0].n_edges} edges, {len(flats[0].members)} member slots")
    print("\nhypergraph substrate self-checks passed.")
