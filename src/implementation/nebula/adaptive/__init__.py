"""
Coarse-to-fine adaptive resolution (ARCHITECTURE §III.2; Decisions #8, #9, #10).
Verified by V0.4 (O(n_active . log n_total) scaling, graceful degradation).

- octree     : the Morton-linearized octree -- "one tree, three hats" (LOD + multigrid +
               Barnes-Hut far-field), with stackless DFS-escape traversal. (CPU)
- octree_gpu : the same DFS-linearized arrays traversed on the GPU (Warp), identical logic.
- refine     : the coarse-to-fine PREDICATE the verification did not build -- D = max(proximity,
               proxy-error=lod_trust, rate=eps, pin) + hysteresis + 2:1 balance + the Interface
               (hanging-node) hyperedge, so the LOD seam is "just another constraint category".
"""
