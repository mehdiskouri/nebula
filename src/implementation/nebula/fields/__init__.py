"""
Nebula implicit field representations (ARCHITECTURE Foundational thesis: "representation is
volumetric/implicit, not mesh-native"). Signed distance fields for solids, heightfields for
terrain; mesh is an export artifact only (geometry.mesh_export).

- sdf         : signed distance field of the grown tree (tapered-capsule union along the
                skeleton) + per-voxel material/layer classification for the restriction operator.
- heightfield : the deterministic terrain the tree is rooted in.
"""
