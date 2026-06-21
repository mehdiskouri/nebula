"""
Nebula geometry export (ARCHITECTURE Foundational thesis: "mesh is an export artifact only";
Decision #2: mesh only at export). The ONLY place a mesh appears -- everything upstream is
implicit (SDF) / hypergraph. Marching cubes on the tree SDF -> a triangle mesh whose vertex
colours are DERIVED from the simulation (char chi, temperature T, layer), never authored.

- mesh_export : marching cubes -> trimesh -> glTF/.glb, with simulation-derived vertex colour.
"""
