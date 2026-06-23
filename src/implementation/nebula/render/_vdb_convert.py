"""
Dense-grid → OpenVDB converter (run with the `vdb` conda env's python, NOT the main venv).

The main pipeline (py3.13 venv) writes the fire fields as a dense `.npz`; this helper, run under
the conda env that has the OpenVDB python bindings, converts them to a sparse `.vdb` with named
grids (`density` = soot, `temperature` = T in Kelvin) that Omniverse / a path tracer reads as a
blackbody-emissive, soot-absorbing volume. Usage:

    /opt/miniforge3/envs/vdb/bin/python _vdb_convert.py fields.npz out.vdb
"""
import sys

import numpy as np

try:
    import pyopenvdb as vdb
except ImportError:                      # some builds expose it as `openvdb`
    import openvdb as vdb


def grids_to_vdb(npz_path, out_path):
    d = np.load(npz_path)
    vs = float(d["voxel_size"])
    origin = d["origin"].astype(float) if "origin" in d else np.zeros(3)
    grids = []
    for name in ("density", "temperature"):
        if name not in d:
            continue
        arr = np.ascontiguousarray(np.asarray(d[name], np.float32))
        g = vdb.FloatGrid()
        g.copyFromArray(arr)
        g.name = name
        g.gridClass = vdb.GridClass.FOG_VOLUME if name == "density" else vdb.GridClass.UNKNOWN
        g.transform = vdb.createLinearTransform(voxelSize=vs)
        # place the grid at the world origin of the fire box
        g.transform.postTranslate(tuple(origin))
        grids.append(g)
    vdb.write(out_path, grids=grids)
    return [g.name for g in grids]


if __name__ == "__main__":
    names = grids_to_vdb(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2], "grids:", names)
