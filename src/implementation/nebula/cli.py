"""
Nebula Phase-0 command line (ARCHITECTURE Part VIII).

    python -m nebula.cli grow-and-burn --seed 7 --age 22 --out tree.glb

Grows a tree, ignites and burns it on the conserved bus, restricts/refines on the single trust
scalar, fractures the charred branches, and exports a glTF whose colour is derived from the
simulation. Deterministic: the same arguments always produce the same asset (same digest).
"""
import argparse
import os
import sys

from .pipeline.tree_slice import run_slice, SliceConfig


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nebula", description="Nebula Phase-0 tree slice")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gb = sub.add_parser("grow-and-burn", help="grow a tree, burn it, export glTF")
    gb.add_argument("--seed", type=int, default=7)
    gb.add_argument("--age", type=int, default=None, help="growth generations (default: max)")
    gb.add_argument("--out", type=str, default="demo_output/tree.glb", help="output .glb path")
    gb.add_argument("--fire-grid", type=int, default=24, help="fire voxels along the longest axis")
    gb.add_argument("--burn-steps", type=int, default=26)
    gb.add_argument("--dt", type=float, default=0.04)
    gb.add_argument("--quiet", action="store_true")

    dm = sub.add_parser("demo", help="render the Phase-0 slice to MP4 (beauty + mechanism)")
    dm.add_argument("--out", type=str, default="demo_output/nebula_tree", help="output path prefix (writes _beauty/_mechanism.mp4)")
    dm.add_argument("--seed", type=int, default=7)
    dm.add_argument("--fire-grid", type=int, default=24)
    dm.add_argument("--burn-steps", type=int, default=30)
    dm.add_argument("--no-beauty", action="store_true")
    dm.add_argument("--no-mechanism", action="store_true")

    args = parser.parse_args(argv)
    if getattr(args, "out", None):
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
    if args.cmd == "demo":
        from .pipeline import demo
        dargv = ["--out", args.out, "--seed", str(args.seed),
                 "--fire-grid", str(args.fire_grid), "--burn-steps", str(args.burn_steps)]
        if args.no_beauty:
            dargv.append("--no-beauty")
        if args.no_mechanism:
            dargv.append("--no-mechanism")
        demo.main(dargv)
        return 0
    if args.cmd == "grow-and-burn":
        cfg = SliceConfig(seed=args.seed, age=args.age, fire_grid=args.fire_grid,
                          burn_steps=args.burn_steps, dt=args.dt)
        res = run_slice(cfg, out_path=args.out, verbose=not args.quiet)
        print(f"\nwrote {res.scene_path}  (digest {res.digest:032x})")
        print(f"  fuel consumed {res.fuel_consumed*100:.1f}% | conservation audit {res.audit_max:.1e} | "
              f"refined cells {res.refine_flagged} | fractured constraints {res.xpbd_fractured} | "
              f"mesh {res.mesh_verts} verts")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
