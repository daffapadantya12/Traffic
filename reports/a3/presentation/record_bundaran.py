"""Record the uncalibrated Bundaran HI demo's viewer frames, for the replay baked into the presentation.

The demo lives on the branch claude/bundaran-hi-demo (scripts/bundaran_hi_3d.py and data/bundaran_hi_demo).  This
runs its simulation loop without the web server and writes one JSON line per 0.5 s step.  --demo-root is a
checkout or export of that branch, for example:

    git archive claude/bundaran-hi-demo scripts data/bundaran_hi_demo | tar -x -C <dir>
    python3 reports/a3/presentation/record_bundaran.py --demo-root <dir> --work <out dir>

The trip and vehicle-type files it rewrites are inside --demo-root, not in this repository.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--start-hour", type=float, default=7.0)
    ap.add_argument("--warmup-min", type=float, default=10.0)
    ap.add_argument("--minutes", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--radius", type=float, default=700.0)
    args = ap.parse_args()

    root = os.path.abspath(args.demo_root)
    os.chdir(root)
    sys.path.insert(0, os.path.join(root, "scripts"))
    import sumolib
    import bundaran_hi_3d as b3d
    demo = b3d.demo

    with open(b3d.GEOMETRY) as f:
        geo = json.load(f)
    with open(b3d.LAYOUT) as f:
        layout = json.load(f)
    vtypes, routes = os.path.join(b3d.VIEWER, "vtypes.add.xml"), os.path.join(b3d.VIEWER, "live_flows.rou.xml")
    demo.write_vtypes(vtypes)
    table = demo.write_flows(layout, routes, args.scale, 86400.0)
    edge_leg = b3d.approach_edges(sumolib.net.readNet(b3d.NET, withInternal=False), layout)
    sim = b3d.Sim(routes, vtypes, args.start_hour, 1.0, args.seed, 60.0 * args.warmup_min, geo, edge_leg)   # never started as a thread

    import libsumo as ts
    ts.start(["sumo", "-n", b3d.NET, "-r", routes, "-a", vtypes, "--begin", "0", "--step-length", "0.5", "--lateral-resolution", "0.8",
              "--collision.action", "warn", "--collision.check-junctions", "true", "--collision.mingap-factor", "0", "--time-to-teleport", "300",
              "--no-step-log", "true", "--no-warnings", "true", "--ignore-route-errors", "true", "--seed", str(args.seed)])   # as bundaran_hi_3d.Sim.run
    os.makedirs(args.work, exist_ok=True)
    out = os.path.join(args.work, "rec_bundaran_hi.jsonl.gz")
    r2, total_s, n = args.radius ** 2, 60.0 * (args.warmup_min + args.minutes), 0
    with gzip.open(out, "wt", compresslevel=5) as fh:
        while sim.t < total_s:
            sim.step(ts)
            if sim.t < 60.0 * args.warmup_min:
                continue
            f = sim.frame(ts)
            near = [r for r in f["vehicles"] if r[1] * r[1] + r[2] * r[2] <= r2]
            fh.write(json.dumps(dict(t=f["t"], clock=f["clock"], tls=f["tls"], entries=f["entries"],
                                     legs={a: [s["n"], s["q_veh"]] for a, s in f["legs"].items()}, veh=near,
                                     n_all=len(f["vehicles"])), separators=(",", ":")) + "\n")
            n += 1
    ts.close()
    shutil.copy(b3d.GEOMETRY, os.path.join(args.work, "bundaran_hi_geometry.json"))
    print(f"bundaran_hi: {sum(table.values())} veh/h assumed (scale {args.scale}); wrote {n} frames to {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
