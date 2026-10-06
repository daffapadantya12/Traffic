"""Build the SUMO network of Perempatan Fatmawati from OpenStreetMap (DECISIONS §13.1).

Traffic-engineering idea: the real place is a divided arterial grid.  The study junction is
Jl. RS Fatmawati x Jl. TB Simatupang / Jl. RA Kartini; the JORR toll road runs above it on a
viaduct in the median, so the at-grade junction is two crossings 51 m apart (westbound traffic
on the south side, eastbound on the north side) run by one signal controller.  Left turns use
free-left slip lanes; right turns cross the median segment and are signalled again.

Steps (python3 -m traffic_twin.fatmawati.build_net):
1. download the roads within `radius_km` of the junction from the Overpass API (cached in
   data/fatmawati/osm/; (c) OpenStreetMap contributors, ODbL);
2. netconvert (left-hand traffic, main road classes motorway..tertiary, signals guessed from OSM,
   amber 3 s, all-red 2 s) -> net/fatmawati_raw.net.xml;
3. second pass: both Fatmawati crossings become one controller "fatmawati"; approaches with >= 2
   lanes at that controller get a motorcycle box (cars stop rhk_depth_m behind the motorcycle stop
   line, SE PUPR 52/SE/M/2015; SUMO stopOffset with exceptions="motorcycle"); the RA Kartini right
   turn is removed (none observed: Sumabrata et al. 2026);
4. the controller's program becomes Dishub's 4-stage plan (Sumabrata et al. 2026, Dishub DKI static plan):
   S1 RA Kartini (W) through + TB Simatupang (E) through (different carriageways, no conflict),
   S2 E through + E right, S3 N through + right, S4 S through + right; greens 80/12/78/45 s,
   amber 2 s + all-red 2 s.  Program "dishub" holds the four stage greens and, for every ordered
   pair of stages, an amber and an all-red phase: links green in both stages stay green, links that
   end get amber then red, and median links (vehicles between the two crossings) keep green through
   the amber and get amber during the all-red.  Phases 0-11 are the Dishub cycle S1-S2-S3-S4.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data", "fatmawati")

JUNCTION_LAT, JUNCTION_LON = -6.29250, 106.79522          # box centre (signal nodes), not the map pin
CROSSINGS = ("cluster_353175558_356234398", "cluster_356228989_356228993")   # south, north crossing
TLS_ID = "fatmawati"
ROAD_TYPES = ("motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
              "secondary", "secondary_link", "tertiary", "tertiary_link")
# approach edges of the controller (raw-network ids) by the arm the traffic comes FROM
APPROACHES = {"N": "953628421#0", "E": "1069005557#0", "S": "1069005556#0", "W": "1069005554#0"}
MEDIAN = {"south": "638194767", "north": "759911875"}   # southbound / northbound median segments
EXITS = {"N": "759911874#0", "E": "698418844", "S": "531632715#0", "W": "1125239069#0"}   # first edge leaving the junction
YELLOW_S, ALLRED_S = 2.0, 2.0                              # Dishub plan: amber 2 s, all-red 2 s
STAGES = ("S1", "S2", "S3", "S4")
DISHUB_GREEN_S = {"S1": 80.0, "S2": 12.0, "S3": 78.0, "S4": 45.0}   # cycle 231 s (the paper prints 253 s)
# (from edge, to edge or "*") pairs whose links are green in each stage
STAGE_LINKS = {
    "S1": [(APPROACHES["W"], "*"), (APPROACHES["E"], EXITS["W"])],
    "S2": [(APPROACHES["E"], "*"), (MEDIAN["north"], EXITS["N"])],
    "S3": [(APPROACHES["N"], "*"), (MEDIAN["south"], "*")],
    "S4": [(APPROACHES["S"], "*"), (MEDIAN["north"], "*")],
}
STAGE_APPROACHES = {"S1": ("W", "E"), "S2": ("E",), "S3": ("N",), "S4": ("S",)}
STAGE_OF_APPROACH = {"W": "S1", "E": "S2", "N": "S3", "S": "S4"}   # the stage that serves all of an approach's signalised traffic
STAGE_LABEL = {"S1": "RA Kartini + TB Simatupang through", "S2": "TB Simatupang through + right", "S3": "RS Fatmawati from the north",
               "S4": "RS Fatmawati from the south"}
RHK_DEPTH_M = 8.0


def sumo_home() -> str:
    import sumo
    return sumo.SUMO_HOME


def bbox(radius_km: float):
    dlat = radius_km / 110.6
    dlon = radius_km / (111.32 * math.cos(math.radians(abs(JUNCTION_LAT))))
    return JUNCTION_LAT - dlat, JUNCTION_LON - dlon, JUNCTION_LAT + dlat, JUNCTION_LON + dlon


def download_osm(radius_km: float, path: str, force: bool = False) -> str:
    """Roads (+ their nodes) and turn restrictions inside the box, OSM XML for netconvert."""
    if os.path.exists(path) and not force:
        return path
    s, w, n, e = bbox(radius_km)
    b = f"{s:.5f},{w:.5f},{n:.5f},{e:.5f}"
    q = (f'[out:xml][timeout:300];(way["highway"]({b});>;);out meta;'
         f'(relation["type"="restriction"]({b}););out meta;')
    req = urllib.request.Request("https://overpass-api.de/api/interpreter",
                                 data=urllib.parse.urlencode({"data": q}).encode(),
                                 headers={"User-Agent": "traffic_twin/3.0 (ITI5145 student project)"})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(req, timeout=400) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def netconvert(args: list) -> None:
    exe = os.path.join(sumo_home(), "bin", "netconvert")
    env = dict(os.environ, SUMO_HOME=sumo_home())
    p = subprocess.run([exe] + args, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])


def build_raw(osm_path: str, out: str, patches_dir: str) -> None:
    typemaps = ",".join([os.path.join(sumo_home(), "data", "typemap", "osmNetconvert.typ.xml"),
                         os.path.join(patches_dir, "jakarta.typ.xml")])
    netconvert([
        "--osm-files", osm_path, "-o", out, "--type-files", typemaps,
        "--lefthand", "--check-lane-foes.all", "--geometry.remove", "--ramps.guess", "--junctions.join", "--junctions.corner-detail", "5",
        "--tls.guess-signals", "--tls.discard-simple", "--tls.join", "--tls.default-type", "static",
        "--tls.yellow.time", str(int(YELLOW_S)), "--tls.allred.time", str(int(ALLRED_S)),
        "--keep-edges.by-type", ",".join("highway." + t for t in ROAD_TYPES),
        "--remove-edges.isolated", "--osm.turn-lanes", "--osm.lane-access", "--osm.layer-elevation", "6",
        "--output.street-names", "--output.original-names", "--proj.utm", "--no-turnarounds",
        "--default.lanewidth", "3.25", "--ignore-widths"])   # OSM width tags split unevenly (1.5 m lanes on TB Simatupang)
    # --no-turnarounds: U-turns only where OSM maps them (median openings drawn as link ways, e.g. the loops
    # either side of Fatmawati).  netconvert would otherwise add one at every junction (203 here), and those
    # U-turns merging into a right turn were where most junction overlaps happened.


def patch_controller(raw: str, out: str, patches_dir: str, rhk_depth_m: float) -> None:
    """Second pass: one controller for both crossings, motorcycle boxes on its approaches."""
    import sumolib
    net = sumolib.net.readNet(raw, withInternal=False)
    os.makedirs(patches_dir, exist_ok=True)
    nod = os.path.join(patches_dir, "fatmawati_tls.nod.xml")
    with open(nod, "w") as f:
        f.write('<!-- Perempatan Fatmawati: both at-grade crossings either side of the JORR toll viaduct are run by one\n'
                '     signal controller; OSM tags its signals on the approach ways, which netconvert does not attach. -->\n<nodes>\n')
        for nid in CROSSINGS:
            x, y = net.getNode(nid).getCoord()
            f.write(f'    <node id="{nid}" x="{x:.2f}" y="{y:.2f}" type="traffic_light" tl="{TLS_ID}"/>\n')
        f.write('</nodes>\n')
    edg = os.path.join(patches_dir, "fatmawati_rhk.edg.xml")
    with open(edg, "w") as f:
        f.write('<!-- Motorcycle boxes (Ruang Henti Khusus, SE PUPR 52/SE/M/2015): on approaches with >= 2 lanes cars stop\n'
                f'     {rhk_depth_m:g} m behind the stop line; motorcycles may use the box in front of them. -->\n<edges>\n')
        for arm, eid in APPROACHES.items():
            e = net.getEdge(eid)
            if e.getLaneNumber() >= 2:
                f.write(f'    <edge id="{eid}">\n        <stopOffset value="{rhk_depth_m:g}" exceptions="motorcycle"/>\n    </edge>\n')
        f.write('</edges>\n')
    narrow = os.path.join(patches_dir, "narrow_lanes.edg.xml")
    with open(narrow, "w") as f:
        f.write('<!-- Two-way streets tagged lanes=1 get one half-width lane per direction from the OSM import;\n'
                '     a car (1.8 m) does not fit in 1.6 m, so these are widened to one 3 m lane each way. -->\n<edges>\n')
        for e in net.getEdges():
            if e.getLaneNumber() == 1 and e.getLanes()[0].getWidth() < 2.5:
                f.write(f'    <edge id="{e.getID()}"><lane index="0" width="3.00"/></edge>\n')
        f.write('</edges>\n')
    con = os.path.join(patches_dir, "fatmawati_turns.con.xml")
    with open(con, "w") as f:
        f.write('<!-- Movements not used at Perempatan Fatmawati (none in the observed counts, Sumabrata et al. 2026):\n'
                '     the RA Kartini right turn and U-turns between the two median segments. -->\n<connections>\n')
        f.write(f'    <delete from="{APPROACHES["W"]}" to="{MEDIAN["south"]}"/>\n')
        f.write(f'    <delete from="{MEDIAN["south"]}" to="{MEDIAN["north"]}"/>\n')
        f.write(f'    <delete from="{MEDIAN["north"]}" to="{MEDIAN["south"]}"/>\n</connections>\n')
    netconvert(["--sumo-net-file", raw, "--node-files", nod, "--edge-files", f"{edg},{narrow}", "--connection-files", con, "-o", out,
                "--lefthand", "--check-lane-foes.all", "--tls.default-type", "static", "--tls.yellow.time", str(int(YELLOW_S)),
                "--tls.allred.time", str(int(ALLRED_S)), "--no-turnarounds"])


def stage_link_sets(net_path: str) -> tuple:
    """({stage: set of link indices green in it}, set of median link indices, number of links)."""
    import sumolib
    net = sumolib.net.readNet(net_path, withInternal=False, withPrograms=True)
    links = net.getTLS(TLS_ID).getConnections()
    n = 1 + max(idx for _, _, idx in links)
    green = {st: set() for st in STAGES}
    median = set()
    for inl, outl, idx in links:
        fe, te = inl.getEdge().getID(), outl.getEdge().getID()
        if fe in MEDIAN.values():
            median.add(idx)
        for st, pairs in STAGE_LINKS.items():
            if any(fe == f and (t == "*" or te == t) for f, t in pairs):
                green[st].add(idx)
    unserved = set(range(n)) - set().union(*green.values())
    if unserved:
        raise ValueError(f"links {sorted(unserved)} of {TLS_ID} are green in no stage")
    return green, median, n


def stage_program(net_path: str) -> list:
    """Phase list [(duration, state, name)] of program "dishub" (see the module docstring)."""
    green, median, n = stage_link_sets(net_path)

    def state(fn):
        return "".join(fn(i) for i in range(n))

    def amber(a, b):
        return state(lambda i: ("G" if i in green[a] else "r") if i in median else
                     ("G" if i in green[a] and i in green[b] else "y" if i in green[a] else "r"))

    def allred(a, b):
        return state(lambda i: ("G" if i in green[a] and i in green[b] else "y" if i in green[a] else "r") if i in median else
                     ("G" if i in green[a] and i in green[b] else "r"))

    order = list(STAGES)
    pairs = [(order[k], order[(k + 1) % 4]) for k in range(4)]                 # the Dishub cycle first
    pairs += [(a, b) for a in order for b in order if a != b and (a, b) not in pairs]
    phases = []
    for k, st in enumerate(order):
        phases.append((DISHUB_GREEN_S[st], state(lambda i: "G" if i in green[st] else "r"), f"{st} green"))
        a, b = pairs[k]
        phases += [(YELLOW_S, amber(a, b), f"{a}>{b} amber"), (ALLRED_S, allred(a, b), f"{a}>{b} all-red")]
    for a, b in pairs[4:]:
        phases += [(YELLOW_S, amber(a, b), f"{a}>{b} amber"), (ALLRED_S, allred(a, b), f"{a}>{b} all-red")]
    return phases


def write_program(net_in: str, net_out: str, patches_dir: str) -> str:
    phases = stage_program(net_in)
    tll = os.path.join(patches_dir, "fatmawati_program.tll.xml")
    with open(tll, "w") as f:
        f.write('<!-- Dishub static plan at Perempatan Fatmawati (Sumabrata et al. 2026): S1 W+E through 80 s, S2 E through+right 12 s,\n'
                '     S3 N 78 s, S4 S 45 s, amber 2 s + all-red 2 s; phases 0-11 are that cycle, the rest are the other stage\n'
                '     transitions for adaptive control (traffic_twin/fatmawati/build_net.py). -->\n')
        f.write(f'<tlLogics>\n    <tlLogic id="{TLS_ID}" type="static" programID="dishub" offset="0">\n')
        for dur, st, name in phases:
            f.write(f'        <phase duration="{dur:g}" state="{st}" name="{name}"/>\n')
        f.write('    </tlLogic>\n</tlLogics>\n')
    netconvert(["--sumo-net-file", net_in, "--tllogic-files", tll, "-o", net_out, "--lefthand", "--check-lane-foes.all"])
    return tll


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius-km", type=float, default=2.7)
    ap.add_argument("--rhk-depth-m", type=float, default=RHK_DEPTH_M)
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args(argv)
    osm = download_osm(args.radius_km, os.path.join(DATA, "osm", "fatmawati_roads.osm.xml"), args.force_download)
    netdir, patches = os.path.join(DATA, "net"), os.path.join(DATA, "patches")
    os.makedirs(netdir, exist_ok=True)
    raw, mid, final = (os.path.join(netdir, f) for f in ("fatmawati_raw.net.xml", "fatmawati_tls.net.xml", "fatmawati.net.xml"))
    build_raw(osm, raw, patches)
    patch_controller(raw, mid, patches, args.rhk_depth_m)
    write_program(mid, final, patches)
    os.remove(mid)
    import sumolib
    net = sumolib.net.readNet(final, withInternal=False, withPrograms=True)
    tl = net.getTLS(TLS_ID)
    print(f"wrote {final}: {len(net.getEdges())} edges, {len(net.getNodes())} nodes, {len(net.getTrafficLights())} signal "
          f"controllers; '{TLS_ID}' controls {len(tl.getConnections())} links with program "
          f"{list(tl.getPrograms())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
