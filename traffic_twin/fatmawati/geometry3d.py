"""Static 3D geometry of the Fatmawati twin for the web viewer (DECISIONS §13.5).

One JSON file, in metres relative to the junction centre (x east, y north, z up), built from the
same SUMO network the simulation runs on (one source of truth) and from OpenStreetMap building
footprints ((c) OpenStreetMap contributors, ODbL):
  lanes      every normal lane: centre line (x, y, z), width, road class
  junctions  junction surfaces (polygons)
  stoplines  the Fatmawati stop lines and motorcycle boxes (Ruang Henti Khusus)
  heads      signal heads of the Fatmawati controller: position + the link indices they show
  flyover    the JORR toll viaduct over the junction: decks, piers and lamp posts
  mrt        the MRT Jakarta viaduct, its stations and the two tracks the trains run on (mrt.py)
  buildings  footprints with a height (OSM height / levels x 3.2 m, else a default by building type)
  hospital   RSUP Fatmawati
"""

from __future__ import annotations

import bisect
import json
import math
import os

from .build_net import APPROACHES, JUNCTION_LAT, JUNCTION_LON, TLS_ID, RHK_DEPTH_M
from .demand import DATA, HOSPITAL_LONLAT, NET

DEFAULT_HEIGHT = {"house": 7.0, "residential": 7.0, "detached": 7.0, "terrace": 8.0, "apartments": 36.0, "commercial": 14.0,
                  "retail": 10.0, "office": 30.0, "hotel": 30.0, "hospital": 22.0, "school": 12.0, "university": 16.0,
                  "mosque": 12.0, "church": 14.0, "industrial": 10.0, "warehouse": 9.0, "garage": 4.0, "roof": 5.0,
                  "yes": 8.0}

# The JORR toll road (Jalan Tol Lingkar Luar Jakarta) crosses the junction on a viaduct.  Its height is the
# network's own: netconvert lifts the OSM bridge ways (layer 1) by 6 m and grades the ends
# (--osm.layer-elevation in build_net), so toll vehicles, which take their z from SUMO, drive on the deck.
# netconvert lifts every bridge that way, also the short river bridges on RA Kartini and TB Simatupang, which
# become 6 m humps (a few centimetres of extra lane length in the simulation).  The viewer therefore shows
# heights on the toll roads only and draws every other road, and the vehicles on it, at ground level.
RAISED_TYPES = ("highway.motorway", "highway.motorway_link")
TAPER_M = 60.0
DECK_DEPTH_M = 1.5               # box girder under the road surface
PARAPET_M, PARAPET_W_M = 1.0, 0.35
SHOULDER_M = 1.0                 # paved shoulder outside the outer lanes, up to the parapet
RAISED_M = 0.3                   # a toll road is drawn as a structure where it is higher than this
PIER_SPAN_M = 30.0               # usual span; longer where a street passes below
PIER_HALF_M = 1.2                # half the column, across the road
PIER_CLEAR_M = 1.0               # kept free between a column and any lane or junction below
LAMP_EVERY_M = 45.0
MRT_LAMP_CLEAR_M = 10.0          # no toll lamp post under the MRT viaduct
# MRT viaduct (mrt.py has the line, the stations and the timetable)
MRT_DECK_DROP_M = 0.5            # deck top below the rail
MRT_GIRDER_M = 1.8
MRT_HALF_M = 2.7                 # deck half width per track (car 2.95 m, walkway, parapet)
MRT_PARTED_M = 12.0              # tracks further apart than this get separate decks and piers
MRT_SPAN_M = 30.0
MRT_PIER_PAD_M = 1.4             # column half width (1.0 m) + clearance: MRT piers stand in road medians
MAST_EVERY_M = 50.0
STATION_FLOOR_M, STATION_ROOF_M = 7.5, 21.0     # concourse underside; roof (the platform floor is 1 m above the rail)


def _r(v, k=2):
    return round(float(v), k)


def raised_parts(net) -> tuple:
    """What the viewer draws raised: ({toll lane id: (points (x, y, z), distances along them, SUMO lane length)},
    {junctions between toll roads only}).  Everything else is drawn at ground level.  A toll ramp that lands on a
    street junction comes down to the ground over its last TAPER_M (SUMO can have such a junction at the height
    of a lifted river bridge, e.g. the western off-ramp onto RA Kartini)."""
    edges = {e.getID(): e for e in net.getEdges() if e.getType() in RAISED_TYPES}
    pure, mixed = set(), set()
    for n in net.getNodes():
        touch = [e.getID() in edges for e in n.getIncoming() + n.getOutgoing()]
        if any(touch):
            (pure if all(touch) else mixed).add(n.getID())
    lanes = {}
    for e in edges.values():
        cut_a, cut_b = e.getFromNode().getID() in mixed, e.getToNode().getID() in mixed
        for ln in e.getLanes():
            shp = [tuple(p) for p in ln.getShape3D()]
            dist = [0.0]
            for a, b in zip(shp, shp[1:]):
                dist.append(dist[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
            L = dist[-1]
            for d0 in ([TAPER_M] if cut_a else []) + ([L - TAPER_M] if cut_b else []):   # bend points of the taper
                if 0.0 < d0 < L and all(abs(d0 - d) > 0.5 for d in dist):
                    i = next(k for k, d in enumerate(dist) if d > d0)
                    f = (d0 - dist[i - 1]) / (dist[i] - dist[i - 1])
                    shp.insert(i, tuple(a + f * (b - a) for a, b in zip(shp[i - 1], shp[i])))
                    dist.insert(i, d0)
            pts = []
            for d, (x, y, z) in zip(dist, shp):
                f = min(1.0, d / TAPER_M if cut_a else 1.0, (L - d) / TAPER_M if cut_b else 1.0)
                pts.append((x, y, z * max(f, 0.0)))
            lanes[ln.getID()] = (pts, dist, ln.getLength())
    return lanes, pure


def raised_z(profile: tuple, pos: float) -> float:
    """Drawn height at SUMO lane position `pos` of a raised lane (profile from raised_parts)."""
    pts, dist, length = profile
    d = pos / length * dist[-1] if length > 0 else 0.0
    i = bisect.bisect_right(dist, d)
    if i <= 0:
        return pts[0][2]
    if i >= len(dist):
        return pts[-1][2]
    f = (d - dist[i - 1]) / ((dist[i] - dist[i - 1]) or 1.0)
    return pts[i - 1][2] + f * (pts[i][2] - pts[i - 1][2])


def _walk(pts: list, step: float) -> list:
    """(s, x, y, z, ux, uy) every `step` metres along a 3D polyline."""
    out, s, nxt = [], 0.0, 0.0
    for (ax, ay, az), (bx, by, bz) in zip(pts, pts[1:]):
        L = math.hypot(bx - ax, by - ay)
        if L == 0:
            continue
        while nxt <= s + L:
            f = (nxt - s) / L
            out.append((nxt, ax + f * (bx - ax), ay + f * (by - ay), az + f * (bz - az), (bx - ax) / L, (by - ay) / L))
            nxt += step
        s += L
    return out


def _seg_dist(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def _inside(px, py, poly) -> bool:
    inside = False
    for (ax, ay), (bx, by) in zip(poly, poly[1:] + poly[:1]):
        if (ay > py) != (by > py) and px < ax + (py - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


class _Clearance:
    """What a pier column must keep clear of: the lanes of other roads (by their half width), ground-level
    junctions and extra strips (polyline, half width), hashed by grid cell."""

    def __init__(self, lanes: list, junctions: list, strips: tuple = (), pad_max: float = PIER_HALF_M + PIER_CLEAR_M,
                 cell: float = 25.0):
        self.cell, self.grid = cell, {}
        for ln in lanes:
            e, hw = ln["id"].rsplit("_", 1)[0], ln["w"] / 2
            self._strip(e, ln["p"], hw, pad_max)
        for k, (pts, hw) in enumerate(strips):
            self._strip(f"strip{k}", pts, hw, pad_max)
        for j in junctions:
            if j["z"] < RAISED_M:
                xs, ys = [p[0] for p in j["p"]], [p[1] for p in j["p"]]
                for c in self._cells(min(xs) - pad_max, min(ys) - pad_max, max(xs) + pad_max, max(ys) + pad_max):
                    self.grid.setdefault(c, []).append(("", j["p"]))

    def _cells(self, x0, y0, x1, y1):
        for i in range(math.floor(x0 / self.cell), math.floor(x1 / self.cell) + 1):
            for j in range(math.floor(y0 / self.cell), math.floor(y1 / self.cell) + 1):
                yield i, j

    def _strip(self, key, pts, hw, pad_max):
        for a, b in zip(pts, pts[1:]):
            r = hw + pad_max
            for c in self._cells(min(a[0], b[0]) - r, min(a[1], b[1]) - r, max(a[0], b[0]) + r, max(a[1], b[1]) + r):
                self.grid.setdefault(c, []).append((key, hw, a[0], a[1], b[0], b[1]))

    def clear(self, x, y, pad: float = PIER_HALF_M + PIER_CLEAR_M, own: str = "") -> bool:
        for item in self.grid.get((math.floor(x / self.cell), math.floor(y / self.cell)), ()):
            if len(item) == 2:
                poly = item[1]
                if _inside(x, y, poly) or any(_seg_dist(x, y, *a, *b) < pad for a, b in zip(poly, poly[1:] + poly[:1])):
                    return False
            elif item[0] != own and _seg_dist(x, y, *item[2:]) < item[1] + pad:
                return False
        return True


def _near(x, y, lines: tuple, d: float) -> bool:
    return any(_seg_dist(x, y, a[0], a[1], b[0], b[1]) < d for pts in lines for a, b in zip(pts, pts[1:]))


def flyover(lanes: list, junctions: list, avoid: tuple = ()) -> dict:
    """Decks, piers and lamp posts of the raised toll road, from the exported lanes (metres from the junction).

    A deck follows each raised motorway edge between its outer lanes plus shoulders.  Piers stand every
    PIER_SPAN_M under it, but never in a lane of another road or in a ground-level junction: there the span
    grows until the column is clear, as over Jl. RS Fatmawati.  No lamp post stands under the `avoid`
    polylines (the MRT viaduct, which crosses above the deck).
    """
    room = _Clearance(lanes, junctions)
    clear = lambda edge, x, y: room.clear(x, y, own=edge)

    by_edge = {}
    for ln in lanes:
        if ln["k"].startswith("motorway"):
            by_edge.setdefault(ln["id"].rsplit("_", 1)[0], []).append(ln)
    decks, piers, lamps = [], [], []
    for edge, group in sorted(by_edge.items()):
        group.sort(key=lambda l: l["idx"])
        a, b = group[0]["p"], group[-1]["p"]
        if max(p[2] for p in a) <= RAISED_M:
            continue
        if len(a) == len(b):
            centre = [[(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2] for p, q in zip(a, b)]
        else:                                    # lanes of one edge normally share their points
            centre = [list(p) for p in group[len(group) // 2]["p"]]
        raised = [i for i, p in enumerate(centre) if p[2] > RAISED_M]
        if not raised:
            continue
        centre = centre[max(raised[0] - 1, 0):raised[-1] + 2]
        if len(centre) < 2:
            continue
        hw = sum(l["w"] for l in group) / 2 + SHOULDER_M
        # side of the median lane (highest index) relative to the left normal of the driving direction
        tx, ty = centre[1][0] - centre[0][0], centre[1][1] - centre[0][1]
        side = 0
        if len(group) > 1:
            side = 1 if -ty * (b[0][0] - a[0][0]) + tx * (b[0][1] - a[0][1]) > 0 else -1
        decks.append(dict(edge=edge, hw=_r(hw), side=side, p=[[_r(v) for v in p] for p in centre]))
        last = -math.inf
        for s, x, y, z, ux, uy in _walk(centre, 2.0):
            if s - last >= PIER_SPAN_M and z - DECK_DEPTH_M >= 0.8 and clear(edge, x, y):
                piers.append([_r(x), _r(y), _r(z - DECK_DEPTH_M), _r(ux, 3), _r(uy, 3), _r(hw - 0.75)])
                last = s
        if side:
            for s, x, y, z, ux, uy in _walk(centre, LAMP_EVERY_M)[1:]:
                o = side * (hw - PARAPET_W_M / 2)
                lx, ly = x - uy * o, y + ux * o
                if z > 2.0 and not _near(lx, ly, avoid, MRT_LAMP_CLEAR_M):
                    lamps.append([_r(lx), _r(ly), _r(z + PARAPET_M), _r(uy * side, 3), _r(-ux * side, 3)])
    return dict(depth=DECK_DEPTH_M, parapet=PARAPET_M, parapet_w=PARAPET_W_M, decks=decks, piers=piers, lamps=lamps,
                name="Jalan Tol Lingkar Luar Jakarta (JORR)")


def _nearest(pts: list, q) -> tuple:
    """(distance, nearest point) of a polyline to q."""
    best = (math.inf, q)
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / L2))
        c = (a[0] + t * dx, a[1] + t * dy)
        d = math.hypot(q[0] - c[0], q[1] - c[1])
        if d < best[0]:
            best = (d, c)
    return best


def metro(lanes: list, junctions: list, toll_decks: list, line: dict) -> dict:
    """The MRT viaduct along the two tracks of mrt.lines().

    - A deck per track, joined where the tracks run side by side.
    - Piers between the tracks every MRT_SPAN_M, clear of the streets and of the toll deck.
    - Catenary masts, and the stations: concourse, glazed platform level, roof.
    """
    from .mrt import RAIL_Z
    tracks = line["tracks"]
    names = sorted(tracks)
    top = RAIL_Z - MRT_DECK_DROP_M
    under = top - MRT_GIRDER_M
    room = _Clearance(lanes, junctions, strips=tuple((d["p"], d["hw"] + 0.3) for d in toll_decks))
    stations = [st["p"] for st in line["stations"]]
    decks, piers, masts = [], [], []
    for name in names:
        pts = tracks[name]["p"]
        other = next((tracks[n]["p"] for n in names if n != name), None)
        sides, inner = [], []
        for i, (x, y) in enumerate(pts):
            a, b = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
            L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
            nx, ny = -(b[1] - a[1]) / L, (b[0] - a[0]) / L          # left normal of the direction of travel
            d, q = _nearest(other, (x, y)) if other else (math.inf, (x, y))
            sides.append(1 if nx * (q[0] - x) + ny * (q[1] - y) > 0 else -1)   # side the other track is on
            inner.append(_r(min(MRT_HALF_M, d / 2) if d < MRT_PARTED_M else MRT_HALF_M))
        decks.append(dict(track=name, p=pts, side=sides, inner=inner))
        for _, x, y, _, ux, uy in _walk([[*q, 0.0] for q in pts], MAST_EVERY_M)[1:]:   # masts on the outer edge
            d, q = _nearest(other, (x, y)) if other else (math.inf, (x, y))
            sg = 1 if -uy * (q[0] - x) + ux * (q[1] - y) > 0 else -1
            mx, my = x + uy * sg * (MRT_HALF_M - 0.3), y - ux * sg * (MRT_HALF_M - 0.3)
            if not any(_inside(mx, my, st) for st in stations):
                masts.append([_r(mx), _r(my), _r(top), _r(-uy * sg, 3), _r(ux * sg, 3)])
    main = tracks.get("northbound") and "northbound" or names[0]
    other = next((tracks[n]["p"] for n in names if n != main), None)
    last = -math.inf
    for s, x, y, _, ux, uy in _walk([[*q, 0.0] for q in tracks[main]["p"]], 2.0):
        d, q = _nearest(other, (x, y)) if other else (math.inf, (x, y))
        cx, cy, half = ((x + q[0]) / 2, (y + q[1]) / 2, d / 2 + MRT_HALF_M) if d < MRT_PARTED_M else (x, y, MRT_HALF_M)
        if s - last >= MRT_SPAN_M and room.clear(cx, cy, pad=MRT_PIER_PAD_M):
            piers.append([_r(cx), _r(cy), _r(under), _r(ux, 3), _r(uy, 3), _r(half - 0.5)])
            last = s
    halls = []
    for st in line["stations"]:
        poly = st["p"]
        cx, cy = sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)
        for a, b in zip(poly, poly[1:] + poly[:1]):                  # columns under the concourse, inset 1.5 m
            n = max(1, round(math.dist(a, b) / 25.0))
            for k in range(n):
                px, py = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
                L = math.hypot(cx - px, cy - py) or 1.0
                px, py = px + (cx - px) / L * 1.5, py + (cy - py) / L * 1.5
                if room.clear(px, py, pad=0.9):
                    piers.append([_r(px), _r(py), STATION_FLOOR_M, 1.0, 0.0, 0.0])
        halls.append(dict(name=st["name"], p=poly, roof_p=_grow(poly, 1.5), floor=STATION_FLOOR_M, platform=_r(RAIL_Z + 1.0),
                          roof=STATION_ROOF_M))
    return dict(rail=RAIL_Z, top=_r(top), under=_r(under), half=MRT_HALF_M, tracks=tracks, decks=decks, piers=piers,
                masts=masts, stations=halls, name="MRT Jakarta, North-South Line")


def _grow(poly: list, d: float) -> list:
    """The polygon moved outward by d metres (mitred corners; for the station roofs' overhang)."""
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(poly, poly[1:] + poly[:1]))
    sgn = 1.0 if area > 0 else -1.0                              # counter-clockwise: the outward normal is on the right
    out, n = [], len(poly)
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        normals = []
        for p, q in ((a, b), (b, c)):
            L = math.hypot(q[0] - p[0], q[1] - p[1]) or 1.0
            normals.append((sgn * (q[1] - p[1]) / L, -sgn * (q[0] - p[0]) / L))
        mx, my = normals[0][0] + normals[1][0], normals[0][1] + normals[1][1]
        k = mx * normals[0][0] + my * normals[0][1]
        k = d / k if abs(k) > 0.2 else d
        out.append([_r(b[0] + mx * k), _r(b[1] + my * k)])
    return out


def _box(net, x0: float, y0: float, radius_km: float) -> tuple:
    """The road network's download box in local metres (x0, y0, x1, y1)."""
    from .build_net import bbox
    s, w, n, e = bbox(radius_km)
    c = [tuple(v - o for v, o in zip(net.convertLonLat2XY(lon, lat), (x0, y0))) for lon in (w, e) for lat in (s, n)]
    return (max(p[0] for p in c if p[0] < 0), max(p[1] for p in c if p[1] < 0),
            min(p[0] for p in c if p[0] > 0), min(p[1] for p in c if p[1] > 0))


def build(out_path: str | None = None, buildings_path: str | None = None, radius_m: float = 1600.0) -> str:
    import sumolib
    net = sumolib.net.readNet(NET, withInternal=False, withPrograms=True)
    x0, y0 = net.convertLonLat2XY(JUNCTION_LON, JUNCTION_LAT)
    rel = lambda x, y: (_r(x - x0), _r(y - y0))
    raised_lanes, raised_nodes = raised_parts(net)
    lanes = []
    for e in net.getEdges():
        if e.getFunction() != "":
            continue
        kind = e.getType().replace("highway.", "")
        for ln in e.getLanes():
            if ln.getID() in raised_lanes:
                pts = [[*rel(x, y), _r(z)] for x, y, z in raised_lanes[ln.getID()][0]]
            else:
                pts = [[*rel(p[0], p[1]), 0.0] for p in ln.getShape()]
            lanes.append(dict(id=ln.getID(), w=_r(ln.getWidth()), k=kind, idx=ln.getIndex(), n=e.getLaneNumber(), p=pts))
    junctions = []
    for n in net.getNodes():
        shp = n.getShape3D()
        if len(shp) >= 3:
            z = sum(p[2] for p in shp) / len(shp) if n.getID() in raised_nodes else 0.0
            junctions.append(dict(id=n.getID(), z=_r(z), p=[list(rel(p[0], p[1])) for p in shp]))
    # stop lines + motorcycle boxes of the Fatmawati controller, signal heads with their link indices
    tl = net.getTLS(TLS_ID)
    by_lane = {}
    for inl, outl, idx in tl.getConnections():
        by_lane.setdefault(inl.getID(), []).append(idx)
    stoplines, heads = [], []
    approach_edges = set(APPROACHES.values())
    for lane_id, idxs in sorted(by_lane.items()):
        ln = net.getLane(lane_id)
        shp = ln.getShape()
        (xa, ya), (xb, yb) = shp[-2][:2], shp[-1][:2]
        L = math.hypot(xb - xa, yb - ya) or 1.0
        ux, uy = (xb - xa) / L, (yb - ya) / L
        nx, ny = -uy, ux
        hw = ln.getWidth() / 2.0
        end = rel(xb, yb)
        stoplines.append(dict(lane=lane_id, a=[_r(end[0] + nx * hw), _r(end[1] + ny * hw)], b=[_r(end[0] - nx * hw), _r(end[1] - ny * hw)],
                              rhk=ln.getEdge().getID() in approach_edges, depth=RHK_DEPTH_M if ln.getEdge().getID() in approach_edges else 0.0,
                              ux=_r(ux, 4), uy=_r(uy, 4), links=sorted(idxs)))
        if ln.getIndex() == ln.getEdge().getLaneNumber() - 1:      # one head per approach edge, on the median side
            heads.append(dict(x=_r(end[0] - nx * (hw + 0.8)), y=_r(end[1] - ny * (hw + 0.8)), edge=ln.getEdge().getID(),
                              links=sorted(i for l2 in ln.getEdge().getLanes() for i in by_lane.get(l2.getID(), []))))
    buildings = []
    bpath = buildings_path or os.path.join(DATA, "osm", "buildings_1600m.json")
    if os.path.exists(bpath):
        with open(bpath) as f:
            elements = json.load(f)["elements"]
        for el in elements:
            g = el.get("geometry")
            if not g or len(g) < 4:
                continue
            tags = el.get("tags", {})
            h = None
            try:
                h = float(str(tags.get("height", "")).replace("m", "").strip())
            except ValueError:
                pass
            if h is None and tags.get("building:levels"):
                try:
                    h = 3.2 * float(tags["building:levels"]) + 1.0
                except ValueError:
                    h = None
            if h is None:
                h = DEFAULT_HEIGHT.get(tags.get("building", "yes"), 8.0)
            pts = []
            for p in g:
                x, y = net.convertLonLat2XY(p["lon"], p["lat"])
                pts.append(list(rel(x, y)))
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            if math.hypot(cx, cy) > radius_m:
                continue
            buildings.append(dict(h=_r(h, 1), p=pts[:-1] if pts[0] == pts[-1] else pts,
                                  hosp=tags.get("amenity") == "hospital" or tags.get("building") == "hospital"))
    hx, hy = net.convertLonLat2XY(*HOSPITAL_LONLAT)
    line = None
    try:                                                         # the MRT line (OpenStreetMap, cached)
        from . import mrt
        mrt.download()
        line = mrt.lines(lambda lon, lat: rel(*net.convertLonLat2XY(lon, lat)), _box(net, x0, y0, mrt.RADIUS_KM))
    except (OSError, ValueError, KeyError) as err:               # offline with nothing cached: no MRT in the view
        print("MRT not drawn:", err)
    fly = flyover(lanes, junctions, avoid=tuple(t["p"] for t in line["tracks"].values()) if line else ())
    data = dict(origin=dict(lat=JUNCTION_LAT, lon=JUNCTION_LON), lanes=lanes, junctions=junctions, stoplines=stoplines, heads=heads,
                flyover=fly, mrt=metro(lanes, junctions, fly["decks"], line) if line else None, buildings=buildings, hospital=dict(name="RSUP Fatmawati", x=_r(hx - x0), y=_r(hy - y0)),
                attribution="Road and building data (c) OpenStreetMap contributors, ODbL")
    out_path = out_path or os.path.join(DATA, "viewer", "geometry.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    return out_path


if __name__ == "__main__":
    p = build()
    d = json.load(open(p))
    print(p, os.path.getsize(p) // 1024, "KB:", {k: len(v) for k, v in d.items() if isinstance(v, list)})
