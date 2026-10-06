"""Grid network ("#" shape) of junctions, directed edges with lanes and connectors.

Traffic-engineering idea: a road network is a directed graph.  Each travel
direction of a road is its own edge (box edge to box edge), and each junction
holds lane-level "connectors" through the box: through traffic keeps its lane,
near-side turns leave the kerb lane and far-side turns the median lane, each turn
into either exit lane (24 paths with two lanes), each on its real path - a straight
line or a circular arc set by the kerb radius (DECISIONS §12.1).  Turn exits are
computed by rotating the inbound heading, so the driving side is never hard-coded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig
import functools

from .geometry import BoxPath, divergence_length, make_path, zone_interval

DIRS = ("N", "E", "S", "W")                 # phase index 0..3 = approach arm
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
HEADING = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
MOVEMENTS = ("T", "NS", "FS")


def heading_to_dir(h) -> str:
    """Unit heading vector -> compass name of the direction of travel."""
    hx, hy = h
    if abs(hx) > abs(hy):
        return "E" if hx > 0 else "W"
    return "N" if hy > 0 else "S"


def rotate(h, degrees: float):
    """Rotate a 2-D vector counter-clockwise by `degrees`."""
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    hx, hy = h
    return (round(c * hx - s * hy, 9), round(s * hx + c * hy, 9))


def exit_heading(h_in, movement: str, drive_side: str):
    """Heading after a movement.  NS (near-side turn) = +90 deg CCW under
    left-hand traffic (a left turn), -90 under right-hand; FS the opposite."""
    if movement == "T":
        return h_in
    sign = 1.0 if drive_side == "left" else -1.0
    if movement == "NS":
        return rotate(h_in, 90.0 * sign)
    if movement == "FS":
        return rotate(h_in, -90.0 * sign)
    raise ValueError(movement)


def movement_between(h_in, h_out, drive_side: str) -> str | None:
    """Which movement turns heading h_in into h_out (None = U-turn, forbidden)."""
    for m in MOVEMENTS:
        if exit_heading(h_in, m, drive_side) == tuple(h_out):
            return m
    return None


def lane_offset(h, lane: int, cfg: SimConfig):
    """Lateral offset (dx, dy) of lane `lane` from the road axis.  Lane 0 is the
    kerb lane, lane lanes_per_dir-1 the median lane.  Left normal n = (-hy, hx)
    under left-hand traffic, -n under right-hand traffic."""
    hx, hy = h
    nx, ny = -hy, hx
    if cfg.drive_side == "right":
        nx, ny = -nx, -ny
    dist = cfg.w_lane * (cfg.lanes_per_dir - lane - 0.5)
    return (nx * dist, ny * dist)


@dataclass
class Node:
    id: str
    x: float
    y: float
    is_junction: bool
    junction_id: int = -1     # 0.. for junctions, -1 for boundary nodes
    road_axis: str = ""       # for boundary nodes: "EW" road or "NS" road


@dataclass
class Edge:
    id: str
    from_node: str
    to_node: str
    heading: tuple
    start: tuple              # (x, y) of the edge start on the road axis
    end: tuple
    length: float
    is_stub: bool
    n_lanes: int
    storage_per_lane: int
    stop_s: float                      # stop line position along the edge
    approach_dir: str | None           # arm the vehicle arrives from at the dest junction
    junction_at_dest: int              # -1 if the edge ends at a boundary node
    junction_at_from: int              # -1 if the edge starts at a boundary node
    exit_dir: str                      # direction of travel (= exit direction of the from-junction)
    lanes: list = field(default_factory=list)   # one list per lane, front vehicle first
    entry_times: list = field(default_factory=list)  # for arr_a_30s

    @property
    def storage(self) -> int:
        return self.storage_per_lane * self.n_lanes

    def n_vehicles(self) -> int:
        return sum(len(l) for l in self.lanes)

    def lane_xy(self, lane: int, s: float, cfg: SimConfig):
        """2-D position of a point at distance s along lane `lane` (viewer only)."""
        ox, oy = lane_offset(self.heading, lane, cfg)
        return (self.start[0] + self.heading[0] * s + ox,
                self.start[1] + self.heading[1] * s + oy)


@dataclass
class Connector:
    id: str
    junction_id: int
    approach: str
    movement: str
    in_lane: int
    out_lane: int
    in_edge: str
    out_edge: str
    path: BoxPath
    length: float                       # true path length [m]
    is_turn: bool
    v_cap: float                        # curvature speed limit (v_limit on straight paths)
    siblings: dict = field(default_factory=dict)   # connector id -> divergence length (paths that share the entry lane)
    vehicles: list = field(default_factory=list)   # front vehicle first

    @property
    def key(self):
        return (self.approach, self.movement, self.in_lane, self.out_lane)


@dataclass
class Junction:
    id: int
    node: str
    x: float
    y: float
    approach_edges: dict = field(default_factory=dict)   # arm -> edge id (vehicles arriving from arm)
    exit_edges: dict = field(default_factory=dict)       # direction -> edge id (vehicles leaving toward it)
    connectors: dict = field(default_factory=dict)       # (approach, movement, in_lane, out_lane) -> Connector
    lane_connectors: dict = field(default_factory=dict)  # (approach, in_lane) -> [Connector, ...] (paths sharing an entry)
    neighbours: dict = field(default_factory=dict)       # direction -> junction id or -1

    def connector_list(self):
        return list(self.connectors.values())


@dataclass
class Network:
    cfg: SimConfig
    nodes: dict
    edges: dict
    junctions: list
    boundary_nodes: list                 # ids in a fixed order (spawn order)
    out_edge_of_boundary: dict           # boundary node -> edge leaving it
    in_edge_of_boundary: dict            # boundary node -> edge arriving at it
    successors: dict                     # edge id -> list of (next edge id, movement)
    conflicts: dict = field(default_factory=dict)       # (path key, path key) -> True, cross-approach (§12.1)
    s_clear: dict = field(default_factory=dict)         # (path key, approach B) -> s past which a rear no longer blocks B's green
    zones_on_path: dict = field(default_factory=dict)   # path key -> [(zone id, s_in, s_out), ...] (cross-approach zones)
    zone_foe: dict = field(default_factory=dict)        # zone id -> the matching zone on the other path
    movement_geometry: dict = field(default_factory=dict)  # movement -> (length, v_cap) of the same-index path

    def edge(self, eid: str) -> Edge:
        return self.edges[eid]

    def junction(self, jid: int) -> Junction:
        return self.junctions[jid]

    def junction_ids(self):
        return [j.id for j in self.junctions]

    def box_empty(self, jid: int) -> bool:
        """No vehicle on any connector of the junction (the v1 clearance gate; kept for tests)."""
        return all(len(c.vehicles) == 0 for c in self.junctions[jid].connectors.values())


def build_grid(cfg: SimConfig) -> Network:
    """Build a grid_x x grid_y grid (1x1 is a single junction with 4 stubs).

    Junction J(i,j) sits at (i*L_block, j*L_block); junction_id = j*grid_x + i.
    Boundary nodes W{j}, E{j} on the horizontal roads and S{i}, N{i} on the
    vertical roads, L_stub from the outer junction centres.
    """
    nodes: dict[str, Node] = {}
    junctions: list[Junction] = []
    for j in range(cfg.grid_y):
        for i in range(cfg.grid_x):
            jid = j * cfg.grid_x + i
            nid = f"J{i}{j}"
            nodes[nid] = Node(nid, i * cfg.L_block, j * cfg.L_block, True, jid)
            junctions.append(Junction(jid, nid, i * cfg.L_block, j * cfg.L_block,
                                      neighbours={d: -1 for d in DIRS}))
    x_max = (cfg.grid_x - 1) * cfg.L_block
    y_max = (cfg.grid_y - 1) * cfg.L_block
    boundary = []
    for j in range(cfg.grid_y):
        nodes[f"W{j}"] = Node(f"W{j}", -cfg.L_stub, j * cfg.L_block, False, road_axis="EW")
        nodes[f"E{j}"] = Node(f"E{j}", x_max + cfg.L_stub, j * cfg.L_block, False, road_axis="EW")
        boundary += [f"W{j}", f"E{j}"]
    for i in range(cfg.grid_x):
        nodes[f"S{i}"] = Node(f"S{i}", i * cfg.L_block, -cfg.L_stub, False, road_axis="NS")
        nodes[f"N{i}"] = Node(f"N{i}", i * cfg.L_block, y_max + cfg.L_stub, False, road_axis="NS")
        boundary += [f"S{i}", f"N{i}"]
    boundary = sorted(boundary)

    # undirected road segments between adjacent nodes
    segments = []
    for j in range(cfg.grid_y):
        row = [f"W{j}"] + [f"J{i}{j}" for i in range(cfg.grid_x)] + [f"E{j}"]
        segments += list(zip(row[:-1], row[1:]))
    for i in range(cfg.grid_x):
        col = [f"S{i}"] + [f"J{i}{j}" for j in range(cfg.grid_y)] + [f"N{i}"]
        segments += list(zip(col[:-1], col[1:]))

    edges: dict[str, Edge] = {}
    for a, b in segments:
        for u, v in ((a, b), (b, a)):
            edges[f"{u}_{v}"] = _make_edge(u, v, nodes, cfg)

    out_of_boundary, in_of_boundary = {}, {}
    for e in edges.values():
        if not nodes[e.from_node].is_junction:
            out_of_boundary[e.from_node] = e.id
        if not nodes[e.to_node].is_junction:
            in_of_boundary[e.to_node] = e.id
        if e.junction_at_dest >= 0:
            junctions[e.junction_at_dest].approach_edges[e.approach_dir] = e.id
        if e.junction_at_from >= 0:
            junctions[e.junction_at_from].exit_edges[e.exit_dir] = e.id

    # lane-level connectors on their real paths (§12.1) and neighbours
    successors: dict[str, list] = {eid: [] for eid in edges}
    n = cfg.lanes_per_dir
    for J in junctions:
        for d in DIRS:
            out_e = edges[J.exit_edges[d]]
            J.neighbours[d] = out_e.junction_at_dest
        for approach in DIRS:
            in_e = edges[J.approach_edges[approach]]
            for m in MOVEMENTS:
                h_out = exit_heading(in_e.heading, m, cfg.drive_side)
                out_e = edges[J.exit_edges[heading_to_dir(h_out)]]
                successors[in_e.id].append((out_e.id, m))
                tables = _junction_tables(cfg.lanes_per_dir, cfg.w_lane, cfg.corner_radius, cfg.drive_side,
                                          cfg.a_lat_max, cfg.v_limit, cfg.max_vehicle_width + cfg.conflict_margin)
                for k in lanes_for_movement(m, n):
                    for k_out in exit_lanes_for(m, k, n):
                        p0 = in_e.lane_xy(k, in_e.length, cfg)
                        p1 = out_e.lane_xy(k_out, 0.0, cfg)
                        path = make_path(approach, m, k, k_out, p0, in_e.heading, p1, out_e.heading,
                                         cfg.a_lat_max, cfg.v_limit)
                        c = Connector(f"{J.node}_{approach}{m}{k}{k_out}", J.id, approach, m, k, k_out, in_e.id,
                                      out_e.id, path, path.length, m != "T", path.v_cap)
                        J.connectors[c.key] = c
                        J.lane_connectors.setdefault((approach, k), []).append(c)
        for group in J.lane_connectors.values():
            for a in group:
                for b in group:
                    if a is not b:
                        a.siblings[b.id] = tables["diverge"][(a.key, b.key)]

    net = Network(cfg, nodes, edges, junctions, boundary, out_of_boundary, in_of_boundary, successors)
    tables = _junction_tables(cfg.lanes_per_dir, cfg.w_lane, cfg.corner_radius, cfg.drive_side,
                              cfg.a_lat_max, cfg.v_limit, cfg.max_vehicle_width + cfg.conflict_margin)
    net.conflicts = tables["conflicts"]
    net.s_clear = tables["s_clear"]
    net.zones_on_path = tables["zones_on_path"]
    net.zone_foe = tables["zone_foe"]
    net.movement_geometry = tables["movement_geometry"]
    return net


@functools.lru_cache(maxsize=16)
def _junction_tables(lanes_per_dir: int, w_lane: float, corner_radius: float, drive_side: str,
                     a_lat: float, v_limit: float, clearance: float) -> dict:
    """Conflict matrix, zones, s_clear and divergence lengths of one junction, in coordinates
    relative to its centre and keyed by (approach, movement, in_lane, out_lane).  Every
    junction has the same geometry, so this is computed once per geometry and cached."""
    cfg = SimConfig(lanes_per_dir=lanes_per_dir, w_lane=w_lane, corner_radius=corner_radius,
                    drive_side=drive_side, a_lat_max=a_lat, v_limit=v_limit, grid_x=1, grid_y=1)
    half = cfg.w_box / 2.0
    n = lanes_per_dir
    paths = []
    for approach in DIRS:
        h_in = (-HEADING[approach][0], -HEADING[approach][1])      # travelling away from arm `approach`
        for m in MOVEMENTS:
            h_out = exit_heading(h_in, m, drive_side)
            for k in lanes_for_movement(m, n):
                for k_out in exit_lanes_for(m, k, n):
                    o_in, o_out = lane_offset(h_in, k, cfg), lane_offset(h_out, k_out, cfg)
                    p0 = (-h_in[0] * half + o_in[0], -h_in[1] * half + o_in[1])
                    p1 = (h_out[0] * half + o_out[0], h_out[1] * half + o_out[1])
                    paths.append(make_path(approach, m, k, k_out, p0, h_in, p1, h_out, a_lat, v_limit))
    by_key = {p.key: p for p in paths}
    conflicts, zones_on_path, zone_foe, s_clear, pair_zone = {}, {k: [] for k in by_key}, {}, {}, {}
    zid = 0
    for i, a in enumerate(paths):
        for b in paths:
            if a is b or a.approach == b.approach:
                continue
            z = zone_interval(a, b, clearance)
            if z is None:
                continue
            conflicts[(a.key, b.key)] = True
            pair_zone[(a.key, b.key)] = zid
            zones_on_path[a.key].append((zid, z[0], z[1]))
            s_clear[(a.key, b.approach)] = max(s_clear.get((a.key, b.approach), 0.0), z[1])
            zid += 1
    for (ka, kb), z in pair_zone.items():
        zone_foe[z] = pair_zone[(kb, ka)]
    diverge = {}
    for a in paths:
        for b in paths:
            if a is not b and a.approach == b.approach and a.in_lane == b.in_lane:
                diverge[(a.key, b.key)] = divergence_length(a, b, clearance)
    movement_geometry = {}
    for m in MOVEMENTS:
        k = lanes_for_movement(m, n)[0]
        p = by_key[(DIRS[0], m, k, k)] if (DIRS[0], m, k, k) in by_key else next(q for q in paths if q.movement == m)
        movement_geometry[m] = (p.length, p.v_cap)
    return dict(conflicts=conflicts, zones_on_path=zones_on_path, zone_foe=zone_foe, s_clear=s_clear,
                diverge=diverge, movement_geometry=movement_geometry)


def _make_edge(u: str, v: str, nodes: dict, cfg: SimConfig) -> Edge:
    nu, nv = nodes[u], nodes[v]
    dx, dy = nv.x - nu.x, nv.y - nu.y
    dist = math.hypot(dx, dy)
    h = (round(dx / dist), round(dy / dist))
    start = (nu.x + h[0] * cfg.w_box / 2, nu.y + h[1] * cfg.w_box / 2) if nu.is_junction else (nu.x, nu.y)
    end = (nv.x - h[0] * cfg.w_box / 2, nv.y - h[1] * cfg.w_box / 2) if nv.is_junction else (nv.x, nv.y)
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    is_stub = not (nu.is_junction and nv.is_junction)
    travel_dir = heading_to_dir(h)
    return Edge(
        id=f"{u}_{v}", from_node=u, to_node=v, heading=h, start=start, end=end,
        length=length, is_stub=is_stub, n_lanes=cfg.lanes_per_dir,
        storage_per_lane=cfg.storage_per_lane(length),
        stop_s=length - cfg.stop_line_setback,
        approach_dir=OPPOSITE[travel_dir] if nv.is_junction else None,
        junction_at_dest=nv.junction_id if nv.is_junction else -1,
        junction_at_from=nu.junction_id if nu.is_junction else -1,
        exit_dir=travel_dir,
        lanes=[[] for _ in range(cfg.lanes_per_dir)],
    )


def network_edges_table(net: Network):
    """Rows for network_edges.csv (one per directed edge)."""
    rows = []
    for e in net.edges.values():
        rows.append(dict(
            edge_id=e.id, from_node=e.from_node, to_node=e.to_node,
            heading_deg=round(math.degrees(math.atan2(e.heading[1], e.heading[0])) % 360, 1),
            hx=e.heading[0], hy=e.heading[1], length_m=round(e.length, 3),
            n_lanes=e.n_lanes, storage_veh_per_lane=e.storage_per_lane,
            storage_veh=e.storage, is_stub=e.is_stub,
            approach_dir_at_dest=e.approach_dir if e.approach_dir else "",
            junction_id_at_dest=e.junction_at_dest, junction_id_at_from=e.junction_at_from,
            x_start=e.start[0], y_start=e.start[1], x_end=e.end[0], y_end=e.end[1],
        ))
    return rows


def od_row(net: Network, origin: str, pattern: str, cfg: SimConfig) -> dict:
    """Destination probabilities for one origin boundary node.

    'default': far end of own road 0.30, each end of each parallel road 0.15,
    each end of each crossing road 0.10 (renormalised when the grid has
    fewer roads than 2x2); 'uniform': 1/(n_boundary - 1).
    """
    others = [b for b in net.boundary_nodes if b != origin]
    if pattern == "uniform":
        return {b: 1.0 / len(others) for b in others}
    o = net.nodes[origin]
    w = {}
    for b in others:
        n = net.nodes[b]
        if n.road_axis == o.road_axis:
            same_road = (n.y == o.y) if o.road_axis == "EW" else (n.x == o.x)
            w[b] = cfg.od_own_road if same_road else cfg.od_parallel
        else:
            w[b] = cfg.od_crossing
    tot = sum(w.values())
    return {b: v / tot for b, v in w.items()}


def shortest_route(net: Network, origin: str, destination: str, cfg: SimConfig, rng=None):
    """Dijkstra over edges with turn penalties (T 0, NS +10 m, FS +25 m).

    Ties are broken by `rng` (the routing RNG); with rng=None the first-found
    route wins.  Returns (list of edge ids, list of (junction_id, movement)).
    """
    import heapq
    penalty = {"T": cfg.turn_penalty_T, "NS": cfg.turn_penalty_NS, "FS": cfg.turn_penalty_FS}
    start = net.out_edge_of_boundary[origin]
    goal = net.in_edge_of_boundary[destination]
    best = {start: net.edges[start].length}
    prev = {start: None}
    heap = [(best[start], 0, start)]
    counter = 1
    while heap:
        cost, _, eid = heapq.heappop(heap)
        if cost > best.get(eid, math.inf) + 1e-9:
            continue
        if eid == goal:
            break
        for nxt, m in net.successors[eid]:
            new_cost = cost + net.edges[nxt].length + penalty[m]
            old = best.get(nxt, math.inf)
            better = new_cost < old - 1e-9
            tie = abs(new_cost - old) <= 1e-9 and rng is not None and rng.random() < 0.5
            if better or tie:
                best[nxt] = new_cost
                prev[nxt] = (eid, m)
                counter += 1
                heapq.heappush(heap, (new_cost, counter, nxt))
    if goal not in prev:
        raise RuntimeError(f"no route {origin} -> {destination}")
    edges_rev, moves_rev = [goal], []
    cur = goal
    while prev[cur] is not None:
        p, m = prev[cur]
        moves_rev.append((net.edges[p].junction_at_dest, m))
        edges_rev.append(p)
        cur = p
    return edges_rev[::-1], moves_rev[::-1]


def free_space_at_start(edge: Edge, lane: int, cfg: SimConfig) -> float:
    """Free metres at the start of a lane = rear bumper of its last vehicle."""
    lst = edge.lanes[lane]
    if not lst:
        return edge.length
    return lst[-1].s - lst[-1].l_veh


def exit_lane_blocked(edge: Edge, lane: int, cfg: SimConfig) -> bool:
    """Don't-block-the-box test (DECISIONS §4): the start of the exit lane is
    blocked when its last vehicle is queued (v < 2 m/s) and leaves less than
    l_veh + s0 + 1 m free.  A vehicle that is driving away does not block."""
    lst = edge.lanes[lane]
    if not lst:
        return False
    last = lst[-1]
    return last.v < cfg.queue_speed and (last.s - last.l_veh) < cfg.box_free_needed


def choose_lane(edge: Edge, next_movement: str | None, cfg: SimConfig) -> int:
    """Lane rule of DECISIONS §4 at edge entry: NS -> lane 0, FS -> lane 1,
    T or boundary-bound -> the lane with fewer vehicles (tie -> lane 0)."""
    if edge.n_lanes == 1:
        return 0
    if next_movement == "NS":
        return 0
    if next_movement == "FS":
        return edge.n_lanes - 1
    counts = [len(l) for l in edge.lanes]
    return int(np.argmin(counts))

def lanes_for_movement(movement: str | None, n_lanes: int) -> list:
    """Entry lanes that serve a movement (DECISIONS §4, §12): NS kerb lane 0, FS the median
    lane, T (and boundary-bound / no movement) every lane."""
    if n_lanes == 1 or movement in (None, "T"):
        return list(range(n_lanes))
    return [0] if movement == "NS" else [n_lanes - 1]


def exit_lanes_for(movement: str, in_lane: int, n_lanes: int) -> list:
    """Lane discipline through the box (DECISIONS §12.1): through traffic keeps its lane index;
    a turn may end in any exit lane (the one its next turn needs, else the emptier one)."""
    if movement == "T":
        return [in_lane]
    return list(range(n_lanes))


def route_from_edge(net: Network, start_edge: str, destination: str, cfg: SimConfig):
    """Dijkstra (same turn penalties as shortest_route, first-found tie-break, no RNG draw)
    from an edge to a boundary node: (edges incl. start_edge, moves).  Used to reroute a
    vehicle that missed its lane (DECISIONS §12)."""
    import heapq
    penalty = {"T": cfg.turn_penalty_T, "NS": cfg.turn_penalty_NS, "FS": cfg.turn_penalty_FS}
    goal = net.in_edge_of_boundary[destination]
    best = {start_edge: 0.0}
    prev = {start_edge: None}
    heap = [(0.0, 0, start_edge)]
    counter = 1
    while heap:
        cost, _, eid = heapq.heappop(heap)
        if cost > best.get(eid, math.inf) + 1e-9:
            continue
        if eid == goal:
            break
        for nxt, m in net.successors[eid]:
            new_cost = cost + net.edges[nxt].length + penalty[m]
            if new_cost < best.get(nxt, math.inf) - 1e-9:
                best[nxt] = new_cost
                prev[nxt] = (eid, m)
                counter += 1
                heapq.heappush(heap, (new_cost, counter, nxt))
    if goal not in prev:
        return None
    edges_rev, moves_rev = [goal], []
    cur = goal
    while prev[cur] is not None:
        p, m = prev[cur]
        moves_rev.append((net.edges[p].junction_at_dest, m))
        edges_rev.append(p)
        cur = p
    return edges_rev[::-1], moves_rev[::-1], best[goal]
