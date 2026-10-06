"""Calibration: measure the saturation flow in the simulator, then convert the
v/c targets of DECISIONS §7 into arrival rates per entry node.

Traffic-engineering idea: when a queue is released by a green, vehicles cross
the stop line at a steady "saturation headway" after a start-up delay.  The
textbook capacity of an approach = saturation flow x (effective green / cycle);
the capacity a junction REALISES under fixed time differs (turning vehicles slow
to the curvature limit of their arc, the clearance gate waits for conflict zones,
drivers change lane), so it is measured directly from a saturated episode
(DECISIONS §11.9) and the demand band is set as a fraction (v/c) of that measured
capacity, never from the textbook 1800 veh/h/lane.  Saturation flow is a per-lane
quantity (HCM), so it is measured with lane changes switched off: otherwise the
test queue spills into the empty neighbouring lane (DECISIONS §12.6).
"""

from __future__ import annotations

import math

import numpy as np

from .config import SimConfig
from .network import DIR_INDEX, DIRS, OPPOSITE, shortest_route
from .sim import Scenario, World
from .signals import ALLRED
from .vehicles import STOPPING, Vehicle, free_flow_time, sample_params


def place_standing_queue(world: World, approach: str, lane: int, n: int) -> list:
    """Put n stationary through-vehicles on one lane of a 1x1 approach edge, the
    first one s0 behind the stop line, the rest bumper-to-bumper with gap s0."""
    cfg = world.cfg
    J = world.net.junctions[0]
    edge = world.net.edges[J.approach_edges[approach]]
    origin = edge.from_node
    destination = world.net.edges[J.exit_edges[OPPOSITE[approach]]].to_node   # straight across
    vehicles = []
    s = edge.stop_s - cfg.s0
    for _ in range(n):
        params = sample_params(world.rng_vehicles, cfg)
        route_edges, route_moves = shortest_route(world.net, origin, destination, cfg)
        veh = Vehicle(world.next_vid, params, origin, destination, route_edges, route_moves, 0.0)
        world.next_vid += 1
        world.n_spawn_requests += 1
        veh.needs = world.compute_needs(veh)
        veh.free_flow_time = free_flow_time(route_edges, route_moves, world.net.edges, veh.v0, cfg,
                                            world.net.movement_geometry)
        veh.edge, veh.lane, veh.s, veh.v = edge, lane, s, 0.0
        veh.lc_prev_lane = lane
        veh.enter_t = 0.0
        veh.latch = STOPPING
        edge.lanes[lane].append(veh)
        world.active.append(veh)
        vehicles.append(veh)
        s -= veh.l_veh + cfg.s0
    return vehicles


def measure_saturation_flow(cfg: SimConfig, approach: str = "N", lane: int = 0,
                            seed: int = 0, n: int | None = None) -> dict:
    """Validation experiment 1 (DECISIONS §8): 20-vehicle standing queue, green on.

    Returns crossing times, headways, the saturation headway (mean headway of
    vehicles 5-15), saturation flow in veh/h/lane and the start-up lost time
    (sum over the first four headways of their excess over the saturation headway).
    """
    n = cfg.discharge_queue_n if n is None else n
    cfg1 = cfg.copy(grid_x=1, grid_y=1, warmup_s=0.0, main_s=200.0, cooldown_s=0.0, lane_changes=False)
    world = World(cfg1, Scenario(lambda_base_vph=0.0, controller="longest_queue"), seed=seed)
    plant = world.plants[0]
    plant.state = ALLRED                       # red now, green for `approach` at t = allred_min
    plant.phase = (DIR_INDEX[approach] + 1) % 4
    plant.pending_phase = DIR_INDEX[approach]
    plant.allred_start_t = 0.0
    vehicles = place_standing_queue(world, approach, lane, n)
    while world.t < cfg1.episode_length_s and any(math.isnan(v.last_stop_cross_t) for v in vehicles):
        world.step()
    t_green = plant.green_onset_t
    t_cross = np.array([v.last_stop_cross_t for v in vehicles])
    headways = np.diff(np.concatenate([[t_green], t_cross]))        # h_1 = t_1 - t_green
    i0, i1 = cfg.sat_headway_vehicles                                # vehicles 5..15 (1-indexed)
    h_sat = float(np.mean(headways[i0 - 1:i1]))
    lost = float(np.sum(headways[:i0 - 1] - h_sat))                 # start-up lost time
    return dict(
        t_green=t_green, t_cross=t_cross, headways=headways, h_sat=h_sat,
        sat_flow_vphpl=3600.0 / h_sat, startup_lost_time_s=lost,
        n=n, approach=approach, lane=lane,
    )


def approach_flow_per_unit_lambda(cfg: SimConfig, od_pattern: str = "default",
                                  demand_asym: str = "sym", n_tie_samples: int = 20) -> dict:
    """Expected vehicles per hour on every approach edge when every entry node
    spawns 1 veh/h (times its asymmetry factor), from the OD rows and the routes.
    Route ties are averaged over `n_tie_samples` random tie-breaks."""
    from .network import build_grid, od_row
    net = build_grid(cfg)
    flow = {eid: 0.0 for eid in net.edges}
    rng = np.random.default_rng(0)
    for origin in net.boundary_nodes:
        axis = net.nodes[origin].road_axis
        asym = 1.0
        if demand_asym == "ns_heavy":
            asym = cfg.asym_ns_factor if axis == "NS" else cfg.asym_ew_factor
        for dest, p in od_row(net, origin, od_pattern, cfg).items():
            for _ in range(n_tie_samples):
                edges, _moves = shortest_route(net, origin, dest, cfg, rng)
                for e in edges:
                    flow[e] += asym * p / n_tie_samples
    return flow


def busiest_approach_flow(cfg: SimConfig, od_pattern: str = "default", demand_asym: str = "sym"):
    """(edge id, veh/h per unit lambda) of the busiest internal approach
    (all approaches when the grid has no internal edges, i.e. 1x1)."""
    from .network import build_grid
    net = build_grid(cfg)
    flow = approach_flow_per_unit_lambda(cfg, od_pattern, demand_asym)
    internal = [e.id for e in net.edges.values() if e.junction_at_dest >= 0 and not e.is_stub]
    candidates = internal if internal else [e.id for e in net.edges.values() if e.junction_at_dest >= 0]
    best = max(candidates, key=lambda e: flow[e])
    return best, flow[best]


def approach_capacity_vph(cfg: SimConfig, sat_flow_vphpl: float) -> float:
    """capacity per approach = measured_sat_flow * lanes_per_dir * 18 / 96 (DECISIONS §7)."""
    return sat_flow_vphpl * cfg.lanes_per_dir * cfg.effective_green_s / cfg.fixed_time_cycle_s


def measure_capacity(cfg: SimConfig, sat_flow_vphpl: float, seed: int = 0, x: float | None = None,
                     main_s: float | None = None, warmup_s: float = 300.0) -> dict:
    """Realised fixed-time capacity per approach (DECISIONS §11.9): one saturated 1x1
    fixed-time episode at nominal x = capacity_probe_x of the FORMULA capacity (every
    approach over-saturated), capacity_vph = main-period throughput / 4 approaches.
    Also returns the formula value and capacity_realisation = measured / formula."""
    x = cfg.capacity_probe_x if x is None else x
    main_s = cfg.capacity_probe_main_s if main_s is None else main_s
    cfg1 = cfg.copy(grid_x=1, grid_y=1, warmup_s=warmup_s, main_s=main_s, cooldown_s=0.0)
    cap_formula = approach_capacity_vph(cfg, sat_flow_vphpl)
    _, k = busiest_approach_flow(cfg1)
    lam = x * cap_formula / k
    sc = Scenario(demand_level="saturated", lambda_base_vph=lam, controller="fixed_time")
    r = World(cfg1, sc, seed=seed).run().episode_row()
    cap = r["throughput_vph"] / 4.0
    return dict(capacity_vph=cap, capacity_formula_vph=cap_formula, capacity_realisation=cap / cap_formula,
                probe_x=x, probe_lambda_base_vph=lam, probe_throughput_vph=r["throughput_vph"],
                probe_main_s=main_s, probe_gridlock=bool(r["gridlock"]),
                probe_n_invariant_violations=int(r["n_invariant_violations"]))


def calibrate_demand(cfg: SimConfig, sat_flow_vphpl: float | None = None, seed: int = 0,
                     capacity_vph: float | None = None) -> dict:
    """Fill cfg.measured_sat_flow_vphpl, cfg.startup_lost_time_s, the three capacity fields
    and cfg.lambda_base_vph[level] = v/c_target * capacity_vph / (flow on the busiest
    internal approach per unit lambda).  `capacity_vph` skips the probe episode (tests).
    Returns a summary dict."""
    if sat_flow_vphpl is None:
        m = measure_saturation_flow(cfg, seed=seed)
        sat_flow_vphpl = m["sat_flow_vphpl"]
        cfg.startup_lost_time_s = round(m["startup_lost_time_s"], 3)
    cfg.measured_sat_flow_vphpl = round(float(sat_flow_vphpl), 2)
    cap_formula = approach_capacity_vph(cfg, sat_flow_vphpl)
    if capacity_vph is None:
        probe = measure_capacity(cfg, sat_flow_vphpl, seed=seed)
        capacity_vph = probe["capacity_vph"]
    else:
        probe = {}
    cfg.capacity_formula_vph = round(float(cap_formula), 2)
    cfg.capacity_vph = round(float(capacity_vph), 2)
    cfg.capacity_realisation = round(float(capacity_vph) / cap_formula, 4)
    edge_id, k = busiest_approach_flow(cfg)
    cfg.lambda_base_vph = {level: round(x * cfg.capacity_vph / k, 3) for level, x in cfg.vc_targets.items()}
    return dict(measured_sat_flow_vphpl=cfg.measured_sat_flow_vphpl,
                capacity_vph=cfg.capacity_vph, capacity_formula_vph=cfg.capacity_formula_vph,
                capacity_realisation=cfg.capacity_realisation, probe=probe,
                busiest_edge=edge_id, flow_per_unit_lambda=k, lambda_base_vph=dict(cfg.lambda_base_vph))


def lambda_for_vc(cfg: SimConfig, vc: float, capacity_vph: float) -> float:
    """Arrival rate per entry that puts the busiest approach at the given v/c against
    the given (measured) capacity per approach."""
    _, k = busiest_approach_flow(cfg)
    return vc * capacity_vph / k


if __name__ == "__main__":
    cfg = SimConfig()
    m = measure_saturation_flow(cfg)
    print(f"saturation headway {m['h_sat']:.3f} s -> {m['sat_flow_vphpl']:.0f} veh/h/lane, "
          f"start-up lost time {m['startup_lost_time_s']:.2f} s")
    c = calibrate_demand(cfg, m["sat_flow_vphpl"])
    print(f"capacity per approach: formula {c['capacity_formula_vph']:.0f} veh/h, measured {c['capacity_vph']:.0f} veh/h "
          f"(realisation {c['capacity_realisation']:.3f}); lambda_base {c['lambda_base_vph']}")
