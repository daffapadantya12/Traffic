"""Stop-or-commit chains, the signal plant and the controllers."""

import math

from traffic_twin import Scenario, SimConfig, World
from traffic_twin.controllers import max_pressure_request
from traffic_twin.network import DIR_INDEX, shortest_route
from traffic_twin.signals import ALLRED, GREEN
from traffic_twin.vehicles import COMMITTED, STOPPING, Vehicle, sample_params


def _place(world, edge, lane, d_to_stop, v):
    """Put a through vehicle on `edge` at distance d_to_stop from the stop line."""
    cfg = world.cfg
    J = world.net.junctions[edge.junction_at_dest]
    dest = world.net.edges[J.exit_edges[{"N": "S", "S": "N", "E": "W", "W": "E"}[edge.approach_dir]]].to_node
    route_edges, route_moves = shortest_route(world.net, edge.from_node, dest, cfg)
    veh = Vehicle(world.next_vid, sample_params(world.rng_vehicles, cfg), edge.from_node, dest,
                  route_edges, route_moves, world.t)
    world.next_vid += 1
    world.n_spawn_requests += 1
    veh.edge, veh.lane, veh.s, veh.v, veh.enter_t = edge, lane, edge.stop_s - d_to_stop, v, world.t
    edge.lanes[lane].append(veh)
    edge.lanes[lane].sort(key=lambda x: -x.s)
    world.active.append(veh)
    return veh


def test_stop_or_commit_chain_each_vehicle_gets_its_own_latch():
    cfg = SimConfig(grid_x=1, grid_y=1, warmup_s=0, main_s=120, cooldown_s=0)
    world = World(cfg, Scenario(lambda_base_vph=0.0, controller="fixed_time"), seed=1)
    J = world.net.junctions[0]
    plant = world.plants[0]
    assert plant.state == GREEN and plant.phase == DIR_INDEX["N"]
    edge = world.net.edges[J.approach_edges["N"]]
    fast_near = _place(world, edge, 0, 10.0, 12.0)      # needs 24 m to stop at 3 m/s^2 -> COMMITTED
    follower = _place(world, edge, 0, 60.0, 12.0)       # can stop -> its own STOPPING latch
    stopped = _place(world, edge, 1, 30.0, 0.0)         # standing vehicle -> STOPPING
    world.stop_or_commit(J, DIR_INDEX["N"])
    assert fast_near.latch == COMMITTED and fast_near.n_red_commits == 1
    assert follower.latch == STOPPING
    assert stopped.latch == STOPPING
    # a vehicle that enters the approach after red onset defaults to STOPPING
    plant.state = ALLRED
    plant.pending_phase = DIR_INDEX["E"]
    plant.allred_start_t = world.t
    late = world.new_request(edge.from_node, world.t)
    late.route_edges, late.route_moves = fast_near.route_edges, fast_near.route_moves
    assert world.try_insert(late, edge.from_node)
    assert late.latch == STOPPING
    # run: the committed vehicle crosses during all-red without a violation, the
    # follower stops before the stop line, the latecomer never crosses
    for _ in range(40):
        world.step()
    assert world.red_violations == 0
    assert follower.edge is edge and follower.s < edge.stop_s and follower.v < 0.1
    assert late.edge is edge and late.s < edge.stop_s
    assert fast_near.edge is not edge                   # it went through the box


def test_plant_timing_rules():
    cfg = SimConfig()
    world = World(cfg.copy(grid_x=1, grid_y=1, warmup_s=0, main_s=60, cooldown_s=0),
                  Scenario(lambda_base_vph=0.0, controller="fixed_time"), seed=0)
    plant = world.plants[0]
    assert plant.at_decision_instant(0.0) is False
    plant.time_in_phase = cfg.min_green
    assert plant.at_decision_instant(cfg.min_green)
    applied, forced, switched, red = plant.request(1, [0, 0, 0, 0], cfg.min_green)
    assert switched and red and plant.state == ALLRED and applied == 1
    assert not plant.try_green_onset(cfg.min_green + 2.0, True)       # all-red shorter than 4 s
    assert not plant.try_green_onset(cfg.min_green + 4.0, False)      # box not empty: extend
    assert plant.try_green_onset(cfg.min_green + 5.0, True)           # 5 s all-red -> extension 1 s
    assert plant.phase == 1 and plant.state == GREEN
    assert math.isclose(plant.allred_extensions[-1], 1.0)
    # max-green rule only forces when another approach has vehicles
    plant.time_in_phase = cfg.max_green
    applied, forced, switched, _ = plant.request(1, [0, 5, 0, 0], 100.0)
    assert not switched and forced == 0
    applied, forced, switched, _ = plant.request(1, [2, 5, 7, 7], 100.0)
    assert switched and forced == 2 and applied == 2                  # largest n, tie -> rotation order


def test_max_pressure_is_a_pure_function_of_the_row():
    cfg = SimConfig()
    row = {"phase": 0, "time_in_phase_s": 10.0}
    for a in "NESW":
        row[f"n_{a}"] = 0
    for d in "NESW":
        row[f"down_n_{d}"] = 0
        row[f"down_blocked_{d}"] = False
    assert max_pressure_request(row, cfg) == 0                       # nothing anywhere: keep
    row["n_E"] = 3
    assert max_pressure_request(row, cfg) == 1                       # 3 > 0 + 2 hysteresis
    row["n_E"] = 2
    assert max_pressure_request(row, cfg) == 0                       # not above the hysteresis
    row["n_E"] = 10
    row["down_n_N"] = row["down_n_S"] = row["down_n_W"] = 30         # E's exits are full
    row["n_N"] = 1
    assert max_pressure_request(row, cfg) == 0                       # pressure_E = 10 - 30 < pressure_N
    row["down_blocked_N"] = row["down_blocked_S"] = row["down_blocked_W"] = True
    assert max_pressure_request(row, cfg) == 0                       # all E exits blocked -> -1e9
    row["time_in_phase_s"] = cfg.max_green
    row["down_blocked_N"] = row["down_blocked_S"] = row["down_blocked_W"] = False
    assert max_pressure_request(row, cfg) in (1, 2, 3)               # forced away from the current phase
