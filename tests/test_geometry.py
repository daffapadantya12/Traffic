"""Network geometry: turn exits by driving side, lane offsets, storage, routing."""

import math

from traffic_twin import SimConfig
from traffic_twin.network import build_grid, exit_heading, lane_offset, movement_between, shortest_route


def test_southbound_near_side_turn_exits_east_under_left_hand_traffic():
    assert exit_heading((0, -1), "NS", "left") == (1.0, -0.0) or exit_heading((0, -1), "NS", "left") == (1.0, 0.0)
    net = build_grid(SimConfig(drive_side="left"))
    J = net.junctions[0]                       # J00; connectors are lane-level (approach, movement, in, out)
    assert J.connectors[("N", "NS", 0, 0)].out_edge == J.exit_edges["E"]
    assert J.connectors[("N", "FS", 1, 1)].out_edge == J.exit_edges["W"]
    assert J.connectors[("N", "T", 0, 0)].out_edge == J.exit_edges["S"]
    assert J.connectors[("N", "T", 1, 1)].out_edge == J.exit_edges["S"]


def test_southbound_near_side_turn_exits_west_under_right_hand_traffic():
    net = build_grid(SimConfig(drive_side="right"))
    J = net.junctions[0]
    assert J.connectors[("N", "NS", 0, 0)].out_edge == J.exit_edges["W"]
    assert J.connectors[("N", "FS", 1, 1)].out_edge == J.exit_edges["E"]


def test_movement_between_is_inverse_of_exit_heading():
    for side in ("left", "right"):
        for h in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            for m in ("T", "NS", "FS"):
                assert movement_between(h, exit_heading(h, m, side), side) == m
            assert movement_between(h, (-h[0], -h[1]), side) is None     # no U-turns


def test_lane_offsets_are_on_the_driving_side():
    cfg = SimConfig(drive_side="left", lanes_per_dir=2)
    # eastbound driver under left-hand traffic drives on the north (left) side
    ox, oy = lane_offset((1.0, 0.0), 0, cfg)        # kerb lane
    assert oy > 0 and math.isclose(oy, 3.5 * 1.5)
    ox, oy = lane_offset((1.0, 0.0), 1, cfg)        # median lane
    assert math.isclose(oy, 3.5 * 0.5)
    cfg_r = SimConfig(drive_side="right", lanes_per_dir=2)
    ox, oy = lane_offset((1.0, 0.0), 0, cfg_r)
    assert oy < 0
    # single lane sits at half a lane width from the axis
    ox, oy = lane_offset((0.0, 1.0), 0, SimConfig(lanes_per_dir=1))
    assert math.isclose(abs(ox), 1.75) and math.isclose(oy, 0.0)


def test_grid_sizes_and_storage():
    cfg = SimConfig()
    net = build_grid(cfg)
    assert len(net.nodes) == 12 and len(net.edges) == 24 and len(net.junctions) == 4
    # per approach: T0->0, T1->1, NS 0->{0,1}, FS 1->{1,0}: 6 lane-level paths, 24 per junction
    assert sum(len(J.connectors) for J in net.junctions) == 96
    assert math.isclose(cfg.w_box, 26.0)                    # road 14 m + two 6 m kerb returns
    internal = net.edges["J00_J10"]
    stub = net.edges["W0_J00"]
    assert math.isclose(internal.length, 174.0) and internal.storage_per_lane == 26
    assert math.isclose(stub.length, 137.0) and stub.storage_per_lane == 21
    assert math.isclose(internal.stop_s, 172.0)
    net1 = build_grid(SimConfig(grid_x=1, grid_y=1))
    assert len(net1.nodes) == 5 and len(net1.edges) == 8 and len(net1.junctions) == 1
    assert all(e.is_stub for e in net1.edges.values())


def test_routes_respect_turn_penalties_and_reach_destination():
    cfg = SimConfig()
    net = build_grid(cfg)
    edges, moves = shortest_route(net, "W0", "E0", cfg)
    assert edges == ["W0_J00", "J00_J10", "J10_E0"] and [m for _, m in moves] == ["T", "T"]
    for o in net.boundary_nodes:
        for d in net.boundary_nodes:
            if o != d:
                edges, moves = shortest_route(net, o, d, cfg)
                assert net.edges[edges[0]].from_node == o and net.edges[edges[-1]].to_node == d


def test_box_paths_are_tangent_arcs_with_true_lengths_and_curvature_caps():
    """Every lane-level path starts at its entry lane's box edge with the entry heading and ends at
    its exit lane's box edge with the exit heading; turns are quarter arcs (plus a straight lead-in
    when they end in the far lane), and the curvature speed cap is sqrt(a_lat r)."""
    for side in ("left", "right"):
        for n in (1, 2):
            cfg = SimConfig(drive_side=side, lanes_per_dir=n)
            net = build_grid(cfg)
            for J in net.junctions:
                for c in J.connectors.values():
                    e_in, e_out = net.edges[c.in_edge], net.edges[c.out_edge]
                    x0, y0, h0 = c.path.pose(0.0)
                    x1, y1, h1 = c.path.pose(c.length)
                    assert math.dist((x0, y0), e_in.lane_xy(c.in_lane, e_in.length, cfg)) < 1e-6
                    assert math.dist((x1, y1), e_out.lane_xy(c.out_lane, 0.0, cfg)) < 1e-6
                    assert math.isclose(math.cos(h0), e_in.heading[0], abs_tol=1e-9)
                    assert math.isclose(math.sin(h1), e_out.heading[1], abs_tol=1e-9)
                    if c.is_turn:
                        assert math.isclose(c.v_cap, min(cfg.v_limit, math.sqrt(cfg.a_lat_max * c.path.radius)))
                        assert math.isclose(c.length, c.path.lead_in + c.path.radius * math.pi / 2 + c.path.lead_out)
    cfg = SimConfig()
    J = build_grid(cfg).junctions[0]
    assert math.isclose(J.connectors[("S", "NS", 0, 0)].length, 7.75 * math.pi / 2)      # kerb turn, r = 6 + 1.75
    assert math.isclose(J.connectors[("S", "FS", 1, 1)].length, 14.75 * math.pi / 2)
    assert math.isclose(J.connectors[("S", "T", 0, 0)].length, 26.0)


def test_conflict_matrix_is_symmetric_and_paths_of_one_approach_never_conflict():
    for side in ("left", "right"):
        net = build_grid(SimConfig(drive_side=side))
        for (a, b) in net.conflicts:
            assert (b, a) in net.conflicts and a[0] != b[0]           # symmetric, cross-approach only
        for zid, foe in net.zone_foe.items():
            assert net.zone_foe[foe] == zid
        # opposing through traffic in its own lanes never conflicts (parallel, 3.5 m apart)
        assert (("N", "T", 0, 0), ("S", "T", 0, 0)) not in net.conflicts
        # a far-side turn crosses the opposing through lanes
        assert (("N", "FS", 1, 1), ("S", "T", 1, 1)) in net.conflicts
