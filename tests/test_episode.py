"""Whole-episode checks: the six safety invariants, conservation, determinism,
label recomputation and the config variants (1 lane, right-hand traffic)."""

import math

import pandas as pd
import pytest

from traffic_twin import Scenario, SimConfig, World
from traffic_twin.controllers import label_request
from traffic_twin.features import SIGNAL_TICK_COLUMNS

SHORT = dict(warmup_s=60.0, main_s=240.0, cooldown_s=60.0)


def _run(cfg, scenario, seed=7):
    return World(cfg, scenario, seed=seed, episode_id=seed - 100000 if seed > 100000 else 0).run()


def _assert_invariants(world):
    inv = world.invariant_counts()
    assert inv["two_green_violations"] == 0, inv          # (1) never two approaches green
    assert inv["box_clear_violations"] == 0, inv          # (2) box empty at every green onset
    assert inv["committed_at_onset_violations"] == 0, inv  # (2b) no COMMITTED vehicle still upstream at a green onset
    assert inv["red_violations"] == 0, inv                # (3) no red crossing unless COMMITTED during its own all-red
    assert inv["min_green_violations"] == 0, inv          # (4) min green respected
    assert inv["allred_short_violations"] == 0, inv       # (5) all-red >= 4 s
    assert inv["conservation_error"] == 0, inv            # (6) spawned = arrived + in network + queued
    for p in world.plants:                                # (4)/(5) re-checked from the plant's own logs
        assert all(d + 1e-9 >= world.cfg.allred_min for d in p.allred_durations)
    # every logged decision row has time_in_phase >= min_green and exactly one green approach
    ticks = pd.DataFrame(world.signal_tick_rows())
    assert (ticks.time_in_phase_s >= world.cfg.min_green - 1e-9).all()
    assert set(ticks.phase.unique()) <= {0, 1, 2, 3}
    r = world.episode_row()
    assert r["n_spawned"] == r["n_arrived"] + r["n_in_network_end"] + r["n_entry_queued_end"]
    assert r["n_invariant_violations"] == 0


@pytest.mark.parametrize("grid", [(1, 1), (2, 2)])
@pytest.mark.parametrize("controller", ["fixed_time", "max_pressure", "random_dwell"])
def test_invariants_short_episode(grid, controller):
    cfg = SimConfig(grid_x=grid[0], grid_y=grid[1], **SHORT)
    world = _run(cfg, Scenario(lambda_base_vph=300.0, controller=controller))
    assert world.n_spawn_requests > 20
    _assert_invariants(world)
    assert world.n_overlap_fixed / max(world.vehicle_steps, 1) < cfg.overlap_rate_limit * 5


def test_determinism_same_seed_identical_tables(cfg_2x2, scenario):
    a = _run(cfg_2x2, scenario, seed=100005)
    b = _run(cfg_2x2, scenario, seed=100005)
    pd.testing.assert_frame_equal(pd.DataFrame(a.vehicle_rows()), pd.DataFrame(b.vehicle_rows()))
    pd.testing.assert_frame_equal(pd.DataFrame(a.signal_tick_rows()), pd.DataFrame(b.signal_tick_rows()))
    c = _run(cfg_2x2, scenario, seed=100006)
    assert not pd.DataFrame(a.vehicle_rows()).equals(pd.DataFrame(c.vehicle_rows()))


def test_labels_recompute(cfg_2x2):
    sc = Scenario(lambda_base_vph=260.0, controller="max_pressure_eps", controller_eps=0.3, sensor_noise_level=0.1)
    world = _run(cfg_2x2, sc)
    ticks = pd.DataFrame(world.signal_tick_rows())
    assert list(ticks.columns) == SIGNAL_TICK_COLUMNS
    # y_mp_action is a pure function of the row's own columns
    for _, row in ticks.iterrows():
        assert label_request(row.to_dict(), cfg_2x2) == row.y_mp_action
    # y_q_total_t30 equals the junction_1hz q_total at the first whole second >= t + 30
    j1 = pd.DataFrame(world.junction_1hz_rows).set_index(["junction_id", "t"]).q_total
    n_checked = 0
    for _, row in ticks.iterrows():
        if row.is_tail or row.is_cooldown:
            assert math.isnan(row.y_q_total_t30)
            continue
        key = (row.junction_id, math.ceil(row.t + cfg_2x2.label_horizon_s - 1e-9))
        if key in j1.index:
            assert row.y_q_total_t30 == j1.loc[key]
            n_checked += 1
    assert n_checked > 50
    # the sensor copy differs from the truth somewhere at noise 0.1 and is bounded sensibly
    assert (ticks.obs_q_N != ticks.q_N).any()
    # the eps controller sometimes requests something other than the label
    assert (ticks.action_requested != ticks.y_mp_action).any()


def test_sensor_copy_equals_truth_at_zero_noise(cfg_1x1, scenario):
    world = _run(cfg_1x1, scenario)
    ticks = pd.DataFrame(world.signal_tick_rows())
    for a in "NESW":
        assert (ticks[f"obs_q_{a}"] == ticks[f"q_{a}"]).all()
        assert (ticks[f"q_{a}_l0"] + ticks[f"q_{a}_l1"] == ticks[f"q_{a}"]).all()
    assert not ticks.nb_N_exists.any() and (ticks.nb_N_phase == -1).all()


def test_single_lane_still_runs():
    cfg = SimConfig(lanes_per_dir=1, **SHORT)
    world = _run(cfg, Scenario(lambda_base_vph=200.0, controller="longest_queue"))
    _assert_invariants(world)
    ticks = pd.DataFrame(world.signal_tick_rows())
    assert (ticks.q_N_l0 == -1).all() and (ticks.q_N_l1 == -1).all()
    assert math.isclose(cfg.w_box, 19.0) and math.isclose(world.net.edges["J00_J10"].length, 181.0)
    assert world.net.edges["J00_J10"].storage == 27          # one lane: floor(181 / 6.5)
    assert world.n_lane_changes == 0                         # nothing to change into


def test_right_hand_traffic_still_runs():
    cfg = SimConfig(drive_side="right", **SHORT)
    world = _run(cfg, Scenario(lambda_base_vph=250.0, controller="max_pressure", od_pattern="uniform", demand_asym="ns_heavy"))
    _assert_invariants(world)
    assert len(world.finished) > 50


def test_vehicle_table_fields(cfg_2x2, scenario):
    world = _run(cfg_2x2, scenario)
    v = pd.DataFrame(world.vehicle_rows())
    done = v[v.completed]
    assert (done.travel_time_s > 0).all()
    assert (done.total_delay_s == (done.entry_queue_delay_s + done.travel_time_s - done.free_flow_time_s).round(3)).all() or \
        ((done.total_delay_s - (done.entry_queue_delay_s + done.travel_time_s - done.free_flow_time_s)).abs() < 1e-2).all()
    assert (done.n_turns <= done.n_junctions).all()
    assert v.veh_id.is_unique
