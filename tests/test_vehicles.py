"""IDM, ballistic update and the parameter sampler."""

import math

import numpy as np

from traffic_twin import SimConfig
from traffic_twin.vehicles import ballistic_step, idm_accel, sample_params


def test_stop_exactly_branch_never_reverses():
    s, v = ballistic_step(10.0, 1.0, -4.0, 0.5)       # v + a dt = -1 < 0
    assert v == 0.0 and math.isclose(s, 10.0 + 1.0 / 8.0)
    s, v = ballistic_step(10.0, 2.0, -4.0, 0.5)       # v + a dt = 0 -> ordinary branch
    assert v == 0.0 and math.isclose(s, 10.0 + 2.0 * 0.5 - 0.5 * 4.0 * 0.25)
    s, v = ballistic_step(0.0, 10.0, 1.0, 0.5)
    assert math.isclose(s, 5.125) and math.isclose(v, 10.5)


def test_idm_free_road_and_equilibrium():
    a = idm_accel(0.0, 13.9, None, 0.0, 1.5, 2.0, 1.2, 2.0)
    assert math.isclose(a, 1.5)
    a = idm_accel(13.9, 13.9, None, 0.0, 1.5, 2.0, 1.2, 2.0)
    assert math.isclose(a, 0.0, abs_tol=1e-12)
    a = idm_accel(0.0, 13.9, 2.0, 0.0, 1.5, 2.0, 1.2, 2.0)   # standing at s0 behind a stopped leader
    assert math.isclose(a, 0.0, abs_tol=1e-12)


def test_platoon_brakes_without_overlap():
    """Ten vehicles at 13.9 m/s approach a stationary leader; gaps stay positive."""
    cfg = SimConfig()
    rng = np.random.default_rng(3)
    n = 10
    params = [sample_params(rng, cfg) for _ in range(n)]
    s = np.array([200.0 - 30.0 * i for i in range(n)])     # vehicle 0 is the front
    v = np.full(n, 13.9)
    wall = 260.0
    min_gap = math.inf
    for _ in range(200):
        a = np.empty(n)
        for i in range(n):
            if i == 0:
                gap, v_lead = wall - s[0], 0.0
            else:
                gap, v_lead = s[i - 1] - cfg.l_veh - s[i], v[i - 1]
            min_gap = min(min_gap, gap)
            a[i] = max(idm_accel(v[i], params[i]["v0"], gap, v_lead, params[i]["a_max"], params[i]["b"],
                                 params[i]["T"], cfg.s0), -cfg.b_emerg)
        for i in range(n):
            s[i], v[i] = ballistic_step(s[i], v[i], a[i], cfg.dt)
    assert min_gap > 0.3
    assert np.all(v < 0.05)


def test_sampled_parameters_are_clipped_and_not_capped_at_v_limit():
    cfg = SimConfig()
    rng = np.random.default_rng(0)
    ps = [sample_params(rng, cfg) for _ in range(2000)]
    v0 = np.array([p["v0"] for p in ps])
    assert v0.min() >= 0.85 * cfg.v_limit - 1e-9 and v0.max() <= 1.15 * cfg.v_limit + 1e-9
    assert (v0 > cfg.v_limit).mean() > 0.3          # a large share of drivers speed
    for key, lo, hi in [("a_max", 0.8, 2.5), ("b", 1.2, 3.0), ("T", 0.8, 1.8)]:
        x = np.array([p[key] for p in ps])
        assert x.min() >= lo - 1e-9 and x.max() <= hi + 1e-9
