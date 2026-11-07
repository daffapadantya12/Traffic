"""The site specifications in data/sites/ must match the constants the Fatmawati pipeline actually uses
(docs/REPRODUCIBILITY.md), and the Bundaran HI skeleton must name a to-do for every missing value."""

import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITES = os.path.join(ROOT, "data", "sites")


def load(slug):
    with open(os.path.join(SITES, f"{slug}.json"), encoding="utf-8") as f:
        return json.load(f)


def test_fatmawati_spec_matches_the_network_and_demand_constants():
    from traffic_twin.fatmawati import build_net as bn, demand as dm

    s = load("fatmawati")
    assert (s["site"]["centre_lat"], s["site"]["centre_lon"]) == (bn.JUNCTION_LAT, bn.JUNCTION_LON)
    assert tuple(s["site"]["osm_road_types"]) == bn.ROAD_TYPES
    j = s["junction"]
    assert j["tls_id"] == bn.TLS_ID and tuple(j["crossings"]) == bn.CROSSINGS
    assert j["approaches"] == bn.APPROACHES and j["exits"] == bn.EXITS and j["median"] == bn.MEDIAN
    assert j["leg_source"] == dm.LEG_SOURCE and j["leg_sink"] == dm.LEG_SINK
    assert j["rhk_depth_m"] == bn.RHK_DEPTH_M
    g = s["signal"]
    assert tuple(g["stages"]) == bn.STAGES and g["green_s"] == bn.DISHUB_GREEN_S
    assert (g["yellow_s"], g["allred_s"]) == (bn.YELLOW_S, bn.ALLRED_S)
    assert g["cycle_s"] == sum(bn.DISHUB_GREEN_S.values()) + len(bn.STAGES) * (bn.YELLOW_S + bn.ALLRED_S)
    assert g["stage_of_approach"] == bn.STAGE_OF_APPROACH and g["stage_label"] == bn.STAGE_LABEL
    assert {k: tuple(v) for k, v in g["stage_approaches"].items()} == bn.STAGE_APPROACHES
    # stage links are written symbolically ("approach:W") so the spec reads without the raw ids
    ref = {"approach": bn.APPROACHES, "exit": bn.EXITS, "median": bn.MEDIAN}

    def resolve(tok):
        if tok == "*":
            return "*"
        kind, key = tok.split(":")
        return ref[kind][key]

    for st, pairs in g["stage_links"].items():
        assert [(resolve(a), resolve(b)) for a, b in pairs] == bn.STAGE_LINKS[st]
    d = s["demand"]
    assert {tuple(k.split(">")): v for k, v in d["movements_veh_h"].items()} == dm.MOVEMENTS
    assert d["mix"] == dm.MIX and d["level_factor"] == dm.LEVEL_FACTOR
    assert d["base_inflow_vph_at_design"] == dm.BASE_INFLOW_VPH["design"] and d["cordon_m"] == dm.CORDON_M
    assert d["ev_per_hour"] == dm.EV_PER_HOUR and tuple(d["ev_legs"]) == dm.EV_LEGS
    assert tuple(s["evaluation"]["hospital_lonlat"]) == dm.HOSPITAL_LONLAT


def test_fatmawati_spec_matches_the_sensor_calibration_and_evaluation_constants():
    pytest.importorskip("sumolib")                       # calibrate/evaluate import the SUMO world
    from traffic_twin.fatmawati import calibrate as cal, evaluate as ev, learn, sensors as sn, sumo_world as sw

    s = load("fatmawati")
    c = s["calibration"]
    assert c["target_max_queue_m"] == cal.TARGET_MAX_QUEUE_M and c["pkji_delay_s"] == cal.PKJI_DELAY_S
    assert c["pkji_ds"] == cal.PKJI_DS and c["vissim_junction_delay_s"] == cal.VISSIM_JUNCTION_DELAY_S
    assert tuple(c["grid"]) == cal.GRID and tuple(c["fit_seeds"]) == cal.SEEDS
    assert (c["warm_s"], c["main_s"]) == (cal.WARM_S, cal.MAIN_S)
    with open(os.path.join(ROOT, "data", "fatmawati", "calibration.json")) as f:
        fitted = json.load(f)
    assert c["fitted_scale"] == fitted["scale"] and c["capacity_limited"] == fitted["capacity_limited"]
    n = s["sensors"]
    assert n["gate_r_m"] == sn.GATE_R and n["camera_m"] == sn.CAMERA_M and n["exit_camera_m"] == sn.EXIT_CAMERA_M
    assert n["loop_setback_m"] == sn.LOOP_SETBACK_M and n["pcu"] == sn.PCU and n["study_m"] == sw.STUDY_M
    assert n["sat_flow_pcu_h"] == learn.SAT_FLOW_DEFAULT
    assert tuple(s["junction"]["legs"]) == sn.LEGS == sw.ARMS
    e = s["evaluation"]
    assert tuple(e["ev_moves"]) == ev.EV_MOVES and e["hospital_leg"] == ev.HOSPITAL_LEG
    assert (e["release_min_m"], e["slices"], e["release_every_s"]) == (ev.RELEASE_MIN_M, ev.SLICES, ev.RELEASE_EVERY_S)
    assert (e["warm_s"], e["tail_s"], e["extra_s"]) == (ev.WARM_S, ev.TAIL_S, ev.EXTRA_S)
    assert tuple(e["methods"]) == ev.METHODS and s["signal"]["cycle_s"] == ev.CYCLE_S
    p = s["signal"]["plant"]
    assert p["starvation_s"] == sw.STARVE_S
    from traffic_twin.fatmawati import kpi
    assert e["gates"]["general_delay_vs_max_pressure"] == kpi.GEN_GATE and e["gates"]["approach_vs_fixed"] == kpi.APPROACH_GATE


def test_fatmawati_spec_matches_the_released_configs():
    s = load("fatmawati")
    with open(os.path.join(ROOT, "data", "fatmawati_v3", "config.json")) as f:
        cfg = json.load(f)
    assert s["site"]["network_sha256"] == cfg["network_sha256"]
    assert s["demand"]["windows_h"] == cfg["windows"] and s["demand"]["level_factor"] == cfg["level_factor"]
    assert s["calibration"]["fitted_scale"] == cfg["calibration_scale"]
    with open(os.path.join(ROOT, "data", "fatmawati_eval", "evaluation_config.json")) as f:
        ecfg = json.load(f)
    assert s["evaluation"]["t_empty_s"] == ecfg["t_empty_s"]
    assert s["evaluation"]["release_points"] == ecfg["release_points"]
    assert s["evaluation"]["grid_seeds"] == ecfg["seeds"]
    with open(os.path.join(ROOT, "data", "fatmawati", "osm", "bbox.txt")) as f:
        assert s["site"]["osm_bbox_s_w_n_e"] == [float(v) for v in f.read().strip().split(",")]


def test_bundaran_hi_skeleton_names_a_todo_for_every_missing_value():
    s = load("bundaran_hi")
    f = load("fatmawati")
    assert set(s) == (set(f) - {"sources"}) | {"known_differences_from_fatmawati", "public_sources_checked_2026_11_07"}
    nulls, missing_todo = [], []
    for group, body in s.items():
        if not isinstance(body, dict):
            continue
        for k, v in body.items():
            if v is None:
                nulls.append(f"{group}.{k}")
                if not isinstance(body.get(k + "_todo"), str) or not body[k + "_todo"].strip():
                    missing_todo.append(f"{group}.{k}")
    assert nulls, "the skeleton should be honest about what is unknown"
    assert not missing_todo, f"null fields without a *_todo note: {missing_todo}"
    assert "unverified" in s["status"]
    # every value the pipeline needs (a non-note field of the Fatmawati spec) is present, filled or null
    for group in ("junction", "signal", "demand", "calibration", "sensors", "evaluation"):
        needed = {k for k in f[group] if not k.endswith("_note") and k not in ("lanes_at_stop_line", "n_sensors", "ev_routes",
                                                                                "primary_regime", "capacity_limited", "rel_error")}
        assert needed <= set(s[group]), f"{group} lacks {sorted(needed - set(s[group]))}"
