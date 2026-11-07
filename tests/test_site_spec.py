"""The site files in data/sites/ drive the Fatmawati pipeline through traffic_twin/fatmawati/site.py (DECISIONS §13.13):
the loaded Fatmawati spec must equal the constants the modules bind and the released configs, and the Bundaran HI
skeleton must load unchecked, fail fast when checked, and name a to-do for every missing value."""

import json
import os

import pytest

from traffic_twin.fatmawati import site as site_mod

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SITES = os.path.join(ROOT, "data", "sites")


def raw(slug):
    with open(os.path.join(SITES, f"{slug}.json"), encoding="utf-8") as f:
        return json.load(f)


def test_site_selection_order(monkeypatch):
    assert site_mod.slug_from_argv(["--plan", "smoke", "--site", "x"]) == "x"
    assert site_mod.slug_from_argv(["--site=y"]) == "y"
    assert site_mod.slug_from_argv(["--plan", "smoke"]) is None
    monkeypatch.setattr(site_mod.sys, "argv", ["prog"])
    monkeypatch.setenv(site_mod.ENV_VAR, "bundaran_hi")
    assert site_mod.current_slug() == "bundaran_hi"
    monkeypatch.delenv(site_mod.ENV_VAR)
    assert site_mod.current_slug() == site_mod.DEFAULT_SLUG
    assert os.environ[site_mod.ENV_VAR] == site_mod.DEFAULT_SLUG        # recorded for worker processes
    with pytest.raises(site_mod.SiteSpecError, match="known sites"):
        site_mod.load_site("no_such_site")


def test_loader_resolves_symbolic_links_and_paths():
    s = site_mod.load_site("fatmawati")
    j = raw("fatmawati")["junction"]
    assert s.resolve_link("approach:W") == j["approaches"]["W"] and s.resolve_link("median:north") == j["median"]["north"]
    assert s.resolve_link("exit:N") == j["exits"]["N"] and s.resolve_link("*") == "*"
    with pytest.raises(site_mod.SiteSpecError):
        s.resolve_link("approach:X")
    assert s.stage_links["S1"] == [(j["approaches"]["W"], "*"), (j["approaches"]["E"], j["exits"]["W"])]
    assert s.deleted_connections[0] == (j["approaches"]["W"], j["median"]["south"])
    assert s.data == os.path.join(ROOT, "data", "fatmawati") and s.network_file.endswith(os.path.join("net", "fatmawati.net.xml"))
    assert s.calibration_file == os.path.join(ROOT, "data", "fatmawati", "calibration.json")
    assert s.cycle_s == 231.0 and s.movements[("N", "S")] == 3135 and s.legs == ("N", "E", "S", "W")


def test_fatmawati_spec_is_what_the_network_and_demand_modules_bind():
    from traffic_twin.fatmawati import build_net as bn, demand as dm

    s = site_mod.load_site("fatmawati")
    assert site_mod.SITE.slug == "fatmawati"                              # the site the modules were imported with
    assert (s.centre_lat, s.centre_lon) == (bn.JUNCTION_LAT, bn.JUNCTION_LON) and s.radius_km == bn.RADIUS_KM
    assert s.osm_road_types == bn.ROAD_TYPES and s.data == bn.DATA == dm.DATA and s.network_file == bn.NET == dm.NET
    assert s.tls_id == bn.TLS_ID and s.program_id == bn.PROGRAM_ID and s.crossings == bn.CROSSINGS
    assert s.approaches == bn.APPROACHES and s.exits == bn.EXITS and s.median == bn.MEDIAN
    assert s.deleted_connections == bn.DELETED_CONNECTIONS
    assert s.leg_source == dm.LEG_SOURCE and s.leg_sink == dm.LEG_SINK and s.legs == dm.LEGS
    assert s.rhk_depth_m == bn.RHK_DEPTH_M
    assert s.stages == bn.STAGES and s.green_s == bn.DISHUB_GREEN_S
    assert (s.yellow_s, s.allred_s) == (bn.YELLOW_S, bn.ALLRED_S)
    assert s.cycle_s == sum(bn.DISHUB_GREEN_S.values()) + len(bn.STAGES) * (bn.YELLOW_S + bn.ALLRED_S)
    assert s.stage_of_approach == bn.STAGE_OF_APPROACH and s.stage_label == bn.STAGE_LABEL
    assert s.stage_approaches == bn.STAGE_APPROACHES and s.stage_links == bn.STAGE_LINKS
    assert s.movements == dm.MOVEMENTS and s.mix == dm.MIX and s.level_factor == dm.LEVEL_FACTOR
    assert s.base_inflow_vph_at_design == dm.BASE_INFLOW_VPH["design"] and s.cordon_m == dm.CORDON_M
    assert s.ev_per_hour == dm.EV_PER_HOUR and s.ev_legs == dm.EV_LEGS
    assert s.hospital_lonlat == dm.HOSPITAL_LONLAT and s.hospital_street_name == dm.HOSPITAL_STREET_NAME
    assert s.hospital_name == dm.HOSPITAL_NAME and s.calibration_file == dm.CALIBRATION


def test_fatmawati_spec_is_what_the_sensor_calibration_evaluation_and_learning_modules_bind():
    pytest.importorskip("sumolib")                       # calibrate/evaluate import the SUMO world
    from traffic_twin.fatmawati import calibrate as cal, evaluate as ev, generate as gen, kpi, learn, sensors as sn, sumo_world as sw

    s = site_mod.load_site("fatmawati")
    assert s.target_max_queue_m == cal.TARGET_MAX_QUEUE_M and s.pkji_delay_s == cal.PKJI_DELAY_S
    assert s.pkji_ds == cal.PKJI_DS and s.vissim_junction_delay_s == cal.VISSIM_JUNCTION_DELAY_S
    assert s.grid == cal.GRID and s.fit_seeds == cal.SEEDS and s.verify_seeds == cal.VERIFY_SEEDS
    assert (s.cal_warm_s, s.cal_main_s) == (cal.WARM_S, cal.MAIN_S) and list(s.sources) == cal.SOURCES
    with open(s.calibration_file) as f:
        fitted = json.load(f)
    assert s.fitted_scale == fitted["scale"] and s.capacity_limited == fitted["capacity_limited"]
    assert s.gate_r_m == sn.GATE_R and s.camera_m == sn.CAMERA_M and s.exit_camera_m == sn.EXIT_CAMERA_M
    assert s.loop_setback_m == sn.LOOP_SETBACK_M and s.pcu == sn.PCU and s.study_m == sw.STUDY_M
    assert s.sat_flow_pcu_h == learn.SAT_FLOW_DEFAULT
    assert s.legs == sn.LEGS == sw.ARMS == learn.LEGS == kpi.LEGS == gen.LEGS
    assert s.stages == learn.STAGES and s.stage_of_approach == learn.STAGE_OF_APPROACH
    assert s.yellow_s + s.allred_s == learn.INTERGREEN_S == sw.INTERGREEN_S and s.plant["decision_s"] == learn.DECISION_S
    assert s.plant["starvation_s"] == sw.STARVE_S and s.plant == sw.PLANT
    assert s.ev_moves == ev.EV_MOVES == kpi.EV_MOVES and s.hospital_leg == ev.HOSPITAL_LEG and s.ev_stage == kpi.EV_STAGE
    assert (s.release_min_m, s.slices, s.release_every_s) == (ev.RELEASE_MIN_M, ev.SLICES, ev.RELEASE_EVERY_S)
    assert (s.eval_warm_s, s.tail_s, s.extra_s) == (ev.WARM_S, ev.TAIL_S, ev.EXTRA_S) and s.extra_s == kpi.EXTRA_S
    assert s.methods == ev.METHODS and s.cycle_s == ev.CYCLE_S
    assert (s.gen_gate, s.approach_gate) == (kpi.GEN_GATE, kpi.APPROACH_GATE)
    assert s.windows_h == gen.WINDOWS and list(s.release_seeds) == gen.PLANS["release"]["seeds"]
    from traffic_twin.fatmawati import jorr, mrt
    assert s.jorr_ends == jorr.ENDS and s.jorr_peak_vph == jorr.PEAK_VPH and s.jorr_mix == jorr.MIX
    assert s.mrt_rail_z_m == mrt.RAIL_Z and s.radius_km == mrt.RADIUS_KM and s.mrt_osm_file == mrt.OSM


def test_fatmawati_spec_matches_the_released_configs():
    s = raw("fatmawati")
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


def test_bundaran_hi_skeleton_fails_fast_on_its_first_missing_value():
    with pytest.raises(site_mod.SiteSpecError) as err:
        site_mod.load_site("bundaran_hi")
    msg = str(err.value)
    assert "bundaran_hi.json: junction.crossings is null" in msg and "to do:" in msg
    s = site_mod.load_site("bundaran_hi", check=False)              # readable as a skeleton
    assert s.slug == "bundaran_hi" and s.legs == ("N", "S", "E", "W") and s.crossings == () and s.jorr_ends is None
    assert s.data == os.path.join(ROOT, "data", "bundaran_hi") and s.sat_flow_pcu_h is None


def test_bundaran_hi_skeleton_names_a_todo_for_every_missing_value():
    s = raw("bundaran_hi")
    f = raw("fatmawati")
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
    # and every field site.REQUIRED checks exists in both files (null allowed only in the skeleton)
    for group, keys in site_mod.REQUIRED:
        for k in keys:
            assert k in s[group], f"{group}.{k} missing from the skeleton"
            assert f[group].get(k) is not None, f"{group}.{k} null in the Fatmawati spec"
