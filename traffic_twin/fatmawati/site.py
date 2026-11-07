"""Site specification: every site fact the pipeline uses, read from data/sites/<slug>.json (DECISIONS §13.13).

The pipeline modules bind their constants from `SITE` when they are imported, so the site is chosen before any of
them loads, in this order: a `--site <slug>` (or `--site=<slug>`) argument on the command line, the `TWIN_SITE`
environment variable, else "fatmawati".  The chosen slug is written back into `TWIN_SITE`, so worker processes
(ProcessPoolExecutor, which re-imports the modules) load the same site.  Every CLI also declares `--site` through
`add_site_argument()`, for `--help` and so that argparse accepts it.

`load_site()` checks the spec before anything runs: a required field that is `null` (or missing) stops the load with
a message naming the field and its `*_todo` note, so a half-filled skeleton such as `data/sites/bundaran_hi.json`
fails at once rather than inside netconvert or SUMO.  Stage links are written symbolically in the spec
(`"approach:W"`, `"exit:N"`, `"median:south"`, `"*"`) and resolved here to raw edge ids.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SITES_DIR = os.path.join(ROOT, "data", "sites")
DEFAULT_SLUG = "fatmawati"
ENV_VAR = "TWIN_SITE"

# the fields a run needs, in the order the transfer protocol fills them (data/sites/README.md "Field reference");
# everything else (notes, published validation numbers, fitted results, viewer-only data) may be absent or null
REQUIRED = (
    ("site", ("centre_lat", "centre_lon", "radius_km", "osm_extract_file", "osm_road_types", "driving_side", "typemap",
              "network_file")),
    ("junction", ("tls_id", "program_id", "crossings", "legs", "street_names", "approaches", "exits", "median", "leg_source",
                  "leg_sink", "deleted_connections", "rhk_depth_m")),
    ("signal", ("stages", "green_s", "yellow_s", "allred_s", "stage_links", "stage_approaches", "stage_of_approach",
                "stage_label", "plant")),
    ("demand", ("movements_veh_h", "mix", "level_factor", "base_inflow_vph_at_design", "cordon_m", "windows_h", "ev_per_hour",
                "ev_legs")),
    ("calibration", ("target_max_queue_m", "grid", "fit_seeds", "verify_seeds", "warm_s", "main_s", "result_file")),
    ("sensors", ("gate_r_m", "camera_m", "exit_camera_m", "loop_setback_m", "study_m", "pcu")),
    ("evaluation", ("hospital_name", "hospital_lonlat", "hospital_street_name", "hospital_leg", "ev_moves", "ev_stage",
                    "release_min_m", "slices", "release_every_s", "warm_s", "tail_s", "extra_s", "methods", "gates",
                    "grid_seeds", "release_seeds")),
)
PLANT_KEYS = ("min_green_s", "max_green_s", "decision_s", "starvation_s", "step_s")


class SiteSpecError(ValueError):
    """The site file cannot be used for a run (a required value is null, missing or inconsistent)."""


def slug_from_argv(argv=None) -> str | None:
    """The value of --site in argv (the CLI flag), or None."""
    argv = sys.argv[1:] if argv is None else argv
    for i, a in enumerate(argv):
        if a == "--site" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--site="):
            return a.split("=", 1)[1]
    return None


def current_slug() -> str:
    """--site on the command line, else $TWIN_SITE, else the default; recorded in $TWIN_SITE for worker processes."""
    slug = slug_from_argv() or os.environ.get(ENV_VAR) or DEFAULT_SLUG
    os.environ[ENV_VAR] = slug
    return slug


def spec_path(slug: str) -> str:
    return os.path.join(SITES_DIR, f"{slug}.json")


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


class Site:
    """The values of one site file, as the pipeline modules use them (raw ids resolved, tuples where the code
    indexes by position, paths absolute)."""

    def __init__(self, spec: dict, path: str, check: bool = True):
        self.spec, self.path = spec, path
        self.slug = spec.get("site", {}).get("slug") or os.path.splitext(os.path.basename(path))[0]
        if check:
            self.check()
        s, j, g, d, c, n, e = (spec.get(k, {}) for k in ("site", "junction", "signal", "demand", "calibration", "sensors", "evaluation"))
        v = spec.get("viewer_only", {})
        # -- site
        self.name, self.city = s.get("name", self.slug), s.get("city", "")
        self.centre_lat, self.centre_lon = s.get("centre_lat"), s.get("centre_lon")
        self.radius_km = s.get("radius_km")
        self.osm_road_types = tuple(s.get("osm_road_types") or ())
        self.driving_side = s.get("driving_side", "left")
        self.data = os.path.join(ROOT, "data", self.slug)
        self.osm_extract_file = _abs(s.get("osm_extract_file") or os.path.join("data", self.slug, "osm", f"{self.slug}_roads.osm.xml"))
        self.typemap = _abs(s.get("typemap") or os.path.join("data", self.slug, "patches", "jakarta.typ.xml"))
        self.network_file = _abs(s.get("network_file") or os.path.join("data", self.slug, "net", f"{self.slug}.net.xml"))
        self.network_sha256 = s.get("network_sha256")
        self.sources = tuple(spec.get("sources") or ())
        # -- junction
        self.tls_id, self.program_id = j.get("tls_id", self.slug), j.get("program_id", "dishub")
        self.crossings = tuple(j.get("crossings") or ())
        self.legs = tuple(j.get("legs") or ())
        self.street_names = dict(j.get("street_names") or {})
        self.approaches, self.exits, self.median = dict(j.get("approaches") or {}), dict(j.get("exits") or {}), dict(j.get("median") or {})
        self.leg_source, self.leg_sink = dict(j.get("leg_source") or {}), dict(j.get("leg_sink") or {})
        self.deleted_connections = [(self.resolve_link(a), self.resolve_link(b)) for a, b in (j.get("deleted_connections") or ())]
        self.rhk_depth_m = j.get("rhk_depth_m")
        # -- signal
        self.stages = tuple(g.get("stages") or ())
        self.green_s = dict(g.get("green_s") or {})
        self.yellow_s, self.allred_s = g.get("yellow_s"), g.get("allred_s")
        self.stage_links = {st: [(self.resolve_link(a), self.resolve_link(b)) for a, b in pairs]
                            for st, pairs in (g.get("stage_links") or {}).items()}
        self.stage_approaches = {st: tuple(a) for st, a in (g.get("stage_approaches") or {}).items()}
        self.stage_of_approach = dict(g.get("stage_of_approach") or {})
        self.stage_label = dict(g.get("stage_label") or {})
        self.plant = dict(g.get("plant") or {})
        self.cycle_s = (sum(self.green_s.values()) + len(self.stages) * (self.yellow_s + self.allred_s)
                        if self.green_s and self.yellow_s is not None and self.allred_s is not None else g.get("cycle_s"))
        # -- demand
        self.movements = {tuple(k.split(">")): v for k, v in (d.get("movements_veh_h") or {}).items()}
        self.mix = dict(d.get("mix") or {})
        self.mix_sensitivity = dict(d.get("mix_sensitivity") or {})
        self.level_factor = dict(d.get("level_factor") or {})
        self.base_inflow_vph_at_design = d.get("base_inflow_vph_at_design")
        self.cordon_m = d.get("cordon_m")
        self.windows_h = dict(d.get("windows_h") or {})
        self.ev_per_hour = d.get("ev_per_hour")
        self.ev_legs = tuple(d.get("ev_legs") or ())
        # -- calibration
        self.target_max_queue_m = dict(c.get("target_max_queue_m") or {})
        self.pkji_delay_s, self.pkji_ds = c.get("pkji_delay_s"), c.get("pkji_ds")
        self.vissim_junction_delay_s = c.get("vissim_junction_delay_s")
        self.grid = tuple(c.get("grid") or ())
        self.fit_seeds, self.verify_seeds = tuple(c.get("fit_seeds") or ()), tuple(c.get("verify_seeds") or ())
        self.cal_warm_s, self.cal_main_s = c.get("warm_s"), c.get("main_s")
        self.fitted_scale, self.capacity_limited = c.get("fitted_scale"), c.get("capacity_limited")
        self.calibration_file = _abs(c.get("result_file") or os.path.join("data", self.slug, "calibration.json"))
        # -- sensors
        self.gate_r_m = dict(n.get("gate_r_m") or {})
        self.camera_m, self.exit_camera_m, self.loop_setback_m = n.get("camera_m"), n.get("exit_camera_m"), n.get("loop_setback_m")
        self.study_m = n.get("study_m")
        self.pcu = dict(n.get("pcu") or {})
        self.sat_flow_pcu_h = dict(n["sat_flow_pcu_h"]) if n.get("sat_flow_pcu_h") else None
        # -- evaluation
        self.hospital_name = e.get("hospital_name")
        self.hospital_lonlat = tuple(e["hospital_lonlat"]) if e.get("hospital_lonlat") else None
        self.hospital_street_name, self.hospital_leg = e.get("hospital_street_name"), e.get("hospital_leg")
        self.ev_moves = tuple(e.get("ev_moves") or ())
        self.ev_stage = dict(e.get("ev_stage") or {})
        self.release_min_m, self.slices = e.get("release_min_m"), e.get("slices")
        self.release_every_s = e.get("release_every_s")
        self.eval_warm_s, self.tail_s, self.extra_s = e.get("warm_s"), e.get("tail_s"), e.get("extra_s")
        self.methods = tuple(e.get("methods") or ())
        gates = e.get("gates") or {}
        self.gen_gate, self.approach_gate = gates.get("general_delay_vs_max_pressure"), gates.get("approach_vs_fixed")
        self.grid_seeds, self.release_seeds = tuple(e.get("grid_seeds") or ()), tuple(e.get("release_seeds") or ())
        # -- viewer only (never in the dataset); null where a site has no such structure
        self.jorr_ends = {k: tuple(x) for k, x in v["jorr_ends"].items()} if v.get("jorr_ends") else None
        self.jorr_peak_vph = v.get("jorr_peak_vph_per_direction")
        self.jorr_mix = dict(v["jorr_mix"]) if v.get("jorr_mix") else None
        self.mrt_rail_z_m = v.get("mrt_rail_z_m")
        self.mrt_osm_file = _abs(v.get("mrt_osm_file") or os.path.join("data", self.slug, "osm", "mrt.json"))
        self.buildings_file = _abs(v.get("buildings_file") or os.path.join("data", self.slug, "osm", "buildings_1600m.json"))

    # -- symbolic links
    def resolve_link(self, token: str) -> str:
        """'approach:W' -> the W approach edge id; 'exit:N', 'median:south' likewise; '*' stays '*'."""
        if token == "*":
            return "*"
        kind, _, key = token.partition(":")
        table = {"approach": "approaches", "exit": "exits", "median": "median"}.get(kind)
        ids = self.spec.get("junction", {}).get(table) if table else None
        if not ids or key not in ids:
            raise SiteSpecError(f"{self.path}: cannot resolve stage link {token!r} (junction.{table or kind} has no {key!r})")
        return ids[key]

    # -- checks
    def check(self) -> None:
        """Raise SiteSpecError naming the first required field that is null or missing, then the inconsistencies."""
        spec, rel = self.spec, os.path.relpath(self.path, ROOT)
        for group, keys in REQUIRED:
            body = spec.get(group)
            if not isinstance(body, dict):
                raise SiteSpecError(f"{rel}: group {group!r} is missing")
            for k in keys:
                if body.get(k) is None:
                    todo = body.get(k + "_todo")
                    raise SiteSpecError(f"{rel}: {group}.{k} is null; to do: {todo}" if todo else f"{rel}: {group}.{k} is missing")
        j, g, d, n, e = spec["junction"], spec["signal"], spec["demand"], spec["sensors"], spec["evaluation"]
        legs = set(j["legs"])
        for name, table in (("junction.street_names", j["street_names"]), ("junction.approaches", j["approaches"]),
                            ("junction.exits", j["exits"]), ("junction.leg_source", j["leg_source"]), ("junction.leg_sink", j["leg_sink"]),
                            ("signal.stage_of_approach", g["stage_of_approach"]), ("sensors.gate_r_m", n["gate_r_m"]),
                            ("calibration.target_max_queue_m", spec["calibration"]["target_max_queue_m"])):
            if set(table) != legs:
                raise SiteSpecError(f"{rel}: {name} keys {sorted(table)} do not match junction.legs {sorted(legs)}")
        stages = list(g["stages"])
        for name, table in (("signal.green_s", g["green_s"]), ("signal.stage_links", g["stage_links"]),
                            ("signal.stage_approaches", g["stage_approaches"]), ("signal.stage_label", g["stage_label"])):
            if set(table) != set(stages):
                raise SiteSpecError(f"{rel}: {name} keys {sorted(table)} do not match signal.stages {stages}")
        if set(g["stage_of_approach"].values()) - set(stages):
            raise SiteSpecError(f"{rel}: signal.stage_of_approach names a stage outside signal.stages")
        missing = [k for k in PLANT_KEYS if g["plant"].get(k) is None]
        if missing:
            raise SiteSpecError(f"{rel}: signal.plant lacks {missing}")
        for st, pairs in g["stage_links"].items():
            for a, b in pairs:
                self.resolve_link(a), self.resolve_link(b)
        for a, b in j["deleted_connections"]:
            self.resolve_link(a), self.resolve_link(b)
        for k in d["movements_veh_h"]:
            a, _, b = k.partition(">")
            if a not in legs or b not in legs:
                raise SiteSpecError(f"{rel}: demand.movements_veh_h key {k!r} is not '<entry leg>><exit leg>'")
        for name, vals in (("demand.ev_legs", d["ev_legs"]), ("evaluation.ev_moves", e["ev_moves"]), ("evaluation.hospital_leg", [e["hospital_leg"]])):
            if set(vals) - legs:
                raise SiteSpecError(f"{rel}: {name} {vals} is not within junction.legs")
        if set(e["ev_stage"]) != set(e["ev_moves"]) or set(v for v in e["ev_stage"].values() if v is not None) - set(stages):
            raise SiteSpecError(f"{rel}: evaluation.ev_stage must name a stage (or null) for every leg of evaluation.ev_moves")
        for k in ("general_delay_vs_max_pressure", "approach_vs_fixed"):
            if e["gates"].get(k) is None:
                raise SiteSpecError(f"{rel}: evaluation.gates.{k} is null")

    def __repr__(self) -> str:
        return f"Site({self.slug!r}, {os.path.relpath(self.path, ROOT)})"


def load_site(slug: str | None = None, check: bool = True) -> Site:
    """The site file of `slug` (default: the current site).  check=False reads a skeleton without validating it."""
    slug = slug or current_slug()
    path = spec_path(slug)
    if not os.path.exists(path):
        have = sorted(os.path.splitext(f)[0] for f in os.listdir(SITES_DIR) if f.endswith(".json")) if os.path.isdir(SITES_DIR) else []
        raise SiteSpecError(f"no site file {os.path.relpath(path, ROOT)}; known sites: {have}")
    with open(path, encoding="utf-8") as f:
        return Site(json.load(f), path, check=check)


def add_site_argument(ap) -> None:
    """The --site flag of every CLI.  Its value is read before the modules load (see the module docstring); the
    argparse entry documents it and accepts it."""
    ap.add_argument("--site", default=SITE.slug, metavar="SLUG",
                    help=f"site file data/sites/<SLUG>.json (default: $TWIN_SITE or '{DEFAULT_SLUG}'; now '{SITE.slug}')")


try:
    SITE = load_site(current_slug())
except SiteSpecError as err:
    if slug_from_argv() is not None:                 # a CLI asked for this site: a one-line answer, not a traceback
        sys.exit(f"cannot run site {current_slug()!r}: {err}")
    raise
