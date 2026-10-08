"""Summarise the Fatmawati sensor release into small, readable tables.

Reads the CSVs in data/fatmawati_v3 (no simulator needed), prints each summary and
writes it to reports/tables/ as CSV, plus one Excel workbook when openpyxl is installed.

    python scripts/explore_dataset.py
    python scripts/explore_dataset.py --data data/fatmawati_pilot --out reports/tables_pilot
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

ARMS = ("N", "E", "S", "W")
LEVEL_ORDER = ["low", "design", "high"]
GRAIN = {  # what one row of each table means
    "sensors": "one sensor",
    "episodes": "one simulated hour (window x level x controller x seed)",
    "loops_1min": "one loop detector lane in one minute",
    "cameras_1min": "one camera in one minute",
    "ticks_5s": "the whole junction every 5 s (ML features + labels)",
    "signal_log": "one signal phase change",
    "turning_counts_15min": "one movement x vehicle class in 15 min",
    "ambulance_trips": "one ambulance trip",
    "simulation_qc": "engine diagnostics for one episode",
}


def load(data: str, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data, f"{name}.csv"), **kw)


def table_overview(data: str) -> pd.DataFrame:
    rows = []
    for name, grain in GRAIN.items():
        path = os.path.join(data, f"{name}.csv")
        head = pd.read_csv(path, nrows=0)
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        rows.append({"table": name, "rows": n, "columns": len(head.columns),
                     "MB": round(os.path.getsize(path) / 1e6, 1), "one row is": grain})
    return pd.DataFrame(rows)


def controller_comparison(ep: pd.DataFrame) -> pd.DataFrame:
    """Mean over windows and seeds, per demand level and controller."""
    cols = ["stopped_delay_s"] + [f"stopped_delay_s_{a}" for a in ARMS] + \
           ["fairness_ratio", "cordon_tt_mean_s", "ambulance_tt_mean_s", "entry_veh_h"]
    out = ep.groupby(["level", "controller"])[cols].mean().round(1).reset_index()
    out["level"] = pd.Categorical(out["level"], LEVEL_ORDER, ordered=True)
    return out.sort_values(["level", "stopped_delay_s"]).reset_index(drop=True)


def vehicle_mix(loops: pd.DataFrame) -> pd.DataFrame:
    stop = loops[loops["kind"] == loops["kind"].unique()[0]] if "stopline" not in set(loops["kind"]) \
        else loops[loops["kind"] == "stopline"]
    counts = stop[["n_motorcycle", "n_car", "n_bus", "n_truck", "n_emergency"]].sum()
    return pd.DataFrame({"vehicles": counts, "share_%": (100 * counts / counts.sum()).round(2)})


def ambulance_summary(amb: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    a = amb.merge(ep[["episode_id", "level", "controller"]], on="episode_id")
    g = a.groupby(["level", "controller"])
    out = pd.DataFrame({
        "trips": g.size(),
        "arrived_%": (100 * g["arrived"].mean()).round(1),
        "travel_time_mean_s": g["travel_time_s"].mean().round(1),
        "travel_time_p90_s": g["travel_time_s"].quantile(0.9).round(1),
        "recovery_mean_s": g["recovery_s"].mean().round(1),
    }).reset_index()
    out["level"] = pd.Categorical(out["level"], LEVEL_ORDER, ordered=True)
    return out.sort_values(["level", "travel_time_mean_s"]).reset_index(drop=True)


def label_balance(ticks: pd.DataFrame) -> pd.DataFrame:
    amb = ticks[[f"ev_{a}" for a in ARMS]].fillna(0).sum(axis=1) > 0
    out = pd.DataFrame({
        "all_ticks_%": (100 * ticks["y_teacher_stage"].value_counts(normalize=True)).round(1),
        "ambulance_ticks_%": (100 * ticks.loc[amb, "y_teacher_stage"].value_counts(normalize=True)).round(1),
    }).fillna(0)
    out.index.name = "teacher_stage"
    return out.reset_index()


def quality(qc: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    q = qc.merge(ep[["episode_id", "controller"]], on="episode_id")
    q["collisions_per_1000_vkm_study"] = 1000 * q["lane_collisions_study"] / q["veh_km_study"]
    cols = ["lane_collisions_study", "junction_overlaps_study", "teleports", "collisions_per_1000_vkm_study"]
    return q.groupby("controller")[cols].mean().round(2).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/fatmawati_v3")
    ap.add_argument("--out", default="reports/tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    ep = load(args.data, "episodes")
    tables = {
        "01_table_overview": table_overview(args.data),
        "02_controller_comparison": controller_comparison(ep),
        "03_vehicle_mix": vehicle_mix(load(args.data, "loops_1min")).reset_index().rename(columns={"index": "class"}),
        "04_ambulance_trips": ambulance_summary(load(args.data, "ambulance_trips"), ep),
        "05_teacher_label_balance": label_balance(load(args.data, "ticks_5s")),
        "06_simulation_quality": quality(load(args.data, "simulation_qc"), ep),
    }
    for name, df in tables.items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))
        df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)

    try:
        with pd.ExcelWriter(os.path.join(args.out, "summary.xlsx")) as xl:
            for name, df in tables.items():
                df.to_excel(xl, sheet_name=name[:31], index=False)
        print(f"\nWrote {len(tables)} CSVs and summary.xlsx to {args.out}/")
    except ImportError:
        print(f"\nWrote {len(tables)} CSVs to {args.out}/ (pip install openpyxl for the Excel workbook)")


if __name__ == "__main__":
    main()
