"""Evaluation grid bookkeeping (DECISIONS §13.10): schedule, recovery and extra wait on synthetic queues."""

import pytest

pytest.importorskip("sumolib")

from traffic_twin.fatmawati.evaluate import CYCLE_S, RELEASE_EVERY_S, SLICES, schedule, passage_rows  # noqa: E402

LEGS = ("N", "E", "S", "W")


def test_schedule_covers_every_slice_of_the_cycle():
    s = schedule()
    assert len(s) == 3 * SLICES
    for leg in ("N", "E", "W"):
        slices = sorted(k for _, g, k in s if g == leg)
        assert slices == list(range(SLICES))
    for t, _, k in s:                                    # release at the slice's offset in the fixed-time cycle
        assert abs((t % CYCLE_S) - k * CYCLE_S / SLICES) <= 0.5 or abs((t % CYCLE_S) - k * CYCLE_S / SLICES - CYCLE_S) <= 0.5
    gaps = [b[0] - a[0] for a, b in zip(s, s[1:])]
    assert min(gaps) >= RELEASE_EVERY_S - CYCLE_S and max(gaps) <= RELEASE_EVERY_S + CYCLE_S


def _run(q_level, bump_until=None, bump=0.0, end=3000):
    q = {g: {float(t): q_level + (bump if bump_until and 1000 <= t < bump_until else 0.0) for t in range(0, end)} for g in LEGS}
    stop = {g: [(float(t), 10) for t in range(0, end, 60)] for g in LEGS}
    return dict(level="design", seed=1, method="fixed", q=q, stop=stop, passages=[])


def test_recovery_and_extra_wait():
    with_ev, without = _run(10.0, bump_until=1300, bump=8.0), _run(10.0)
    with_ev["passages"] = [dict(passage=0, leg="N", slice=0, t_release=1000.0, t_exit_gate=1200.0, t_arrive=1250.0)]
    (row,) = passage_rows(with_ev, without, {"N": 50.0, "E": 40.0, "W": 170.0})
    assert row["junction_time_s"] == 200.0 and row["delay_s"] == 150.0 and row["travel_time_s"] == 250.0
    # the 231 s rolling mean is back within 1 vehicle once the bump has left the window, then holds for a cycle
    assert not row["recovery_censored"] and 100.0 <= row["recovery_s"] <= 100.0 + CYCLE_S
    # extra standing: 8 vehicles x 300 s on each approach over 600 s, 10 vehicles served per minute -> 100 served
    assert row["extra_wait_s_N"] == pytest.approx(8.0 * 300 / 100, rel=0.01)


def test_unfinished_passage_is_censored():
    with_ev, without = _run(10.0), _run(10.0)
    with_ev["passages"] = [dict(passage=0, leg="E", slice=3, t_release=2500.0, t_exit_gate=None, t_arrive=None)]
    (row,) = passage_rows(with_ev, without, {"N": 50.0, "E": 40.0, "W": 170.0})
    assert row["junction_time_s"] is None and row["recovery_censored"]


def test_longest_red_counts_the_through_lanes_of_tb_simatupang():
    from traffic_twin.fatmawati.evaluate import longest_red
    log = [dict(t=0.0, stage="S1", state="green"), dict(t=80.0, stage="S1>S2", state="amber"),
           dict(t=82.0, stage="S1>S2", state="all-red"), dict(t=84.0, stage="S2", state="green"),
           dict(t=96.0, stage="S2>S3", state="amber"), dict(t=98.0, stage="S2>S3", state="all-red"),
           dict(t=100.0, stage="S3", state="green"), dict(t=178.0, stage="S3>S4", state="amber"),
           dict(t=182.0, stage="S4", state="green"), dict(t=227.0, stage="S4>S1", state="amber"),
           dict(t=231.0, stage="S1", state="green")]
    assert longest_red(log, "E", 0.0, 300.0) == 135.0       # 96 (S2 amber) .. 231 (S1 green)
    assert longest_red(log, "W", 0.0, 300.0) == 151.0       # 80 .. 231
    assert longest_red(log, "N", 0.0, 300.0) == 122.0       # 0 .. 100 and 178 .. 300: the longer one


def test_passages_without_a_reference_run():
    with_ev = _run(10.0)
    with_ev["passages"] = [dict(passage=0, leg="W", slice=1, t_release=1000.0, t_exit_gate=1300.0, t_arrive=1400.0)]
    (row,) = passage_rows(with_ev, None, {"N": 50.0, "E": 40.0, "W": 170.0})
    assert row["junction_time_s"] == 300.0 and row["delay_s"] == 130.0
    assert row["recovery_s"] is None and row["recovery_censored"] is None and row["extra_wait_s_N"] is None
