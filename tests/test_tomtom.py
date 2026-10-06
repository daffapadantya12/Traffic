"""Live TomTom flow is display only (DECISIONS §13.8): memory-only readings, counted requests, safe failures."""

import json

from traffic_twin.fatmawati.tomtom import TomTomLive, Usage, redact

PROBES = [dict(probe="E_in", leg="E", direction="in", lat=-6.2925, lon=106.7966, street="Jalan Tahi Bonar Simatupang"),
          dict(probe="E_out", leg="E", direction="out", lat=-6.2918, lon=106.7966, street="Jalan Tahi Bonar Simatupang")]
OK = {"flowSegmentData": {"frc": "FRC2", "currentSpeed": 18, "freeFlowSpeed": 45, "confidence": 0.9, "roadClosure": False}}


def _live(tmp_path, get, key="secret-key", cap=100, active=lambda: True):
    return TomTomLive(PROBES, key=key, active=active, get=get, usage=Usage(str(tmp_path / "usage.json"), cap=cap),
                      poll_s=0.0, sleep=lambda s: None)


def test_off_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("TOMTOM_API_KEY", raising=False)
    calls = []
    t = _live(tmp_path, lambda url: calls.append(url), key=None)
    t.run()
    assert t.stopped and not calls and t.snapshot()["readings"] == {} and t.status.startswith("off")


def test_readings_live_in_memory_and_only_counts_are_written(tmp_path):
    t = _live(tmp_path, lambda url: (200, OK))
    t.poll_once()
    snap = t.snapshot()
    assert snap["readings"]["E_in"]["current_kmh"] == 18 and snap["status"] == "live"
    stored = json.load(open(tmp_path / "usage.json"))
    assert set(stored) == {"month", "requests", "cap", "note"} and stored["requests"] == 2
    assert "18" not in json.dumps({k: v for k, v in stored.items() if k != "month"})


def test_refused_key_stops_polling(tmp_path):
    t = _live(tmp_path, lambda url: (401, None))
    t.poll_once()
    assert t.stopped and "refused" in t.status


def test_rate_limit_backs_off(tmp_path):
    t = _live(tmp_path, lambda url: (429, None))
    t.poll_once()
    assert t.backoff >= 30.0 and "rate limited" in t.status and not t.stopped


def test_key_never_shown(tmp_path):
    def boom(url):
        raise OSError(f"connection reset while fetching {url}")
    t = _live(tmp_path, boom)
    t.poll_once()
    assert "secret-key" not in t.status and "<key>" in t.status
    assert redact("a secret-key b", "secret-key") == "a <key> b"


def test_monthly_cap_and_idle(tmp_path):
    t = _live(tmp_path, lambda url: (200, OK), cap=1)
    t.poll_once()
    assert t.usage.count == 1 and "cap" in t.status
    calls = []
    idle = _live(tmp_path / "x", lambda url: calls.append(url) or (200, OK), active=lambda: False)
    (tmp_path / "x").mkdir()
    idle.poll_once()
    assert not calls
