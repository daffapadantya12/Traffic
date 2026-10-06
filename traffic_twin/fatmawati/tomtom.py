"""Live TomTom traffic flow next to the twin, display only (DECISIONS §13.8).

TomTom Flow Segment Data (Traffic API v4) gives the current and free-flow speed of the road segment nearest a
point.  The licence of the free tier shapes everything here (TomTom T&C, checked 2026-10-03):
- results may only be cached in a client within the response cache headers (11.4); this endpoint sends
  max-age=0, so a reading is shown as it arrives, kept in memory only until the next poll, and never written to
  disk, logs or the dataset;
- no derived databases and no mixing with our own or openly licensed (OSM) data (11.6.1, 20.2.3): the readings do
  not feed the simulation, the demand, the controllers or any ambulance decision;
- no use in systems critical to health and safety (11.6.9), and simulated values are never shown as TomTom's
  (11.2); the viewer labels them separately with the attribution "©TomTom";
- 20,000 free requests a month: polling runs only while a viewer is connected, every 60 s with the probes
  staggered, and stops at a monthly cap.  Only the request count is stored (data/fatmawati/tomtom_usage.json).

The key comes from the TOMTOM_API_KEY environment variable and never reaches the browser.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone

from .demand import DATA

ENDPOINT = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/18/json"
CAPTION_URL = "https://api.tomtom.com/map/2/copyrights/caption.json"
USAGE = os.environ.get("TOMTOM_USAGE_FILE", os.path.join(DATA, "tomtom_usage.json"))   # Cloud Run: on the bucket mount
MONTHLY_CAP = 18_000                     # of the 20,000 free Flow Segment Data requests a month
POLL_S = 60.0
STALE_S = 150.0


def probes(layout: dict) -> list:
    """One probe per carriageway: the kerb lane of each entry gate (traffic towards the junction) and exit gate."""
    out = []
    for s in layout["loops"]:
        if s["kind"] in ("entry_gate", "exit_gate") and s["lane"] == 0:
            out.append(dict(probe=f"{s['leg']}_{'in' if s['kind'] == 'entry_gate' else 'out'}", leg=s["leg"],
                            direction="in" if s["kind"] == "entry_gate" else "out", lat=s["lat"], lon=s["lon"],
                            street=s["street"]))
    return out


def redact(text: str, key: str | None) -> str:
    return text.replace(key, "<key>") if key else text


def http_get(url: str, timeout: float = 10.0) -> tuple:
    """(status, json body or None).  Separate so tests can replace it."""
    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "Tracking-ID": str(uuid.uuid4())})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None


class Usage:
    """Monthly request counter: counts only, never results."""

    def __init__(self, path: str = USAGE, cap: int = MONTHLY_CAP):
        self.path, self.cap = path, cap
        self.month, self.count = self._month(), 0
        try:
            with open(path) as f:
                d = json.load(f)
            if d.get("month") == self.month:
                self.count = int(d.get("requests", 0))
        except (OSError, ValueError):
            pass

    @staticmethod
    def _month() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def allow(self) -> bool:
        if self._month() != self.month:
            self.month, self.count = self._month(), 0
        return self.count < self.cap

    def add(self) -> None:
        self.count += 1
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(dict(month=self.month, requests=self.count, cap=self.cap,
                           note="request count only; no TomTom results are stored"), f)


class TomTomLive(threading.Thread):
    """Polls the probes while `active()` is true and keeps only the latest reading of each, in memory."""

    def __init__(self, probe_list: list, key: str | None = None, active=lambda: True, get=http_get,
                 usage: Usage | None = None, poll_s: float = POLL_S, sleep=time.sleep):
        super().__init__(daemon=True)
        self.probes, self.key, self.active, self.get = probe_list, key or os.environ.get("TOMTOM_API_KEY"), active, get
        self.usage = usage or Usage()
        self.poll_s, self.sleep = poll_s, sleep
        self.readings = {}                       # probe -> latest reading (memory only)
        self.status = "off: set TOMTOM_API_KEY to show live TomTom speeds" if not self.key else "starting"
        self.caption = "©TomTom"
        self.stopped = not self.key
        self.backoff = 0.0
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        """What the viewer gets: fresh readings only (older than STALE_S are dropped), the status and attribution."""
        now = time.time()
        with self.lock:
            fresh = {p: r for p, r in self.readings.items() if now - r["fetched"] <= STALE_S}
        return dict(status=self.status, caption=self.caption, readings=fresh, requests_this_month=self.usage.count)

    def fetch_caption(self) -> None:
        if not self.usage.allow():
            return
        self.usage.add()
        status, body = self.get(f"{CAPTION_URL}?key={self.key}")
        if status == 200 and isinstance(body, dict) and body.get("copyrightsCaption"):
            self.caption = str(body["copyrightsCaption"])

    def poll_once(self) -> None:
        """One round over the probes (staggered across the poll interval)."""
        gap = self.poll_s / max(len(self.probes), 1)
        for p in self.probes:
            if self.stopped or not self.active():
                return
            if not self.usage.allow():
                self.status = f"paused: monthly cap of {self.usage.cap} requests reached"
                return
            self.usage.add()
            try:
                status, body = self.get(f"{ENDPOINT}?key={self.key}&point={p['lat']},{p['lon']}")
            except Exception as e:                       # network trouble: keep going, readings go stale
                self.status = "stale: " + redact(str(e), self.key)[:120]
                self.sleep(gap)
                continue
            if status == 200 and isinstance(body, dict) and "flowSegmentData" in body:
                d = body["flowSegmentData"]
                with self.lock:
                    self.readings[p["probe"]] = dict(
                        leg=p["leg"], direction=p["direction"], current_kmh=d.get("currentSpeed"),
                        free_flow_kmh=d.get("freeFlowSpeed"), confidence=d.get("confidence"),
                        road_closure=bool(d.get("roadClosure", False)), frc=d.get("frc"), fetched=time.time())
                self.status, self.backoff = "live", 0.0
            elif status in (401, 403, 596):
                self.status, self.stopped = f"stopped: the key was refused (HTTP {status})", True
                return
            elif status == 400:
                self.status = f"probe {p['probe']} rejected (HTTP 400)"
            elif status == 429:
                self.backoff = min(900.0, max(30.0, 2 * self.backoff))
                self.status = f"rate limited, retrying in {self.backoff:.0f} s"
                self.sleep(self.backoff)
                return
            else:
                self.status = f"stale: HTTP {status}"
            self.sleep(gap)

    def run(self) -> None:
        if self.stopped:
            return
        self.fetch_caption()
        while not self.stopped:
            if self.active():
                self.poll_once()
            else:
                self.status = "idle: no viewer connected"
                self.sleep(5.0)
