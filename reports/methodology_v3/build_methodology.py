"""Fill the v3 methodology template with values read from the Fatmawati model and render it.

Run from the project root:  python3 reports/methodology_v3/build_methodology.py
Reads (read-only) the SUMO network, the vehicle types, the signal programme and the design constants of
the Fatmawati twin, fills {{PLACEHOLDERS}} in Methodology_template.md, and writes
Methodology_Fatmawati.{md,docx,html,pdf}.  Rendering uses RStudio's bundled pandoc and headless Chrome.
"""
from __future__ import annotations

import datetime
import math
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
PANDOC = "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64/pandoc"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DATA = os.path.join(ROOT, "data", "fatmawati")


def fmt(x, nd=0):
    return f"{x:,.{nd}f}"


def network_values() -> dict:
    import sumolib
    from traffic_twin.fatmawati import build_net as bn
    net = sumolib.net.readNet(os.path.join(DATA, "net", "fatmawati.net.xml"), withInternal=False, withPrograms=True)
    V = {}
    edges = net.getEdges()
    V["NET_EDGES"] = fmt(len(edges))
    V["NET_NODES"] = fmt(len(net.getNodes()))
    V["NET_TLS"] = fmt(len(net.getTrafficLights()))
    km = Counter()
    for e in edges:
        km[e.getType().replace("highway.", "")] += e.getLength() * e.getLaneNumber() / 1000.0
    order = [("motorway", "toll road (motorway)"), ("motorway_link", "toll ramps"), ("trunk", "trunk"), ("trunk_link", "trunk links"),
             ("primary", "primary"), ("primary_link", "primary links"), ("secondary", "secondary"), ("secondary_link", "secondary links"),
             ("tertiary", "tertiary"), ("tertiary_link", "tertiary links")]
    rows = ["| Road class | Lane-kilometres |", "|---|---|"]
    for k, lab in order:
        if km.get(k, 0) >= 0.05:
            rows.append(f"| {lab} | {km[k]:.1f} |")
    rows.append(f"| **total** | **{sum(km.values()):.1f}** |")
    V["LANE_KM_TABLE"] = "\n".join(rows)
    V["LANE_KM_TOTAL"] = f"{sum(km.values()):.0f}"
    jx, jy = net.convertLonLat2XY(bn.JUNCTION_LON, bn.JUNCTION_LAT)
    from traffic_twin.fatmawati import demand as dm
    hx, hy = net.convertLonLat2XY(*dm.HOSPITAL_LONLAT)
    V["HOSP_DIST_M"] = fmt(math.hypot(hx - jx, hy - jy))
    cs = net.getNode(bn.CROSSINGS[0]).getCoord(); cn = net.getNode(bn.CROSSINGS[1]).getCoord()
    V["CROSSING_GAP_M"] = fmt(math.hypot(cn[0] - cs[0], cn[1] - cs[1]))
    lanes = {}
    for arm, eid in bn.APPROACHES.items():
        try:
            lanes[arm] = net.getEdge(eid).getLaneNumber()
        except KeyError:
            lanes[arm] = None
    V["LANES_N"], V["LANES_E"], V["LANES_S"], V["LANES_W"] = (str(lanes[a]) for a in "NESW")
    tls = net.getTLS(bn.TLS_ID)
    V["TLS_LINKS"] = str(len(tls.getConnections()))
    V["RHK_DEPTH_M"] = fmt(bn.RHK_DEPTH_M)
    # signal timings are written in the template (the city's four-stage plan), not read from the model
    with open(os.path.join(DATA, "osm", "bbox.txt")) as f:
        s, w, n, e = (float(v) for v in f.read().strip().split(","))
    V["BOX_KM"] = f"{(n - s) * 110.6:.1f}"
    V["HALF_BOX_KM"] = f"{(n - s) * 110.6 / 2:.1f}"
    V["LAT0"], V["LON0"] = f"{bn.JUNCTION_LAT:.5f}", f"{bn.JUNCTION_LON:.5f}"
    # demand constants are written in the template (counted movements, level factors); nothing to read here
    # vehicle-type table
    vt = ET.parse(os.path.join(DATA, "demand", "vtypes.add.xml")).getroot()
    want = [("car_a5", "car (α = 0.5)"), ("moto_a5", "motorcycle (α = 0.5)"), ("bus", "bus / angkot"), ("truck", "truck"), ("ambulance", "ambulance")]
    by_id = {t.get("id"): t.attrib for t in vt.iter("vType")}
    rows = ["| Class | Length × width (m) | Min. gap (m) | Max. accel. (m/s²) | Comfort decel. (m/s²) | Headway τ (s) | Desired-speed factor | Lateral behaviour |",
            "|---|---|---|---|---|---|---|---|"]
    for vid, lab in want:
        a = by_id.get(vid)
        if not a:
            continue
        sf = a.get("speedFactor", "1")
        m = re.match(r"normc\(([\d.]+),([\d.]+),([\d.]+),([\d.]+)\)", sf)
        sf_txt = f"N({float(m.group(1)):.2f}, {float(m.group(2)):.2f}) in [{float(m.group(3)):.2f}, {float(m.group(4)):.2f}]" if m else sf
        lat = "filters anywhere in the lane" if a.get("latAlignment") == "arbitrary" else "keeps to the lane centre"
        rows.append(f"| {lab} | {float(a['length']):.1f} × {float(a['width']):.1f} | {float(a['minGap']):.1f} | {float(a['accel']):.1f} | "
                    f"{float(a['decel']):.1f} | {float(a['tau']):.1f} | {sf_txt} | {lat} |")
    V["VTYPE_TABLE"] = "\n".join(rows)
    return V


def main():
    V = network_values()
    V["TODAY"] = datetime.date.today().isoformat()
    V["SUBLANE_M"] = "0.8"          # SUMO lateral resolution used by the twin (sublane strip width, m)
    tpl = open(os.path.join(HERE, "Methodology_template.md"), encoding="utf-8").read()
    missing = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", tpl)) - set(V))
    if missing:
        print("UNFILLED placeholders:", missing)
    text = tpl
    for k, v in V.items():
        text = text.replace("{{" + k + "}}", str(v))
    out_md = os.path.join(HERE, "Methodology_Fatmawati.md")
    open(out_md, "w", encoding="utf-8").write(text)
    css = [os.path.join(ROOT, "reports", "_build", "style.css"), os.path.join(ROOT, "reports", "_build", "methodology.css")]
    base = ["--from", "markdown+smart+tex_math_dollars", "--resource-path", HERE, "--toc", "--toc-depth=2"]
    subprocess.run([PANDOC, out_md, "-o", os.path.join(HERE, "Methodology_Fatmawati.docx")] + base, check=True, cwd=HERE)
    html = os.path.join(HERE, "Methodology_Fatmawati.html")
    cmd = [PANDOC, out_md, "-o", html, "--standalone", "--embed-resources", "--section-divs", "--mathml"] + base
    for c in css:
        cmd += ["--css", c]
    subprocess.run(cmd, check=True, cwd=HERE)
    pdf = os.path.join(HERE, "Methodology_Fatmawati.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={pdf}", f"file://{html}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in ("Methodology_Fatmawati.docx", "Methodology_Fatmawati.pdf"):
        p = os.path.join(HERE, f)
        print(f, f"{os.path.getsize(p)/1e6:.1f} MB" if os.path.exists(p) else "MISSING")


if __name__ == "__main__":
    main()
