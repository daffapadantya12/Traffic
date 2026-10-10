import os
import traci
import pandas as pd

APPROACHES = ["N2C", "E2C", "S2C", "W2C"]
MIN_GREEN = 10       # s  safety: never switch sooner
MAX_GREEN = 60       # s  fairness: switch if others are waiting this long
MARGIN = 3           # vehicles: stops the lights flickering
DECIDE_EVERY = 5     # s  the controller decides every 5 s
AMB_RANGE = 150      # m  ambulance detection distance


def start(seed, tripfile, gui=False):
    """Start SUMO and work out which lanes each green phase serves."""
    cmd = ["sumo-gui" if gui else "sumo", "-n", "junction.net.xml", "-r", "traffic.rou.xml",
           "--lateral-resolution", "0.8", "--seed", str(seed),
           "--tripinfo-output", tripfile, "--no-step-log", "true", "--no-warnings", "true"]
    if gui:
        cmd += ["--start", "--delay", "50"]
    traci.start(cmd)
    tls = traci.trafficlight.getIDList()[0]
    phases = traci.trafficlight.getAllProgramLogics(tls)[0].phases
    links = traci.trafficlight.getControlledLinks(tls)
    greens = {}
    for i, p in enumerate(phases):
        if "y" in p.state or "G" not in p.state:
            continue
        greens[i] = {links[k][0][0] for k, c in enumerate(p.state) if c in "Gg" and links[k]}
    return tls, phases, greens


def queue(lanes):
    """Standing vehicles on these lanes (the queue camera)."""
    return sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)


def ambulance_target(greens):
    """The green phase the nearest ambulance within AMB_RANGE needs, or None."""
    best = None
    for v in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(v) != "ambulance":
            continue
        lane = traci.vehicle.getLaneID(v)
        for g, lanes in greens.items():
            if lane in lanes:
                dist = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(v)
                if dist <= AMB_RANGE and (best is None or dist < best[1]):
                    best = (g, dist)
                break
    return None if best is None else best[0]


def run(controller, seed, gui=False):
    os.makedirs("out", exist_ok=True)
    trip = f"out/tripinfo_{controller}_{seed}.xml"
    tls, phases, greens = start(seed, trip, gui)
    g = traci.trafficlight.getPhase(tls)
    mode, since, target = "green", 0.0, None
    if controller != "fixed":
        traci.trafficlight.setPhaseDuration(tls, 10_000)

    rows = []
    while traci.simulation.getTime() < 3600:
        traci.simulationStep()
        t = traci.simulation.getTime()

        if controller == "fixed":
            pass

        elif mode == "yellow":
            if t - since >= phases[g + 1].duration:
                g, mode, since = target, "green", t
                traci.trafficlight.setPhase(tls, g)
                traci.trafficlight.setPhaseDuration(tls, 10_000)

        elif t % DECIDE_EVERY == 0 and t - since >= MIN_GREEN:
            q = {p: queue(lanes) for p, lanes in greens.items()}
            others = {p: v for p, v in q.items() if p != g}
            want = max(others, key=others.get)
            switch = others[want] > q[g] + MARGIN or (t - since >= MAX_GREEN and others[want] > 0)

            if controller == "priority":
                amb = ambulance_target(greens)
                if amb is not None:
                    want, switch = amb, amb != g

            if switch:
                target, mode, since = want, "yellow", t
                traci.trafficlight.setPhase(tls, g + 1)
                traci.trafficlight.setPhaseDuration(tls, 10_000)

        if t % 5 == 0:
            row = {"controller": controller, "seed": seed, "t": t, "phase": traci.trafficlight.getPhase(tls)}
            for e in APPROACHES:
                row[f"q_{e}"] = traci.edge.getLastStepHaltingNumber(e)
            row["amb_phase"] = ambulance_target(greens)
            rows.append(row)
    traci.close()

    trips = pd.read_xml(trip, xpath=".//tripinfo", parser="etree")
    trips = trips[trips.depart >= 300]
    amb = trips[trips.vType == "ambulance"]
    others = trips[trips.vType != "ambulance"]
    result = {"controller": controller, "seed": seed,
              "mean_delay_s": round(others.timeLoss.mean(), 1),
              "amb_delay_s": round(amb.timeLoss.mean(), 1),
              "n_finished": len(others), "n_amb": len(amb)}
    return result, rows


if __name__ == "__main__":
    results, all_rows = [], []
    for controller in ["fixed", "longest_queue", "priority"]:
        for seed in range(1, 11):
            res, rows = run(controller, seed)
            print(res)
            results.append(res)
            all_rows += rows
    pd.DataFrame(results).to_csv("out/episodes.csv", index=False)
    pd.DataFrame(all_rows).to_csv("out/ticks.csv", index=False)
    print("done: out/episodes.csv and out/ticks.csv")