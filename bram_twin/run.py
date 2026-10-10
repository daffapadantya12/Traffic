import traci
import pandas as pd

APPROACHES = ["N2C", "E2C", "S2C", "W2C"]

traci.start(["sumo", "-n", "junction.net.xml", "-r", "traffic.rou.xml",
             "--lateral-resolution", "0.8", "--seed", "1",
             "--tripinfo-output", "tripinfo.xml"])
tls = traci.trafficlight.getIDList()[0]

rows = []
while traci.simulation.getTime() < 3600:
    traci.simulationStep()
    t = traci.simulation.getTime()
    if t % 5 == 0:                                         # a "sensor reading" every 5 s
        row = {"t": t, "phase": traci.trafficlight.getPhase(tls)}
        for e in APPROACHES:
            row[f"q_{e}"] = traci.edge.getLastStepHaltingNumber(e)   # standing vehicles = queue camera
        rows.append(row)
traci.close()

pd.DataFrame(rows).to_csv("ticks.csv", index=False)
print("saved", len(rows), "rows")

trips = pd.read_xml("tripinfo.xml", xpath="//tripinfo")
trips["timeLoss"].mean()                 # average delay per vehicle, in seconds