"""traffic_twin: a small microscopic traffic simulator for a 2x2 signalised grid.

Modules: config (every number), network (grid, edges, connectors), vehicles (IDM),
signals (plant per junction), controllers (baselines + label function), features
(signal_ticks row), sim (World), calibrate, validate, generate, viewer.
"""

from .config import GENERATOR_VERSION, SimConfig, load_config
from .sim import Scenario, World, run_episode

__all__ = ["GENERATOR_VERSION", "SimConfig", "load_config", "Scenario", "World", "run_episode"]
