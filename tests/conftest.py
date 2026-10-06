"""Shared fixtures: short episodes so the whole test suite runs in seconds."""

import pytest

from traffic_twin import Scenario, SimConfig

SHORT = dict(warmup_s=60.0, main_s=240.0, cooldown_s=60.0)


@pytest.fixture
def cfg_2x2():
    return SimConfig(**SHORT)


@pytest.fixture
def cfg_1x1():
    return SimConfig(grid_x=1, grid_y=1, **SHORT)


@pytest.fixture
def scenario():
    return Scenario(demand_level="test", lambda_base_vph=220.0, controller="max_pressure")
