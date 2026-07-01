"""Tests that the hydrodynamics config section in base_experiment.yaml is
well-formed and actually usable to construct the tested hydrodynamics /
ROV-clearance objects (not just present as inert YAML).
"""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from reefsmith.agent_utils.rov_clearance import ClearanceObstacle, compute_rov_clearance
from reefsmith.utils.hydrodynamics import SEAWATER_DENSITY, WaterCurrentField

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configurations" / "experiment" / "base_experiment.yaml"


@pytest.fixture(scope="module")
def cfg():
    return OmegaConf.load(CONFIG_PATH)


def test_config_file_exists():
    assert CONFIG_PATH.exists()


def test_hydrodynamics_section_present_with_expected_keys(cfg):
    assert "hydrodynamics" in cfg
    hydro = cfg.hydrodynamics
    assert "seawater_density" in hydro
    assert "current" in hydro
    assert "rov_clearance" in hydro
    assert "direction" in hydro.current
    assert "speed" in hydro.current
    assert "channels" in hydro.current
    assert "rov_radius_m" in hydro.rov_clearance
    assert "voxel_size_m" in hydro.rov_clearance


def test_seawater_density_matches_module_constant(cfg):
    # Config should not silently drift from the value the code assumes.
    assert cfg.hydrodynamics.seawater_density == pytest.approx(SEAWATER_DENSITY)


def test_current_values_are_physically_sane(cfg):
    current = cfg.hydrodynamics.current
    assert current.speed >= 0.0
    assert len(current.direction) == 3
    assert current.channels == []  # Populated at runtime, not statically.


def test_rov_clearance_values_are_physically_sane(cfg):
    rov = cfg.hydrodynamics.rov_clearance
    assert rov.rov_radius_m > 0.0
    assert rov.voxel_size_m > 0.0
    # Voxel resolution finer than the ROV body avoids badly aliased clearance
    # checks (a voxel much bigger than the ROV could hide real passages).
    assert rov.voxel_size_m <= rov.rov_radius_m * 2.0


def test_config_current_constructs_a_working_water_current_field(cfg):
    current_cfg = cfg.hydrodynamics.current
    field = WaterCurrentField(
        direction=np.array(list(current_cfg.direction)), speed=float(current_cfg.speed)
    )
    velocity = field.velocity_at(np.array([1.0, 2.0, 3.0]))
    assert velocity[2] == 0.0  # Current is horizontal.
    assert np.linalg.norm(velocity) == pytest.approx(float(current_cfg.speed))


def test_config_rov_settings_construct_a_working_clearance_check(cfg):
    rov_cfg = cfg.hydrodynamics.rov_clearance
    obstacle = ClearanceObstacle(
        "reef_wall", np.array([4.5, 0.0, 0.0]), np.array([5.5, 10.0, 4.0])
    )
    result = compute_rov_clearance(
        bounds_min=np.array([0.0, 0.0, 0.0]),
        bounds_max=np.array([10.0, 10.0, 4.0]),
        obstacles=[obstacle],
        voxel_size=float(rov_cfg.voxel_size_m),
        rov_radius=float(rov_cfg.rov_radius_m),
    )
    assert not result.is_fully_reachable
    assert "reef_wall" in result.blocking_obstacle_ids
