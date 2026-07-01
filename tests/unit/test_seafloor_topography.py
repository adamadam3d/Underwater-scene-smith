"""Unit tests for reefsmith.agent_utils.seafloor_topography."""

import numpy as np
import pytest

from reefsmith.agent_utils.seafloor_topography import (
    AnchorZone,
    SeafloorGrid,
    TrenchSpec,
    add_rock_plateau,
    carve_drop_off,
    carve_trench,
    compute_slope_field,
    find_anchor_zones,
    flat_seafloor,
    trenches_to_flow_channels,
)


# --- SeafloorGrid basics ---------------------------------------------------- #


def test_flat_seafloor_is_uniform():
    grid = flat_seafloor(10.0, 8.0, cell_size=1.0, base_height=-5.0)
    assert np.all(grid.heights == -5.0)
    assert grid.cell_size == 1.0


def test_flat_seafloor_rejects_bad_inputs():
    with pytest.raises(ValueError):
        flat_seafloor(0.0, 8.0, cell_size=1.0)
    with pytest.raises(ValueError):
        flat_seafloor(10.0, 8.0, cell_size=0.0)


def test_world_extent_matches_dimensions():
    grid = flat_seafloor(10.0, 5.0, cell_size=1.0)
    xmin, xmax, ymin, ymax = grid.world_extent
    assert xmin == pytest.approx(0.0)
    assert xmax == pytest.approx(10.0)
    assert ymin == pytest.approx(0.0)
    assert ymax == pytest.approx(5.0)


def test_height_at_interpolates_linearly():
    heights = np.array([[0.0, 0.0], [2.0, 2.0]])
    grid = SeafloorGrid(heights, cell_size=1.0)
    assert grid.height_at(np.array([0.0, 0.0])) == pytest.approx(0.0)
    assert grid.height_at(np.array([1.0, 0.0])) == pytest.approx(2.0)
    assert grid.height_at(np.array([0.5, 0.0])) == pytest.approx(1.0)


def test_height_at_clamps_outside_grid():
    grid = flat_seafloor(5.0, 5.0, cell_size=1.0, base_height=3.0)
    assert grid.height_at(np.array([-10.0, -10.0])) == pytest.approx(3.0)
    assert grid.height_at(np.array([1000.0, 1000.0])) == pytest.approx(3.0)


def test_grid_rejects_bad_shape():
    with pytest.raises(ValueError):
        SeafloorGrid(np.array([1.0, 2.0, 3.0]), cell_size=1.0)
    with pytest.raises(ValueError):
        SeafloorGrid(np.zeros((3, 3)), cell_size=0.0)


def test_copy_is_independent():
    grid = flat_seafloor(5.0, 5.0, cell_size=1.0)
    copy = grid.copy()
    copy.heights[0, 0] = 999.0
    assert grid.heights[0, 0] != 999.0


def test_to_trimesh_produces_valid_mesh():
    trimesh = pytest.importorskip("trimesh")
    grid = flat_seafloor(4.0, 4.0, cell_size=1.0)
    mesh = grid.to_trimesh()
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) == grid.shape[0] * grid.shape[1]
    assert len(mesh.faces) == 2 * (grid.shape[0] - 1) * (grid.shape[1] - 1)


# --- Rock plateaus ----------------------------------------------------------- #


def test_rock_plateau_raises_seafloor_near_center():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    raised = add_rock_plateau(grid, center_xy=np.array([5.0, 5.0]), radius=2.0, height=1.5)
    assert raised.height_at(np.array([5.0, 5.0])) == pytest.approx(1.5, abs=0.05)
    # Far from the plateau, seafloor is unchanged.
    assert raised.height_at(np.array([0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)


def test_overlapping_plateaus_merge_via_max_not_stack():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    once = add_rock_plateau(grid, np.array([5.0, 5.0]), radius=2.0, height=1.0)
    twice = add_rock_plateau(once, np.array([5.0, 5.0]), radius=2.0, height=1.0)
    # Same plateau added twice at the same spot should not double the height.
    assert twice.height_at(np.array([5.0, 5.0])) == pytest.approx(1.0, abs=0.05)


def test_plateau_rejects_bad_inputs():
    grid = flat_seafloor(5.0, 5.0, cell_size=0.5)
    with pytest.raises(ValueError):
        add_rock_plateau(grid, np.array([1.0, 1.0]), radius=0.0, height=1.0)
    with pytest.raises(ValueError):
        add_rock_plateau(grid, np.array([1.0, 1.0]), radius=1.0, height=0.0)


# --- Trenches ----------------------------------------------------------------- #


def test_trench_lowers_seafloor_along_line():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    trenched = carve_trench(
        grid, start_xy=np.array([0.0, 5.0]), end_xy=np.array([10.0, 5.0]),
        width=1.0, depth=2.0,
    )
    assert trenched.height_at(np.array([5.0, 5.0])) == pytest.approx(-2.0, abs=0.1)
    # Far from the trench line, seafloor is unaffected.
    assert trenched.height_at(np.array([5.0, 0.0])) == pytest.approx(0.0, abs=1e-6)


def test_trench_only_cuts_down_never_raises():
    grid = add_rock_plateau(
        flat_seafloor(10.0, 10.0, cell_size=0.5), np.array([5.0, 5.0]), 3.0, 2.0
    )
    trenched = carve_trench(
        grid, np.array([5.0, 0.0]), np.array([5.0, 10.0]), width=1.0, depth=0.1
    )
    # A shallow trench through an existing tall plateau should not exceed
    # the plateau's original height anywhere (min() combination).
    assert np.all(trenched.heights <= grid.heights + 1e-9)


def test_trench_rejects_bad_inputs():
    grid = flat_seafloor(5.0, 5.0, cell_size=0.5)
    with pytest.raises(ValueError):
        carve_trench(grid, np.array([0.0, 0.0]), np.array([1.0, 0.0]), width=0.0, depth=1.0)
    with pytest.raises(ValueError):
        carve_trench(grid, np.array([0.0, 0.0]), np.array([1.0, 0.0]), width=1.0, depth=0.0)


# --- Drop-offs ------------------------------------------------------------- #


def test_drop_off_steps_down_beyond_edge():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    dropped = carve_drop_off(
        grid, edge_start_xy=np.array([0.0, 5.0]), edge_end_xy=np.array([10.0, 5.0]),
        drop_depth=3.0, transition_width=0.5,
    )
    before = dropped.height_at(np.array([5.0, 0.0]))
    after = dropped.height_at(np.array([5.0, 10.0]))
    assert before == pytest.approx(0.0, abs=0.1)
    assert after == pytest.approx(-3.0, abs=0.1)


def test_drop_off_rejects_degenerate_edge():
    grid = flat_seafloor(5.0, 5.0, cell_size=0.5)
    with pytest.raises(ValueError):
        carve_drop_off(grid, np.array([1.0, 1.0]), np.array([1.0, 1.0]), drop_depth=1.0)


# --- Slope field ------------------------------------------------------------ #


def test_slope_field_is_zero_on_flat_seafloor():
    grid = flat_seafloor(5.0, 5.0, cell_size=0.5)
    slope = compute_slope_field(grid)
    assert np.allclose(slope, 0.0)


def test_slope_field_is_nonzero_at_a_slope():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    sloped = add_rock_plateau(grid, np.array([5.0, 5.0]), radius=1.0, height=2.0, falloff=1.0)
    slope = compute_slope_field(sloped)
    assert np.max(slope) > 0.1


# --- Anchor zones ------------------------------------------------------------ #


def test_anchor_zones_found_on_flat_terrain():
    grid = flat_seafloor(10.0, 10.0, cell_size=1.0)
    zones = find_anchor_zones(grid, search_radius=2.0, max_slope=0.5)
    assert len(zones) > 0
    assert all(isinstance(z, AnchorZone) for z in zones)
    assert all(z.max_slope <= 0.5 for z in zones)


def test_anchor_zones_respect_min_separation():
    grid = flat_seafloor(20.0, 20.0, cell_size=1.0)
    zones = find_anchor_zones(grid, search_radius=3.0, max_slope=0.5, grid_stride=1)
    for i, a in enumerate(zones):
        for b in zones[i + 1 :]:
            assert np.linalg.norm(a.center - b.center) >= 3.0 - 1e-6


def test_anchor_zones_reject_steep_terrain():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.25)
    steep = add_rock_plateau(grid, np.array([5.0, 5.0]), radius=0.5, height=5.0, falloff=0.25)
    zones = find_anchor_zones(steep, search_radius=1.0, max_slope=0.1)
    # None of the accepted anchors should be right at the steep peak center.
    for z in zones:
        assert np.linalg.norm(z.center - np.array([5.0, 5.0])) > 0.3


def test_anchor_zones_min_height_filters_trenches():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    trenched = carve_trench(
        grid, np.array([0.0, 5.0]), np.array([10.0, 5.0]), width=2.0, depth=3.0
    )
    zones = find_anchor_zones(
        trenched, search_radius=1.0, max_slope=1.0, min_height=-0.5
    )
    for z in zones:
        assert z.mean_height >= -0.5 - 1e-6


def test_anchor_zones_rejects_bad_inputs():
    grid = flat_seafloor(5.0, 5.0, cell_size=0.5)
    with pytest.raises(ValueError):
        find_anchor_zones(grid, search_radius=0.0, max_slope=0.5)
    with pytest.raises(ValueError):
        find_anchor_zones(grid, search_radius=1.0, max_slope=0.0)


# --- Trenches -> FlowChannel integration ------------------------------------ #


def test_trenches_to_flow_channels_produces_valid_channels():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    trench = TrenchSpec(
        start_xy=np.array([0.0, 5.0]), end_xy=np.array([10.0, 5.0]),
        width=2.0, depth=1.0,
    )
    channels = trenches_to_flow_channels([trench], grid)
    assert len(channels) == 1
    channel = channels[0]
    assert channel.radius == pytest.approx(1.0)
    assert channel.speed_gain > 1.0  # Deeper/narrower channel accelerates flow.
    assert channel.start[2] < 0  # Below the (flat, height=0) surrounding seafloor.


def test_deeper_narrower_trench_has_higher_speed_gain():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    shallow_wide = TrenchSpec(np.array([0.0, 2.0]), np.array([10.0, 2.0]), width=4.0, depth=0.5)
    deep_narrow = TrenchSpec(np.array([0.0, 8.0]), np.array([10.0, 8.0]), width=0.5, depth=4.0)
    channels = trenches_to_flow_channels([shallow_wide, deep_narrow], grid)
    assert channels[1].speed_gain > channels[0].speed_gain


def test_flow_channel_speed_gain_is_capped():
    grid = flat_seafloor(10.0, 10.0, cell_size=0.5)
    extreme = TrenchSpec(np.array([0.0, 5.0]), np.array([10.0, 5.0]), width=0.1, depth=50.0)
    channels = trenches_to_flow_channels([extreme], grid)
    assert channels[0].speed_gain <= 3.0
