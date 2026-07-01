"""Unit tests for reefsmith.agent_utils.rov_clearance (3D volumetric ROV clearance).

These tests exercise the core claim that motivates this module: an ROV is a
free-swimming 3D body, not a floor-bound 2D footprint, so it can reach places
a flat 2D projection would flag as unreachable (e.g. by swimming over a
boulder that spans a room's full XY footprint but leaves headroom above it).
"""

import numpy as np
import pytest

from reefsmith.agent_utils.rov_clearance import (
    ClearanceObstacle,
    compute_rov_clearance,
    find_vertical_bypass_passages,
    format_rov_clearance_result,
    voxelize_occupancy,
)

BOUNDS_MIN = np.array([0.0, 0.0, 0.0])
BOUNDS_MAX = np.array([10.0, 10.0, 4.0])


def test_obstacle_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        ClearanceObstacle("bad", np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))


def test_obstacle_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ClearanceObstacle("bad", np.array([0.0, 0.0]), np.array([1.0, 1.0, 1.0]))


def test_empty_scene_is_fully_reachable():
    result = compute_rov_clearance(BOUNDS_MIN, BOUNDS_MAX, [], voxel_size=0.5, rov_radius=0.2)
    assert result.is_fully_reachable
    assert result.num_disconnected_regions == 1
    assert result.reachability_ratio == pytest.approx(1.0)
    assert result.free_voxel_count == result.total_voxel_count


def test_full_volume_obstacle_leaves_no_free_space():
    obstacle = ClearanceObstacle("wall", BOUNDS_MIN, BOUNDS_MAX)
    result = compute_rov_clearance(
        BOUNDS_MIN, BOUNDS_MAX, [obstacle], voxel_size=0.5, rov_radius=0.1
    )
    assert not result.is_fully_reachable
    assert result.free_voxel_count == 0
    assert result.num_disconnected_regions == 0


def test_thin_wall_with_gap_is_reachable():
    # A wall spanning the full width/height but with a gap in the middle
    # (in Y) that the ROV can swim through.
    wall_left = ClearanceObstacle(
        "wall_left",
        np.array([4.5, 0.0, 0.0]),
        np.array([5.5, 4.0, 4.0]),
    )
    wall_right = ClearanceObstacle(
        "wall_right",
        np.array([4.5, 6.0, 0.0]),
        np.array([5.5, 10.0, 4.0]),
    )
    result = compute_rov_clearance(
        BOUNDS_MIN, BOUNDS_MAX, [wall_left, wall_right], voxel_size=0.25, rov_radius=0.2
    )
    assert result.is_fully_reachable


def test_full_wall_splits_scene_into_two_regions():
    wall = ClearanceObstacle(
        "full_wall", np.array([4.5, 0.0, 0.0]), np.array([5.5, 10.0, 4.0])
    )
    result = compute_rov_clearance(
        BOUNDS_MIN, BOUNDS_MAX, [wall], voxel_size=0.25, rov_radius=0.1
    )
    assert not result.is_fully_reachable
    assert result.num_disconnected_regions == 2
    assert "full_wall" in result.blocking_obstacle_ids
    assert 0.0 < result.reachability_ratio < 1.0


def test_reef_scene_boulder_and_gorgonian_stay_connected():
    # A boulder cluster and a thin sea fan that both leave the water volume
    # well connected (the common, non-blocking case).
    boulder = ClearanceObstacle(
        "boulder_1", np.array([2.0, 2.0, 0.0]), np.array([3.5, 3.5, 1.2])
    )
    sea_fan = ClearanceObstacle(
        "sea_fan_1", np.array([7.0, 7.0, 0.0]), np.array([7.1, 8.0, 1.5])
    )
    result = compute_rov_clearance(
        BOUNDS_MIN, BOUNDS_MAX, [boulder, sea_fan], voxel_size=0.25, rov_radius=0.15
    )
    assert result.is_fully_reachable
    assert result.blocking_obstacle_ids == []


def test_rov_can_swim_over_a_boulder_that_spans_full_footprint():
    # The headline underwater-specific behavior: a boulder spans the FULL XY
    # footprint of the scene (a 2D ground-robot check would call this a
    # total dead end - the "wall" case above) but only rises partway up, so
    # a free-swimming ROV can pass over the top of it.
    boulder = ClearanceObstacle(
        "big_boulder", np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 1.5])
    )
    result = compute_rov_clearance(
        BOUNDS_MIN, BOUNDS_MAX, [boulder], voxel_size=0.25, rov_radius=0.2
    )
    assert result.is_fully_reachable  # Free water above the boulder connects everything.

    bypass_fraction = find_vertical_bypass_passages(
        BOUNDS_MIN, BOUNDS_MAX, [boulder], voxel_size=0.25, rov_radius=0.2
    )
    # Every column is blocked in the flat 2D projection (the boulder spans
    # the full footprint), yet the ROV can bypass all of them by swimming
    # over the top - this is exactly what a 2D reachability check would miss.
    assert bypass_fraction == pytest.approx(1.0)


def test_no_bypass_when_2d_already_reports_nothing_blocked():
    # A small boulder that doesn't span the full footprint: 2D check finds
    # no fully-blocked columns, so there is nothing to "bypass".
    boulder = ClearanceObstacle(
        "small_boulder", np.array([2.0, 2.0, 0.0]), np.array([3.0, 3.0, 4.0])
    )
    bypass_fraction = find_vertical_bypass_passages(
        BOUNDS_MIN, BOUNDS_MAX, [boulder], voxel_size=0.25, rov_radius=0.1
    )
    assert bypass_fraction == 0.0


def test_true_full_height_wall_has_no_bypass():
    # A wall that truly blocks every column at every height offers no 3D
    # bypass at all - the 2D and 3D checks should agree completely.
    wall = ClearanceObstacle(
        "true_wall", np.array([4.5, 0.0, 0.0]), np.array([5.5, 10.0, 4.0])
    )
    bypass_fraction = find_vertical_bypass_passages(
        BOUNDS_MIN, BOUNDS_MAX, [wall], voxel_size=0.25, rov_radius=0.1
    )
    assert bypass_fraction == pytest.approx(0.0)


def test_larger_rov_radius_reduces_reachability():
    wall_left = ClearanceObstacle(
        "wall_left", np.array([4.5, 0.0, 0.0]), np.array([5.5, 4.7, 4.0])
    )
    wall_right = ClearanceObstacle(
        "wall_right", np.array([4.5, 5.3, 0.0]), np.array([5.5, 10.0, 4.0])
    )
    small_rov = compute_rov_clearance(
        BOUNDS_MIN, BOUNDS_MAX, [wall_left, wall_right], voxel_size=0.1, rov_radius=0.1
    )
    large_rov = compute_rov_clearance(
        BOUNDS_MIN, BOUNDS_MAX, [wall_left, wall_right], voxel_size=0.1, rov_radius=1.0
    )
    assert small_rov.is_fully_reachable
    assert not large_rov.is_fully_reachable


def test_voxelize_occupancy_respects_exclude_id():
    obstacle = ClearanceObstacle("x", np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]))
    with_it = voxelize_occupancy(
        BOUNDS_MIN, BOUNDS_MAX, [obstacle], voxel_size=0.5, rov_radius=0.0
    )
    without_it = voxelize_occupancy(
        BOUNDS_MIN,
        BOUNDS_MAX,
        [obstacle],
        voxel_size=0.5,
        rov_radius=0.0,
        exclude_id="x",
    )
    assert np.count_nonzero(with_it) > 0
    assert np.count_nonzero(without_it) == 0


def test_voxelize_occupancy_validates_inputs():
    with pytest.raises(ValueError):
        voxelize_occupancy(BOUNDS_MIN, BOUNDS_MAX, [], voxel_size=0.0, rov_radius=0.1)
    with pytest.raises(ValueError):
        voxelize_occupancy(BOUNDS_MIN, BOUNDS_MAX, [], voxel_size=0.5, rov_radius=-1.0)
    with pytest.raises(ValueError):
        voxelize_occupancy(BOUNDS_MAX, BOUNDS_MIN, [], voxel_size=0.5, rov_radius=0.1)


def test_obstacle_outside_bounds_is_ignored():
    far_away = ClearanceObstacle(
        "far", np.array([100.0, 100.0, 100.0]), np.array([101.0, 101.0, 101.0])
    )
    occupancy = voxelize_occupancy(
        BOUNDS_MIN, BOUNDS_MAX, [far_away], voxel_size=0.5, rov_radius=0.1
    )
    assert not occupancy.any()


def test_format_result_fully_reachable():
    result = compute_rov_clearance(BOUNDS_MIN, BOUNDS_MAX, [], voxel_size=1.0, rov_radius=0.2)
    text = format_rov_clearance_result(result)
    assert "fully navigable" in text


def test_format_result_blocked_lists_blockers():
    wall = ClearanceObstacle(
        "reef_wall", np.array([4.5, 0.0, 0.0]), np.array([5.5, 10.0, 4.0])
    )
    result = compute_rov_clearance(
        BOUNDS_MIN, BOUNDS_MAX, [wall], voxel_size=0.25, rov_radius=0.1
    )
    text = format_rov_clearance_result(result)
    assert "disconnected" in text
    assert "reef_wall" in text
