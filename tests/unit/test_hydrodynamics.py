"""Unit tests for reefsmith.utils.hydrodynamics (marine physics primitives)."""

import math

import numpy as np
import pytest

from reefsmith.utils.hydrodynamics import (
    GRAVITY,
    SEAWATER_DENSITY,
    FlowChannel,
    HydrodynamicProperties,
    MarineMaterial,
    WaterCurrentField,
    align_quaternion_to_current,
    apparent_submerged_weight,
    buoyant_force,
    current_alignment_yaw,
    drag_coefficient_from_shape,
    drag_force,
    estimate_hydrodynamic_properties,
    is_positively_buoyant,
    net_buoyant_force,
    reference_area,
    surface_current_facing_angle_degrees,
    yaw_to_quaternion,
)


# --- Material resolution -------------------------------------------------- #


def test_material_from_label_exact_and_keyword():
    assert MarineMaterial.from_label("sponge") is MarineMaterial.SPONGE
    assert MarineMaterial.from_label("purple branching coral") is (
        MarineMaterial.BRANCHING_CORAL
    )
    assert MarineMaterial.from_label("giant brain coral") is (
        MarineMaterial.MASSIVE_CORAL
    )
    assert MarineMaterial.from_label("a sea fan / gorgonian") is MarineMaterial.SEA_FAN
    assert MarineMaterial.from_label("red starfish") is MarineMaterial.SOFT_TISSUE
    assert MarineMaterial.from_label("something unknown") is MarineMaterial.DEFAULT
    assert MarineMaterial.from_label(None) is MarineMaterial.DEFAULT


def test_material_densities_bracket_seawater():
    # Rock is much denser than water; saturated sponge is only slightly denser.
    assert MarineMaterial.ROCK.density > SEAWATER_DENSITY
    assert MarineMaterial.SPONGE.density > SEAWATER_DENSITY
    assert MarineMaterial.SOFT_TISSUE.density > SEAWATER_DENSITY


# --- Buoyancy ------------------------------------------------------------- #


def test_buoyant_force_archimedes():
    # 1 m^3 displaces seawater -> rho * V * g newtons.
    assert buoyant_force(1.0) == pytest.approx(SEAWATER_DENSITY * GRAVITY)
    assert buoyant_force(0.0) == 0.0


def test_buoyant_force_rejects_negative_volume():
    with pytest.raises(ValueError):
        buoyant_force(-1.0)


def test_rock_sinks_coral_rests_sponge_barely():
    v = 0.5
    rock = net_buoyant_force(v, MarineMaterial.ROCK.density)
    assert rock < 0  # sinks
    assert not is_positively_buoyant(v, MarineMaterial.ROCK.density)
    # Apparent submerged weight is the downward seabed load (positive).
    assert apparent_submerged_weight(v, MarineMaterial.ROCK.density) == pytest.approx(
        -rock
    )
    assert apparent_submerged_weight(v, MarineMaterial.ROCK.density) > 0


def test_hypothetical_positively_buoyant_body_rises():
    # A body lighter than seawater (e.g. a gas-filled float) rises.
    light_density = 200.0
    assert net_buoyant_force(0.1, light_density) > 0
    assert is_positively_buoyant(0.1, light_density)
    assert apparent_submerged_weight(0.1, light_density) < 0


def test_net_buoyancy_neutral_at_seawater_density():
    assert net_buoyant_force(2.0, SEAWATER_DENSITY) == pytest.approx(0.0)


# --- Drag ----------------------------------------------------------------- #


def test_drag_force_scales_quadratically():
    f1 = drag_force(1.0, 2.0, 1.0)
    f2 = drag_force(2.0, 2.0, 1.0)
    assert f2 == pytest.approx(4.0 * f1)
    assert f1 == pytest.approx(0.5 * SEAWATER_DENSITY * 1.0 * 2.0 * 1.0)


def test_flat_specimen_has_higher_drag_than_cube():
    cube = drag_coefficient_from_shape(np.array([1.0, 1.0, 1.0]), MarineMaterial.SEA_FAN)
    flat = drag_coefficient_from_shape(
        np.array([1.0, 1.0, 0.1]), MarineMaterial.SEA_FAN
    )
    assert flat > cube
    assert 0.4 <= flat <= 2.0


def test_reference_area_is_largest_face():
    assert reference_area(np.array([2.0, 3.0, 0.5])) == pytest.approx(6.0)


def test_drag_coefficient_validates_extents():
    with pytest.raises(ValueError):
        drag_coefficient_from_shape(np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        drag_coefficient_from_shape(np.array([1.0, 1.0]))


# --- Property bundle ------------------------------------------------------ #


def test_estimate_hydrodynamic_properties_rock_boulder():
    props = estimate_hydrodynamic_properties(
        volume_m3=0.8,
        extents=np.array([1.2, 1.0, 0.9]),
        material=MarineMaterial.BOULDER,
    )
    assert isinstance(props, HydrodynamicProperties)
    assert props.material is MarineMaterial.BOULDER
    assert props.body_density == MarineMaterial.BOULDER.density
    assert props.net_buoyant_force_n < 0  # sinks
    assert props.apparent_submerged_weight_n > 0
    assert props.buoyant_force_n == pytest.approx(SEAWATER_DENSITY * 0.8 * GRAVITY)
    assert not props.positively_buoyant


def test_estimate_with_density_override_from_mass():
    # density = mass / volume, mirroring the SDF generator's approach.
    volume = 0.25
    mass = 500.0  # kg -> density 2000
    props = estimate_hydrodynamic_properties(
        volume_m3=volume,
        extents=np.array([0.6, 0.6, 0.7]),
        material=MarineMaterial.MASSIVE_CORAL,
        body_density=mass / volume,
    )
    assert props.body_density == pytest.approx(2000.0)
    assert props.net_buoyant_force_n < 0


def test_estimate_rejects_nonpositive_volume():
    with pytest.raises(ValueError):
        estimate_hydrodynamic_properties(0.0, np.array([1.0, 1.0, 1.0]))


# --- Current field -------------------------------------------------------- #


def test_uniform_current_is_constant_and_horizontal():
    field_ = WaterCurrentField(direction=np.array([0.0, 1.0, 0.0]), speed=0.5)
    v = field_.velocity_at(np.array([3.0, 2.0, 1.0]))
    assert v == pytest.approx(np.array([0.0, 0.5, 0.0]))
    assert v[2] == 0.0
    assert field_.speed_at(np.array([10.0, -4.0, 2.0])) == pytest.approx(0.5)


def test_direction_is_normalized_and_flattened():
    field_ = WaterCurrentField(direction=np.array([3.0, 0.0, 4.0]), speed=1.0)
    # z component dropped, remaining normalized -> (1, 0, 0).
    assert field_.direction == pytest.approx(np.array([1.0, 0.0, 0.0]))


def test_flow_channel_accelerates_and_redirects_inside_tube():
    channel = FlowChannel(
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([10.0, 0.0, 0.0]),
        radius=1.0,
        speed_gain=2.0,
    )
    field_ = WaterCurrentField(
        direction=np.array([0.0, 1.0, 0.0]), speed=0.4, channels=[channel]
    )
    # Point inside the channel tube: flow follows +X at boosted speed.
    inside = field_.velocity_at(np.array([5.0, 0.2, 0.0]))
    assert inside == pytest.approx(np.array([0.8, 0.0, 0.0]))
    # Point well outside: background +Y flow.
    outside = field_.velocity_at(np.array([5.0, 5.0, 0.0]))
    assert outside == pytest.approx(np.array([0.0, 0.4, 0.0]))


def test_flow_channel_distance_and_contains():
    channel = FlowChannel(
        start=np.array([0.0, 0.0, 0.0]),
        end=np.array([0.0, 4.0, 0.0]),
        radius=0.5,
    )
    assert channel.contains(np.array([0.3, 2.0, 0.0]))
    assert not channel.contains(np.array([2.0, 2.0, 0.0]))
    assert channel.distance_to_axis(np.array([1.0, 2.0, 0.0])) == pytest.approx(1.0)
    # Beyond the segment end, distance measured from the endpoint.
    assert channel.distance_to_axis(np.array([0.0, 6.0, 0.0])) == pytest.approx(2.0)


def test_flow_channel_rejects_bad_inputs():
    with pytest.raises(ValueError):
        FlowChannel(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), radius=0.0)


# --- Current alignment (sea fans) ----------------------------------------- #


def test_current_alignment_yaw_matches_flow_direction():
    assert current_alignment_yaw(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert current_alignment_yaw(np.array([0.0, 1.0, 0.0])) == pytest.approx(
        math.pi / 2
    )
    assert current_alignment_yaw(np.array([-1.0, 0.0, 0.0])) == pytest.approx(math.pi)
    # Negligible current -> no rotation.
    assert current_alignment_yaw(np.array([0.0, 0.0, 1.0])) == 0.0


def test_alignment_quaternion_rotates_local_x_into_current():
    # A current pointing +Y should rotate the fan's local +X normal to +Y.
    quat = align_quaternion_to_current(np.array([0.0, 1.0, 0.0]))
    w, x, y, z = quat
    # Rotate local +X by the quaternion and check it lands on world +Y.
    rotated = _rotate_vector_by_quaternion(np.array([1.0, 0.0, 0.0]), quat)
    assert rotated == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-9)
    assert np.linalg.norm(quat) == pytest.approx(1.0)


def test_yaw_quaternion_is_unit():
    q = yaw_to_quaternion(1.234)
    assert np.linalg.norm(q) == pytest.approx(1.0)


# --- Surface-mounted current facing (wall/accent placement) --------------- #


def test_surface_facing_zero_when_current_already_aligned_with_up():
    # Wall faces +X; "rotation_degrees=0" reference axis is +Z.
    angle = surface_current_facing_angle_degrees(
        surface_normal=np.array([1.0, 0.0, 0.0]),
        surface_up=np.array([0.0, 0.0, 1.0]),
        current_velocity=np.array([0.0, 0.0, 2.0]),
    )
    assert angle == pytest.approx(0.0)


def test_surface_facing_ninety_degrees_for_perpendicular_in_plane_current():
    # right_in_plane = normal x up = X x Z = -Y, so a current along -Y is a
    # pure 90-degree in-plane rotation away from the "up" reference axis.
    angle = surface_current_facing_angle_degrees(
        surface_normal=np.array([1.0, 0.0, 0.0]),
        surface_up=np.array([0.0, 0.0, 1.0]),
        current_velocity=np.array([0.0, -3.0, 0.0]),
    )
    assert angle == pytest.approx(90.0)


def test_surface_facing_zero_when_current_hits_surface_head_on():
    # Current flows straight along the surface normal - no in-plane
    # component to face into.
    angle = surface_current_facing_angle_degrees(
        surface_normal=np.array([1.0, 0.0, 0.0]),
        surface_up=np.array([0.0, 0.0, 1.0]),
        current_velocity=np.array([5.0, 0.0, 0.0]),
    )
    assert angle == pytest.approx(0.0)


def test_surface_facing_handles_non_orthogonal_up_by_projecting():
    # surface_up need not be exactly orthogonal to the normal.
    angle = surface_current_facing_angle_degrees(
        surface_normal=np.array([1.0, 0.0, 0.0]),
        surface_up=np.array([0.3, 0.0, 1.0]),  # has a component along normal
        current_velocity=np.array([0.0, 0.0, 4.0]),
    )
    assert angle == pytest.approx(0.0)


def test_surface_facing_rejects_zero_normal():
    with pytest.raises(ValueError):
        surface_current_facing_angle_degrees(
            surface_normal=np.array([0.0, 0.0, 0.0]),
            surface_up=np.array([0.0, 0.0, 1.0]),
            current_velocity=np.array([1.0, 0.0, 0.0]),
        )


def test_surface_facing_rejects_up_parallel_to_normal():
    with pytest.raises(ValueError):
        surface_current_facing_angle_degrees(
            surface_normal=np.array([1.0, 0.0, 0.0]),
            surface_up=np.array([2.0, 0.0, 0.0]),
            current_velocity=np.array([0.0, 1.0, 0.0]),
        )


def test_surface_facing_rejects_bad_shapes():
    with pytest.raises(ValueError):
        surface_current_facing_angle_degrees(
            surface_normal=np.array([1.0, 0.0]),
            surface_up=np.array([0.0, 0.0, 1.0]),
            current_velocity=np.array([0.0, 1.0, 0.0]),
        )


def _rotate_vector_by_quaternion(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by quaternion [w, x, y, z] (test helper)."""
    w, x, y, z = q
    # Rotation matrix from quaternion.
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rot @ v
