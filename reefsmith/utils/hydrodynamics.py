"""Hydrodynamic property estimation for underwater (coral reef) scenes.

This module provides the marine-physics primitives that distinguish ReefSmith
from its indoor ancestor: buoyancy, fluid drag, a (optionally channelized)
water-current field, and current-alignment for flow-sensitive accents such as
sea fans / gorgonians.

Everything here is pure ``numpy`` math with no dependency on Drake or Blender,
so it can be unit-tested in isolation and consumed both by the SDF generator
(to attach buoyancy/drag properties to generated meshes) and by the accent
agents (to orient current-facing structures into the prevailing flow).

Coordinate convention (matching the rest of the codebase): ``+Z`` is world up,
gravity acts along ``-Z``. Currents are horizontal (in the world ``XY`` plane).
"""

import logging
import math

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

console_logger = logging.getLogger(__name__)

# Density of seawater at typical reef temperature/salinity (kg/m^3).
SEAWATER_DENSITY = 1025.0
# Standard gravitational acceleration (m/s^2).
GRAVITY = 9.81


class MarineMaterial(Enum):
    """Marine material categories with saturated (wet) densities and base drag.

    Densities are saturated/submerged values in kg/m^3 - i.e. the effective
    density of the specimen once water has filled its pores, which is what
    matters for buoyancy in simulation. ``base_drag_coefficient`` is a
    dimensionless reference :math:`C_d` for a bluff body of that family,
    later modulated by shape (see :func:`drag_coefficient_from_shape`).
    """

    ROCK = ("rock", 2400.0, 0.8)
    BOULDER = ("boulder", 2500.0, 0.8)
    SAND = ("sand", 1900.0, 0.9)
    MASSIVE_CORAL = ("massive_coral", 1700.0, 1.0)  # e.g. brain coral
    BRANCHING_CORAL = ("branching_coral", 1400.0, 1.2)  # e.g. staghorn
    SPONGE = ("sponge", 1100.0, 1.1)  # e.g. tube sponge (saturated)
    SEA_FAN = ("sea_fan", 1200.0, 1.4)  # gorgonian; broad flat face, high drag
    SOFT_TISSUE = ("soft_tissue", 1050.0, 1.0)  # anemone, starfish, crab flesh
    DEFAULT = ("default", 1500.0, 1.0)

    def __init__(self, label: str, density: float, base_drag_coefficient: float):
        self.label = label
        self.density = density
        self.base_drag_coefficient = base_drag_coefficient

    @classmethod
    def from_label(cls, label: str) -> "MarineMaterial":
        """Resolve a material from a free-form label, falling back to DEFAULT.

        Matching is case-insensitive and tolerant of substrings, so VLM-produced
        descriptors like "purple branching coral" resolve to ``BRANCHING_CORAL``.
        """
        if label is None:
            return cls.DEFAULT
        text = label.strip().lower().replace(" ", "_")
        # Exact label match first.
        for member in cls:
            if member.label == text:
                return member
        # Substring / keyword match (most specific keywords first).
        keywords = [
            ("staghorn", cls.BRANCHING_CORAL),
            ("branching", cls.BRANCHING_CORAL),
            ("brain", cls.MASSIVE_CORAL),
            ("massive", cls.MASSIVE_CORAL),
            ("coral", cls.MASSIVE_CORAL),
            ("gorgonian", cls.SEA_FAN),
            ("fan", cls.SEA_FAN),
            ("sponge", cls.SPONGE),
            ("boulder", cls.BOULDER),
            ("rock", cls.ROCK),
            ("sand", cls.SAND),
            ("anemone", cls.SOFT_TISSUE),
            ("starfish", cls.SOFT_TISSUE),
            ("urchin", cls.SOFT_TISSUE),
            ("crab", cls.SOFT_TISSUE),
        ]
        for keyword, member in keywords:
            if keyword in text:
                return member
        return cls.DEFAULT


def buoyant_force(volume_m3: float, fluid_density: float = SEAWATER_DENSITY) -> float:
    """Upward buoyant force (N) on a fully submerged body (Archimedes' principle).

    Args:
        volume_m3: Displaced volume of the body in cubic metres.
        fluid_density: Surrounding fluid density (defaults to seawater).

    Returns:
        Magnitude of the buoyant force in newtons (always >= 0).
    """
    if volume_m3 < 0:
        raise ValueError(f"volume_m3 must be non-negative, got {volume_m3}")
    return fluid_density * volume_m3 * GRAVITY


def net_buoyant_force(
    volume_m3: float, body_density: float, fluid_density: float = SEAWATER_DENSITY
) -> float:
    """Net vertical force (N) on a submerged body, positive = upward (rises).

    A positive result means the body is positively buoyant (floats up); a
    negative result means it sinks and will rest on the seabed; near zero is
    neutrally buoyant.

    Args:
        volume_m3: Displaced volume in cubic metres.
        body_density: Effective (saturated) density of the body in kg/m^3.
        fluid_density: Surrounding fluid density (defaults to seawater).
    """
    return (fluid_density - body_density) * volume_m3 * GRAVITY


def apparent_submerged_weight(
    volume_m3: float, body_density: float, fluid_density: float = SEAWATER_DENSITY
) -> float:
    """Apparent (downward) weight of a submerged body in newtons.

    This is the seabed-normal load a resting structure exerts underwater - the
    dry weight reduced by buoyancy. Positive means it presses down on the
    seabed; <= 0 means it would lift off (positively buoyant).
    """
    return -net_buoyant_force(volume_m3, body_density, fluid_density)


def is_positively_buoyant(
    volume_m3: float, body_density: float, fluid_density: float = SEAWATER_DENSITY
) -> bool:
    """True if the body would float/rise rather than rest on the seabed."""
    return body_density < fluid_density


def drag_coefficient_from_shape(
    extents: np.ndarray, material: MarineMaterial = MarineMaterial.DEFAULT
) -> float:
    """Estimate a drag coefficient :math:`C_d` from bounding-box shape + material.

    Starts from the material's bluff-body base coefficient and modulates it by
    the bounding-box aspect ratio: very flat/plate-like specimens (e.g. sea
    fans) present a broad face and drag more, while elongated/streamlined
    specimens drag slightly less. The result is clamped to a physically
    reasonable ``[0.4, 2.0]`` range.

    Args:
        extents: Length-3 array of axis-aligned bounding-box side lengths (m).
        material: Marine material family providing the base coefficient.

    Returns:
        Dimensionless drag coefficient.
    """
    extents = np.asarray(extents, dtype=float)
    if extents.shape != (3,):
        raise ValueError(f"extents must have shape (3,), got {extents.shape}")
    if np.any(extents <= 0):
        raise ValueError(f"extents must be strictly positive, got {extents}")

    cd = material.base_drag_coefficient
    smallest, _, largest = np.sort(extents)
    aspect = largest / smallest  # >= 1

    if aspect >= 3.0:
        # Plate-like / flat: broad face -> more drag.
        cd *= 1.0 + 0.1 * min(aspect - 3.0 + 2.0, 5.0)
    return float(np.clip(cd, 0.4, 2.0))


def reference_area(extents: np.ndarray) -> float:
    """Frontal reference area (m^2): the largest of the three face areas.

    A conservative choice for drag - the broadest cross-section the flow can
    push against, regardless of incidence.
    """
    extents = np.asarray(extents, dtype=float)
    if extents.shape != (3,):
        raise ValueError(f"extents must have shape (3,), got {extents.shape}")
    x, y, z = extents
    return float(max(x * y, x * z, y * z))


def drag_force(
    flow_speed: float,
    area_m2: float,
    drag_coefficient: float,
    fluid_density: float = SEAWATER_DENSITY,
) -> float:
    """Quadratic fluid drag magnitude (N): ``0.5 * rho * Cd * A * v^2``.

    Args:
        flow_speed: Relative flow speed past the body in m/s.
        area_m2: Reference (frontal) area in m^2.
        drag_coefficient: Dimensionless drag coefficient.
        fluid_density: Surrounding fluid density (defaults to seawater).
    """
    return 0.5 * fluid_density * drag_coefficient * area_m2 * flow_speed * flow_speed


@dataclass(frozen=True)
class HydrodynamicProperties:
    """Bundle of estimated underwater physical properties for a marine asset.

    Intended to be attached alongside the existing mass/inertia data when an
    asset is made simulation-ready, giving Drake the information needed to model
    buoyancy and drag.
    """

    material: MarineMaterial
    volume_m3: float
    body_density: float
    buoyant_force_n: float
    net_buoyant_force_n: float
    apparent_submerged_weight_n: float
    drag_coefficient: float
    reference_area_m2: float
    positively_buoyant: bool


def estimate_hydrodynamic_properties(
    volume_m3: float,
    extents: np.ndarray,
    material: MarineMaterial = MarineMaterial.DEFAULT,
    body_density: float | None = None,
    fluid_density: float = SEAWATER_DENSITY,
) -> HydrodynamicProperties:
    """Estimate the full hydrodynamic property bundle for a marine asset.

    Args:
        volume_m3: Mesh volume in cubic metres.
        extents: Length-3 bounding-box side lengths (m).
        material: Marine material family (drives default density + drag).
        body_density: Override the material's saturated density (e.g. when a
            real mass estimate is available: ``density = mass / volume``).
        fluid_density: Surrounding fluid density (defaults to seawater).
    """
    if volume_m3 <= 0:
        raise ValueError(f"volume_m3 must be positive, got {volume_m3}")
    density = material.density if body_density is None else float(body_density)
    cd = drag_coefficient_from_shape(extents, material)
    return HydrodynamicProperties(
        material=material,
        volume_m3=float(volume_m3),
        body_density=density,
        buoyant_force_n=buoyant_force(volume_m3, fluid_density),
        net_buoyant_force_n=net_buoyant_force(volume_m3, density, fluid_density),
        apparent_submerged_weight_n=apparent_submerged_weight(
            volume_m3, density, fluid_density
        ),
        drag_coefficient=cd,
        reference_area_m2=reference_area(extents),
        positively_buoyant=is_positively_buoyant(volume_m3, density, fluid_density),
    )


def _unit_horizontal(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto the horizontal plane and normalize (XY, z=0)."""
    v = np.asarray(vector, dtype=float).copy()
    if v.shape != (3,):
        raise ValueError(f"vector must have shape (3,), got {v.shape}")
    v[2] = 0.0
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return v / norm


@dataclass
class FlowChannel:
    """A localized channel of accelerated, directed flow along a line segment.

    Inside the tube of the given ``radius`` around the segment, the current
    follows the segment's axis at ``base_speed * speed_gain``. This models the
    flow channels and trenches the Seabed/Reef agents are meant to keep clear.
    """

    start: np.ndarray
    end: np.ndarray
    radius: float
    speed_gain: float = 1.5

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=float)
        self.end = np.asarray(self.end, dtype=float)
        if self.start.shape != (3,) or self.end.shape != (3,):
            raise ValueError("FlowChannel start/end must be length-3 vectors")
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")

    def distance_to_axis(self, point: np.ndarray) -> float:
        """Shortest distance from ``point`` to the channel center segment."""
        p = np.asarray(point, dtype=float)
        seg = self.end - self.start
        seg_len_sq = float(seg @ seg)
        if seg_len_sq < 1e-12:
            return float(np.linalg.norm(p - self.start))
        t = float((p - self.start) @ seg / seg_len_sq)
        t = min(1.0, max(0.0, t))
        closest = self.start + t * seg
        return float(np.linalg.norm(p - closest))

    def axis_direction(self) -> np.ndarray:
        """Unit horizontal direction of flow along the channel."""
        return _unit_horizontal(self.end - self.start)

    def contains(self, point: np.ndarray) -> bool:
        """True if ``point`` lies within the channel tube."""
        return self.distance_to_axis(point) <= self.radius


@dataclass
class WaterCurrentField:
    """A simple water-current field: a uniform background flow plus channels.

    ``velocity_at`` returns the horizontal current velocity vector (m/s) at a
    world point. Inside any :class:`FlowChannel` the channel's accelerated,
    axis-aligned flow takes precedence over the background; if a point falls in
    multiple channels the fastest wins.
    """

    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    speed: float = 0.3
    channels: list[FlowChannel] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.direction = _unit_horizontal(self.direction)
        if self.speed < 0:
            raise ValueError(f"speed must be non-negative, got {self.speed}")

    def velocity_at(self, point: np.ndarray) -> np.ndarray:
        """Horizontal current velocity (m/s) at a world point."""
        best = self.direction * self.speed
        best_speed = self.speed
        for channel in self.channels:
            if channel.contains(point):
                channel_speed = self.speed * channel.speed_gain
                if channel_speed > best_speed:
                    best = channel.axis_direction() * channel_speed
                    best_speed = channel_speed
        return best

    def speed_at(self, point: np.ndarray) -> float:
        """Scalar current speed (m/s) at a world point."""
        return float(np.linalg.norm(self.velocity_at(point)))


def current_alignment_yaw(current_velocity: np.ndarray) -> float:
    """Yaw angle (radians, about +Z) that points an object's local +X normal
    into the given horizontal current.

    Used to orient flow-sensitive accents - e.g. a sea fan, whose broad
    photosynthetic face should sit perpendicular to the flow to maximize
    feeding - so its face normal (local +X) faces upstream into the current.

    Returns 0.0 for a negligible/zero current.
    """
    v = np.asarray(current_velocity, dtype=float)
    if v.shape != (3,):
        raise ValueError(f"current_velocity must have shape (3,), got {v.shape}")
    if np.linalg.norm(v[:2]) < 1e-12:
        return 0.0
    return float(math.atan2(v[1], v[0]))


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    """Quaternion ``[w, x, y, z]`` for a rotation of ``yaw`` radians about +Z."""
    half = yaw / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)])


def align_quaternion_to_current(current_velocity: np.ndarray) -> np.ndarray:
    """Convenience: quaternion ``[w, x, y, z]`` orienting local +X into the flow.

    Combines :func:`current_alignment_yaw` and :func:`yaw_to_quaternion`.
    """
    return yaw_to_quaternion(current_alignment_yaw(current_velocity))
