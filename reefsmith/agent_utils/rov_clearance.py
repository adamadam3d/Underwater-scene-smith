"""3D volumetric clearance analysis for ROV/AUV navigation through a reef scene.

This module is the underwater-specific replacement for the indoor-derived
``reachability.py``, which checks whether a *ground* robot confined to the
floor plane (a 2D footprint) can reach every part of a room. An underwater
robot (ROV/AUV) is not confined to a floor plane - it swims freely in 3D, so
it can pass *over* a boulder, *under* an overhang, or *through* a gap between
two reef structures that a flat 2D projection would incorrectly flag as a
dead end.

Algorithm
---------
1. Voxelize the scene's bounding volume into a regular 3D grid.
2. Mark a voxel "occupied" if it falls inside any obstacle's axis-aligned
   bounding box, inflated by the ROV's radius (standard configuration-space
   trick: inflate obstacles by the robot radius so the robot can be treated
   as a point for connectivity purposes).
3. Label connected components of free space with 6-connectivity (face
   adjacency only - the physically conservative choice, since a rigid ROV
   body cannot slip through a diagonal-only gap).
4. The largest connected component is the "reachable" region from any point
   within it; disconnected pockets are unreachable without leaving the water
   volume or passing through an obstacle.

For comparison and diagnostics, :func:`find_vertical_bypass_passages` also
runs the *old* 2D-floor-projection style of check (collapsing every obstacle
down to its full-height footprint, as ``reachability.py`` does for furniture)
and reports where the two disagree - i.e., exactly the passages a 3D-aware
ROV can use that a 2D ground-robot check would have rejected.

This module operates on plain-data obstacles (id + AABB corners) rather than
on :class:`~reefsmith.agent_utils.room.SceneObject`, so it can be unit tested
without a Drake installation. A thin adapter that extracts AABBs from a
``RoomScene`` (analogous to ``reachability.py``'s ``_get_furniture_obbs``)
is the natural next integration point once this is used by the Reef Agent.
"""

import logging

from dataclasses import dataclass

import numpy as np

from scipy import ndimage

console_logger = logging.getLogger(__name__)

# 6-connectivity structuring element (face-adjacent voxels only). Using the
# conservative face-connectivity (rather than 26-connectivity, which also
# counts diagonal/corner touches) avoids reporting a passage as open when a
# rigid ROV body could not actually fit through a diagonal-only gap.
_CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(3, 1)


@dataclass(frozen=True)
class ClearanceObstacle:
    """A single obstacle's axis-aligned bounding box in world coordinates."""

    id: str
    aabb_min: np.ndarray
    """Length-3 array: minimum corner (x, y, z) in metres."""
    aabb_max: np.ndarray
    """Length-3 array: maximum corner (x, y, z) in metres."""

    def __post_init__(self) -> None:
        amin = np.asarray(self.aabb_min, dtype=float)
        amax = np.asarray(self.aabb_max, dtype=float)
        if amin.shape != (3,) or amax.shape != (3,):
            raise ValueError("aabb_min/aabb_max must be length-3 vectors")
        if np.any(amax < amin):
            raise ValueError(
                f"Obstacle '{self.id}' has aabb_max < aabb_min: {amin} vs {amax}"
            )
        object.__setattr__(self, "aabb_min", amin)
        object.__setattr__(self, "aabb_max", amax)


@dataclass
class ROVClearanceResult:
    """Result of 3D volumetric clearance analysis."""

    is_fully_reachable: bool
    """True if the free water volume forms a single connected region."""

    num_disconnected_regions: int
    """Number of disconnected free-space regions (0 if no free space at all)."""

    reachability_ratio: float
    """Voxel count of the largest region / total free voxel count (1.0 = fully connected)."""

    blocking_obstacle_ids: list[str]
    """IDs of obstacles that individually split the free space when present."""

    free_voxel_count: int
    """Total number of navigable (unoccupied) voxels."""

    total_voxel_count: int
    """Total number of voxels in the scene's bounding volume."""


def _grid_shape(bounds_min: np.ndarray, bounds_max: np.ndarray, voxel_size: float):
    extent = bounds_max - bounds_min
    return tuple(max(1, int(np.ceil(v / voxel_size))) for v in extent)


def _aabb_to_index_range(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    bounds_min: np.ndarray,
    voxel_size: float,
    shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Convert a world-space AABB to inclusive-exclusive voxel index ranges."""
    lo = np.floor((aabb_min - bounds_min) / voxel_size).astype(int)
    hi = np.ceil((aabb_max - bounds_min) / voxel_size).astype(int)
    lo = np.clip(lo, 0, np.array(shape))
    hi = np.clip(hi, 0, np.array(shape))
    return tuple(lo), tuple(hi)


def voxelize_occupancy(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    obstacles: list[ClearanceObstacle],
    voxel_size: float,
    rov_radius: float = 0.0,
    exclude_id: str | None = None,
) -> np.ndarray:
    """Build a boolean occupancy grid (True = occupied) over the scene volume.

    Obstacles are inflated by ``rov_radius`` in every direction before
    rasterization, implementing the standard configuration-space trick so the
    ROV can subsequently be treated as a point for connectivity analysis.

    Args:
        bounds_min: Length-3 world-space minimum corner of the scene volume.
        bounds_max: Length-3 world-space maximum corner of the scene volume.
        obstacles: Obstacles to rasterize as occupied space.
        voxel_size: Edge length of each cubic voxel in metres.
        rov_radius: Radius of the ROV body used to inflate obstacles.
        exclude_id: If set, skip the obstacle with this id (used for removal
            tests when identifying which obstacle is blocking a passage).

    Returns:
        Boolean array of shape ``_grid_shape(bounds_min, bounds_max, voxel_size)``.
    """
    bounds_min = np.asarray(bounds_min, dtype=float)
    bounds_max = np.asarray(bounds_max, dtype=float)
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if rov_radius < 0:
        raise ValueError(f"rov_radius must be non-negative, got {rov_radius}")
    if np.any(bounds_max <= bounds_min):
        raise ValueError("bounds_max must be strictly greater than bounds_min")

    shape = _grid_shape(bounds_min, bounds_max, voxel_size)
    occupancy = np.zeros(shape, dtype=bool)

    inflation = np.array([rov_radius, rov_radius, rov_radius])
    for obstacle in obstacles:
        if exclude_id is not None and obstacle.id == exclude_id:
            continue
        inflated_min = obstacle.aabb_min - inflation
        inflated_max = obstacle.aabb_max + inflation
        lo, hi = _aabb_to_index_range(
            inflated_min, inflated_max, bounds_min, voxel_size, shape
        )
        if any(h <= l for l, h in zip(lo, hi)):
            continue  # Obstacle entirely outside the scene volume.
        occupancy[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True

    return occupancy


def _label_free_space(occupancy: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected free-space regions with 6-connectivity."""
    free = ~occupancy
    labels, num = ndimage.label(free, structure=_CONNECTIVITY_STRUCTURE)
    return labels, num


def compute_rov_clearance(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    obstacles: list[ClearanceObstacle],
    voxel_size: float,
    rov_radius: float,
) -> ROVClearanceResult:
    """Compute 3D volumetric clearance for an ROV through a reef scene.

    Analogous to ``reachability.py``'s ``compute_reachability``, but treats
    the ROV as a free-swimming point in an inflated 3D configuration space
    rather than a footprint confined to the floor plane.

    Args:
        bounds_min: Length-3 world-space minimum corner of the scene volume.
        bounds_max: Length-3 world-space maximum corner of the scene volume.
        obstacles: Reef structures / rocks / corals to treat as obstacles.
        voxel_size: Edge length of each cubic voxel in metres (resolution).
        rov_radius: Radius of the ROV body in metres.

    Returns:
        ROVClearanceResult describing 3D connectivity of the free water volume.
    """
    occupancy = voxelize_occupancy(
        bounds_min, bounds_max, obstacles, voxel_size, rov_radius
    )
    free_count = int(np.count_nonzero(~occupancy))
    total_count = int(occupancy.size)

    if free_count == 0:
        console_logger.warning("No free voxels: ROV cannot fit anywhere in scene")
        return ROVClearanceResult(
            is_fully_reachable=False,
            num_disconnected_regions=0,
            reachability_ratio=0.0,
            blocking_obstacle_ids=[],
            free_voxel_count=0,
            total_voxel_count=total_count,
        )

    labels, num_regions = _label_free_space(occupancy)

    if num_regions <= 1:
        reachability_ratio = 1.0
    else:
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # Label 0 is occupied space, not a free region.
        reachability_ratio = float(counts.max()) / free_count

    blockers: list[str] = []
    if num_regions > 1:
        blockers = _identify_blocking_obstacles(
            bounds_min, bounds_max, obstacles, voxel_size, rov_radius, num_regions
        )

    console_logger.info(
        f"ROV clearance: regions={num_regions}, ratio={reachability_ratio:.2f}, "
        f"blockers={blockers}"
    )

    return ROVClearanceResult(
        is_fully_reachable=num_regions == 1,
        num_disconnected_regions=num_regions,
        reachability_ratio=reachability_ratio,
        blocking_obstacle_ids=blockers,
        free_voxel_count=free_count,
        total_voxel_count=total_count,
    )


def _identify_blocking_obstacles(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    obstacles: list[ClearanceObstacle],
    voxel_size: float,
    rov_radius: float,
    baseline_regions: int,
) -> list[str]:
    """Removal test: an obstacle "blocks" a passage if removing it reconnects
    free space into fewer regions."""
    blockers = []
    for obstacle in obstacles:
        occupancy_without = voxelize_occupancy(
            bounds_min,
            bounds_max,
            obstacles,
            voxel_size,
            rov_radius,
            exclude_id=obstacle.id,
        )
        _, regions_without = _label_free_space(occupancy_without)
        if regions_without < baseline_regions:
            blockers.append(obstacle.id)
    return blockers


def find_vertical_bypass_passages(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    obstacles: list[ClearanceObstacle],
    voxel_size: float,
    rov_radius: float,
) -> float:
    """Quantify how much a 3D-aware ROV benefits over a flat 2D ground check.

    Runs both:
    - A 2D-projection check that treats a column as blocked the moment an
      obstacle touches it at ANY height - the same height-blind assumption a
      ground-robot 2D reachability check like ``reachability.py`` makes when
      it subtracts a furniture footprint from the walkable floor without ever
      looking at furniture height.
    - The full 3D volumetric check, which respects each obstacle's real
      vertical extent.

    Returns the fraction of XY columns that the 2D check marks as blocked
    (occupied at any height) but that the 3D check finds contain at least one
    voxel belonging to the main (largest) reachable region - i.e. passages an
    ROV can use by swimming over/under an obstacle that a surface-bound robot
    could never take.

    Args:
        bounds_min: Length-3 world-space minimum corner of the scene volume.
        bounds_max: Length-3 world-space maximum corner of the scene volume.
        obstacles: Reef structures / rocks / corals to treat as obstacles.
        voxel_size: Edge length of each cubic voxel in metres (resolution).
        rov_radius: Radius of the ROV body in metres.

    Returns:
        Fraction in [0, 1] of 2D-blocked columns that are actually 3D-passable.
        Returns 0.0 if the 2D check finds no blocked columns at all.
    """
    occupancy = voxelize_occupancy(
        bounds_min, bounds_max, obstacles, voxel_size, rov_radius
    )
    if occupancy.size == 0:
        return 0.0

    # 2D projection: a column is "blocked" if occupied at ANY height, since a
    # height-blind floor check has no way to know there might be headroom.
    column_fully_occupied = np.any(occupancy, axis=2)
    num_2d_blocked = int(np.count_nonzero(column_fully_occupied))
    if num_2d_blocked == 0:
        return 0.0

    labels, num_regions = _label_free_space(occupancy)
    if num_regions == 0:
        return 0.0
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    main_label = int(np.argmax(counts))

    in_main_region = labels == main_label
    column_has_main_region = np.any(in_main_region, axis=2)

    bypassable = column_fully_occupied & column_has_main_region
    return float(np.count_nonzero(bypassable)) / num_2d_blocked


def format_rov_clearance_result(result: ROVClearanceResult) -> str:
    """Format an ROVClearanceResult as human-readable text (mirrors
    ``reachability.py``'s ``format_reachability_result``)."""
    if result.is_fully_reachable:
        return "Reef scene is fully navigable - all free water volume is connected."

    lines = [
        f"Reef scene has {result.num_disconnected_regions} disconnected "
        "navigable regions",
        f"Volumetric reachability ratio: {result.reachability_ratio:.1%}",
    ]
    if result.blocking_obstacle_ids:
        lines.append(
            "Blocking structures: " + ", ".join(result.blocking_obstacle_ids)
        )
    else:
        lines.append(
            "No single blocking structure identified - reef may need rearrangement"
        )
    return "\n".join(lines)
