"""Seafloor topography generation: sand patches, rock foundations, trenches,
drop-offs, and rock-anchor suitability analysis.

This is the underwater-specific geometry the indoor-derived floor plan agent
never had: its ``RoomGeometry`` treats the floor as a single flat plane, with
no notion of depth variation. A real reef seabed is a heightfield - flat sand
patches, elevated rock plateaus that anchor large corals, carved trenches
that channel water current, and drop-offs where the reef wall falls away into
open water.

This module represents the seafloor as a regular heightfield grid (a 2D array
of Z heights, positive = shallower/higher, negative = deeper) and provides
terrain-sculpting operations plus a rock-anchor-suitability scan that replaces
the indoor "room adjacency slot" concept with "locally flat/elevated rock
where a structure can be safely anchored".

Trenches carved here convert directly into
:class:`reefsmith.utils.hydrodynamics.FlowChannel` objects, so the same
geometry that shapes the seabed also drives the current field structures
orient into (see :func:`trenches_to_flow_channels`).

Pure numpy/scipy (with an optional trimesh export), no Drake dependency, so
this is fully unit-testable in isolation. A thin adapter converting the
resulting heightfield into a Drake-compatible SDF/collision mesh is the
natural next integration point for the Seabed Agent. Lives in agent_utils (like sdf_generator.py, mesh_utils.py) rather
than under seabed_agents/tools/, since that package's __init__ eagerly imports
Drake-dependent modules that are unavailable in this environment - keeping this
module dependency-light lets it be imported and unit-tested standalone.
"""

import logging
import math

from dataclasses import dataclass, field

import numpy as np

from scipy.ndimage import maximum_filter

from reefsmith.utils.hydrodynamics import FlowChannel

console_logger = logging.getLogger(__name__)


@dataclass
class SeafloorGrid:
    """A regular heightfield grid representing seafloor elevation.

    ``heights[i, j]`` is the seafloor Z height (metres) at world position
    ``origin + (i, j) * cell_size``. Higher values = shallower / elevated
    rock; lower values = deeper / trenches.
    """

    heights: np.ndarray
    cell_size: float
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self) -> None:
        self.heights = np.asarray(self.heights, dtype=float)
        if self.heights.ndim != 2:
            raise ValueError(f"heights must be 2D, got shape {self.heights.shape}")
        if self.cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {self.cell_size}")
        self.origin = np.asarray(self.origin, dtype=float)
        if self.origin.shape != (2,):
            raise ValueError(f"origin must be shape (2,), got {self.origin.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.heights.shape

    @property
    def world_extent(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) covered by this grid in world coordinates."""
        nx, ny = self.shape
        xmin, ymin = self.origin
        return (
            float(xmin),
            float(xmin + (nx - 1) * self.cell_size),
            float(ymin),
            float(ymin + (ny - 1) * self.cell_size),
        )

    def copy(self) -> "SeafloorGrid":
        return SeafloorGrid(self.heights.copy(), self.cell_size, self.origin.copy())

    def _xy_to_grid(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy, dtype=float) - self.origin) / self.cell_size

    def height_at(self, xy: np.ndarray) -> float:
        """Bilinearly interpolated seafloor height at a world XY point.

        Points outside the grid are clamped to the nearest edge.
        """
        gx, gy = self._xy_to_grid(xy)
        nx, ny = self.shape
        gx = np.clip(gx, 0, nx - 1)
        gy = np.clip(gy, 0, ny - 1)

        i0, j0 = int(np.floor(gx)), int(np.floor(gy))
        i1, j1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1)
        tx, ty = gx - i0, gy - j0

        h00 = self.heights[i0, j0]
        h10 = self.heights[i1, j0]
        h01 = self.heights[i0, j1]
        h11 = self.heights[i1, j1]
        h0 = h00 * (1 - tx) + h10 * tx
        h1 = h01 * (1 - tx) + h11 * tx
        return float(h0 * (1 - ty) + h1 * ty)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage (e.g. checkpoint/resume)."""
        return {
            "heights": self.heights.tolist(),
            "cell_size": self.cell_size,
            "origin": self.origin.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SeafloorGrid":
        """Deserialize from dictionary produced by :meth:`to_dict`."""
        return cls(
            heights=np.array(data["heights"], dtype=float),
            cell_size=data["cell_size"],
            origin=np.array(data["origin"], dtype=float),
        )

    def to_trimesh(self):
        """Convert the heightfield to a triangulated surface mesh.

        Imports ``trimesh`` lazily so this module has no hard dependency on
        it - callers that only need the heightfield/anchor-analysis math
        never need trimesh installed.
        """
        import trimesh

        nx, ny = self.shape
        xs = self.origin[0] + np.arange(nx) * self.cell_size
        ys = self.origin[1] + np.arange(ny) * self.cell_size
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        vertices = np.stack([xx, yy, self.heights], axis=-1).reshape(-1, 3)

        faces = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                v00 = i * ny + j
                v10 = (i + 1) * ny + j
                v01 = i * ny + (j + 1)
                v11 = (i + 1) * ny + (j + 1)
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def flat_seafloor(
    width_m: float, length_m: float, cell_size: float, base_height: float = 0.0
) -> SeafloorGrid:
    """Create a flat sand-patch seafloor grid (the starting point before
    sculpting rock foundations, trenches, and drop-offs).

    Args:
        width_m: Extent along X in metres.
        length_m: Extent along Y in metres.
        cell_size: Grid resolution in metres.
        base_height: Uniform starting seafloor height.
    """
    if width_m <= 0 or length_m <= 0:
        raise ValueError("width_m and length_m must be positive")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    nx = max(2, int(round(width_m / cell_size)) + 1)
    ny = max(2, int(round(length_m / cell_size)) + 1)
    heights = np.full((nx, ny), base_height, dtype=float)
    return SeafloorGrid(heights, cell_size, origin=np.zeros(2))


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Smooth 0->1 falloff for values of t in [0, 1] (clamped outside)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def add_rock_plateau(
    grid: SeafloorGrid,
    center_xy: np.ndarray,
    radius: float,
    height: float,
    falloff: float | None = None,
) -> SeafloorGrid:
    """Raise a rock plateau/foundation, a flat-topped anchor point for large
    structural corals or boulders.

    Combines with existing terrain via ``max()`` so overlapping plateaus
    merge into a single foundation rather than stacking additively.

    Args:
        grid: Seafloor to modify (not mutated; a new grid is returned).
        center_xy: World XY center of the plateau.
        radius: Radius of the flat plateau top in metres.
        height: Height of the plateau above its base (must be positive).
        falloff: Extra distance over which the plateau smoothly blends back
            down to the surrounding terrain. Defaults to ``radius / 2``.

    Returns:
        A new SeafloorGrid with the plateau added.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    falloff = radius / 2.0 if falloff is None else falloff
    if falloff < 0:
        raise ValueError(f"falloff must be non-negative, got {falloff}")

    new_grid = grid.copy()
    nx, ny = grid.shape
    xs = grid.origin[0] + np.arange(nx) * grid.cell_size
    ys = grid.origin[1] + np.arange(ny) * grid.cell_size
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    dist = np.sqrt((xx - center_xy[0]) ** 2 + (yy - center_xy[1]) ** 2)

    # 1.0 inside radius, smoothly falls to 0.0 over [radius, radius+falloff].
    if falloff > 0:
        t = (dist - radius) / falloff
        profile = 1.0 - _smoothstep(t)
    else:
        profile = (dist <= radius).astype(float)

    # NOTE: the plateau's contribution is an absolute local elevation target
    # (0 outside its influence, rising to `height` at the center), combined
    # via max() rather than added onto the current heights. This keeps
    # repeated/overlapping plateau calls idempotent (adding the same plateau
    # twice does not double-stack it) and makes multiple overlapping
    # plateaus merge to whichever is taller at each point, rather than
    # summing. Because the target floors out at 0 outside the influence
    # radius, plateaus should generally be added to terrain that is not
    # already carved below 0 in that footprint (typical usage: carve
    # trenches/drop-offs first, then place plateaus at anchor zones found
    # via find_anchor_zones, which already excludes low/trenched terrain).
    target = profile * height
    new_grid.heights = np.maximum(grid.heights, target)
    return new_grid


def carve_trench(
    grid: SeafloorGrid,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    width: float,
    depth: float,
    falloff: float | None = None,
) -> SeafloorGrid:
    """Carve a linear trench/channel into the seafloor - a natural flow
    channel that keeps water current (and ROV passage) unobstructed.

    Combines with existing terrain via ``min()`` so trenches only ever cut
    down, never raise the floor.

    Args:
        grid: Seafloor to modify (not mutated; a new grid is returned).
        start_xy: World XY start point of the trench centerline.
        end_xy: World XY end point of the trench centerline.
        width: Width of the trench floor in metres.
        depth: Depth of the trench below the surrounding terrain.
        falloff: Extra distance over which trench walls blend back up to the
            surrounding terrain. Defaults to ``width / 2``.

    Returns:
        A new SeafloorGrid with the trench carved in.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    falloff = width / 2.0 if falloff is None else falloff
    if falloff < 0:
        raise ValueError(f"falloff must be non-negative, got {falloff}")

    start_xy = np.asarray(start_xy, dtype=float)
    end_xy = np.asarray(end_xy, dtype=float)
    new_grid = grid.copy()
    nx, ny = grid.shape
    xs = grid.origin[0] + np.arange(nx) * grid.cell_size
    ys = grid.origin[1] + np.arange(ny) * grid.cell_size
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    seg = end_xy - start_xy
    seg_len_sq = float(seg @ seg)
    if seg_len_sq < 1e-12:
        dist = np.sqrt((xx - start_xy[0]) ** 2 + (yy - start_xy[1]) ** 2)
    else:
        px = xx - start_xy[0]
        py = yy - start_xy[1]
        t = np.clip((px * seg[0] + py * seg[1]) / seg_len_sq, 0.0, 1.0)
        closest_x = start_xy[0] + t * seg[0]
        closest_y = start_xy[1] + t * seg[1]
        dist = np.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)

    half_width = width / 2.0
    if falloff > 0:
        t = (dist - half_width) / falloff
        profile = 1.0 - _smoothstep(t)
    else:
        profile = (dist <= half_width).astype(float)

    carved = grid.heights - profile * depth
    new_grid.heights = np.minimum(grid.heights, carved)
    return new_grid


def carve_drop_off(
    grid: SeafloorGrid,
    edge_start_xy: np.ndarray,
    edge_end_xy: np.ndarray,
    drop_depth: float,
    transition_width: float = 1.0,
) -> SeafloorGrid:
    """Carve a drop-off: the seafloor steps down beyond a line, as at a reef
    wall falling away into open water.

    The side that drops is determined by the sign of the cross product of
    the edge direction with the vector to each grid point: the region to the
    right of ``edge_start_xy -> edge_end_xy`` (i.e. in the direction of
    ``edge_dir`` rotated -90 degrees) drops; the left stays put.

    Args:
        grid: Seafloor to modify (not mutated; a new grid is returned).
        edge_start_xy: World XY start point of the drop-off edge line.
        edge_end_xy: World XY end point of the drop-off edge line.
        drop_depth: How far the seafloor drops beyond the edge.
        transition_width: Distance over which the drop transitions smoothly.

    Returns:
        A new SeafloorGrid with the drop-off carved in.
    """
    if drop_depth <= 0:
        raise ValueError(f"drop_depth must be positive, got {drop_depth}")
    if transition_width <= 0:
        raise ValueError(f"transition_width must be positive, got {transition_width}")

    edge_start_xy = np.asarray(edge_start_xy, dtype=float)
    edge_end_xy = np.asarray(edge_end_xy, dtype=float)
    edge_dir = edge_end_xy - edge_start_xy
    edge_len = np.linalg.norm(edge_dir)
    if edge_len < 1e-12:
        raise ValueError("edge_start_xy and edge_end_xy must differ")

    new_grid = grid.copy()
    nx, ny = grid.shape
    xs = grid.origin[0] + np.arange(nx) * grid.cell_size
    ys = grid.origin[1] + np.arange(ny) * grid.cell_size
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    px = xx - edge_start_xy[0]
    py = yy - edge_start_xy[1]
    # Signed perpendicular distance to the (infinite) edge line.
    signed_dist = (py * edge_dir[0] - px * edge_dir[1]) / edge_len

    # 0 well before the edge, 1 well beyond it, smooth transition between.
    t = signed_dist / transition_width
    profile = _smoothstep(t)

    new_grid.heights = grid.heights - profile * drop_depth
    return new_grid


def compute_slope_field(grid: SeafloorGrid) -> np.ndarray:
    """Local slope magnitude (rise/run, dimensionless) at every grid node,
    via central finite differences."""
    gy, gx = np.gradient(grid.heights, grid.cell_size)
    return np.sqrt(gx**2 + gy**2)


@dataclass(frozen=True)
class AnchorZone:
    """A candidate rock-foundation anchor point for a large reef structure."""

    center: np.ndarray
    """World XY center of the anchor zone."""
    radius: float
    """Radius over which this anchor zone was evaluated, in metres."""
    mean_height: float
    """Mean seafloor height within the zone."""
    max_slope: float
    """Maximum local slope magnitude within the zone."""


def find_anchor_zones(
    grid: SeafloorGrid,
    search_radius: float,
    max_slope: float,
    min_height: float | None = None,
    grid_stride: int = 1,
) -> list[AnchorZone]:
    """Scan the seafloor for locally flat, sufficiently elevated rock where a
    large structure (boulder, massive coral) can be safely anchored.

    This replaces the indoor floor-plan agent's "room adjacency slot" concept
    (discrete rectangle-edge attachment points) with a terrain-aware notion
    of anchor suitability: a candidate point is suitable if the seafloor
    around it is not too steep (won't destabilize a heavy structure) and,
    optionally, is elevated enough to be rock rather than a sand patch or
    trench floor.

    Candidates are found by scanning all grid nodes and keeping local maxima
    of a "flatness" score via a maximum filter, then greedily suppressing
    candidates that are closer together than ``search_radius`` (so the
    result is a small set of well-separated anchor points rather than every
    grid node that happens to qualify).

    Args:
        grid: Seafloor to analyze.
        search_radius: Radius over which local height/slope are averaged,
            and the minimum separation enforced between returned anchors.
        max_slope: Maximum acceptable local slope magnitude for a candidate.
        min_height: If set, only consider candidates with mean height at or
            above this value (i.e. require rock, not trench/sand).
        grid_stride: Subsample candidate nodes every N cells for speed on
            large grids (1 = check every node).

    Returns:
        List of AnchorZone, sorted by suitability (flattest/highest first).
    """
    if search_radius <= 0:
        raise ValueError(f"search_radius must be positive, got {search_radius}")
    if max_slope <= 0:
        raise ValueError(f"max_slope must be positive, got {max_slope}")

    slope_field = compute_slope_field(grid)
    radius_cells = max(1, int(round(search_radius / grid.cell_size)))

    # Local mean height and max slope via uniform/maximum filters (vectorized,
    # avoids an O(n^2) python loop over every grid node).
    from scipy.ndimage import uniform_filter

    mean_height_field = uniform_filter(grid.heights, size=2 * radius_cells + 1)
    max_slope_field = maximum_filter(slope_field, size=2 * radius_cells + 1)

    nx, ny = grid.shape
    candidates: list[AnchorZone] = []
    for i in range(0, nx, grid_stride):
        for j in range(0, ny, grid_stride):
            local_max_slope = float(max_slope_field[i, j])
            if local_max_slope > max_slope:
                continue
            local_mean_height = float(mean_height_field[i, j])
            if min_height is not None and local_mean_height < min_height:
                continue
            world_x = grid.origin[0] + i * grid.cell_size
            world_y = grid.origin[1] + j * grid.cell_size
            candidates.append(
                AnchorZone(
                    center=np.array([world_x, world_y]),
                    radius=search_radius,
                    mean_height=local_mean_height,
                    max_slope=local_max_slope,
                )
            )

    # Sort best-first: flattest and (among ties) highest/shallowest.
    candidates.sort(key=lambda a: (a.max_slope, -a.mean_height))

    # Greedy non-maximum suppression: keep a candidate only if it is not too
    # close to an already-accepted (better) candidate.
    accepted: list[AnchorZone] = []
    min_separation = search_radius
    for cand in candidates:
        if all(
            np.linalg.norm(cand.center - a.center) >= min_separation for a in accepted
        ):
            accepted.append(cand)

    console_logger.info(
        f"Found {len(accepted)} anchor zone(s) from {len(candidates)} candidates "
        f"(max_slope<={max_slope}, min_height={min_height})"
    )
    return accepted


@dataclass(frozen=True)
class TrenchSpec:
    """Specification of a carved trench, kept alongside the terrain so it
    can later be converted into a hydrodynamic flow channel."""

    start_xy: np.ndarray
    end_xy: np.ndarray
    width: float
    depth: float


def trenches_to_flow_channels(
    trenches: list[TrenchSpec], grid: SeafloorGrid
) -> list[FlowChannel]:
    """Convert carved trenches into hydrodynamic FlowChannel objects.

    The trench's Z position is taken as the local seafloor height at its
    midpoint minus half its depth (a representative mid-channel height), and
    its ``speed_gain`` grows with the ratio of depth to width - modeling the
    real venturi-like effect where water accelerates through a narrower,
    deeper channel. Gain is capped at a physically reasonable 3x.

    Args:
        trenches: Trench specifications carved into the seafloor.
        grid: The seafloor grid the trenches were carved into (used to look
            up representative Z heights for the resulting 3D flow channels).

    Returns:
        List of FlowChannel objects usable with
        :class:`reefsmith.utils.hydrodynamics.WaterCurrentField`.
    """
    channels = []
    for trench in trenches:
        mid_xy = (trench.start_xy + trench.end_xy) / 2.0
        mid_height = grid.height_at(mid_xy) - trench.depth / 2.0

        start_z = grid.height_at(trench.start_xy) - trench.depth / 2.0
        end_z = grid.height_at(trench.end_xy) - trench.depth / 2.0
        # Use the trench's own endpoints in 3D (start/end height, not the
        # single midpoint estimate) for a more accurate channel axis.
        start_3d = np.array([trench.start_xy[0], trench.start_xy[1], start_z])
        end_3d = np.array([trench.end_xy[0], trench.end_xy[1], end_z])

        aspect = trench.depth / trench.width
        speed_gain = min(1.0 + aspect_scaled(aspect), 3.0)

        channels.append(
            FlowChannel(
                start=start_3d,
                end=end_3d,
                radius=trench.width / 2.0,
                speed_gain=speed_gain,
            )
        )
    return channels


def aspect_scaled(aspect: float) -> float:
    """Internal helper: dimensionless speed boost from a trench's depth/width
    aspect ratio, saturating smoothly rather than growing without bound."""
    return 2.0 * math.tanh(aspect)
