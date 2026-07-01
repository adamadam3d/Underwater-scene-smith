"""Seabed terrain sculpting tools for the Seabed Agent.

Wires reefsmith.agent_utils.seafloor_topography (pure numpy/scipy heightfield
math) into the agent-facing tool surface, following the same closure pattern
as FloorPlanTools. Terrain is stored per-room on HouseLayout.seafloor_grids
(via HouseLayout.get_or_init_seafloor_grid) and consumed by
StatefulSeabedAgent._generate_room_geometry when building each room's
RoomGeometry.

Sculpting a room's terrain does not yet change its floor mesh/collision -
those are still generated as a flat box (see _generate_floor_geometry in
stateful_seabed_agent.py). Wiring the heightfield into GLTF/SDF export and
Blender rendering is a separate, larger change that requires validating
mesh-collision behavior against real pydrake/Blender, which isn't available
in this environment.
"""

import logging

from dataclasses import dataclass, field

import numpy as np

from agents import function_tool

from reefsmith.agent_utils.house import HouseLayout
from reefsmith.agent_utils.seafloor_topography import (
    add_rock_plateau,
    carve_trench,
    find_anchor_zones,
)

console_logger = logging.getLogger(__name__)


@dataclass
class Result:
    """Generic result from seafloor tools."""

    success: bool
    message: str


@dataclass
class AnchorZoneInfo:
    """A single anchor zone candidate, JSON-serializable for tool output."""

    center_x: float
    center_y: float
    mean_height: float
    max_slope: float


@dataclass
class AnchorZonesResult:
    """Result from find_seafloor_anchor_zones tool."""

    success: bool
    message: str
    zones: list[AnchorZoneInfo] = field(default_factory=list)


class SeafloorTools:
    """Tools for sculpting seafloor terrain and querying anchor-suitable zones.

    Coordinates for all tools are room-local meters, origin at the room's
    min corner (matching PlacedRoom.width/depth extents).
    """

    def __init__(self, layout: HouseLayout, cell_size: float = 0.5):
        """Initialize seafloor tools.

        Args:
            layout: The HouseLayout to modify.
            cell_size: Grid resolution in meters, used only when a room's
                terrain hasn't been sculpted yet and a flat grid is created.
        """
        self.layout = layout
        self.cell_size = cell_size
        self.tools = self._create_tool_closures()

    def _fail(self, message: str) -> Result:
        """Log failure and return Result with success=False."""
        console_logger.info(f"Tool failed: {message}")
        return Result(success=False, message=message)

    def _create_tool_closures(self) -> dict:
        """Create tool closures with access to instance data.

        Uses closure pattern to avoid including 'self' in OpenAI function
        schemas (matches FloorPlanTools._create_tool_closures).
        """

        @function_tool
        def carve_seafloor_trench(
            room_id: str,
            start_x: float,
            start_y: float,
            end_x: float,
            end_y: float,
            width: float,
            depth: float,
        ) -> Result:
            """Carve a trench/flow channel into a room's seafloor.

            A natural channel that keeps water current and ROV passage
            unobstructed. Coordinates are room-local meters (origin at the
            room's min corner).

            Args:
                room_id: Room to sculpt.
                start_x: Trench centerline start X in meters.
                start_y: Trench centerline start Y in meters.
                end_x: Trench centerline end X in meters.
                end_y: Trench centerline end Y in meters.
                width: Trench floor width in meters.
                depth: Trench depth below surrounding terrain in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._carve_seafloor_trench_impl(
                room_id, start_x, start_y, end_x, end_y, width, depth
            )

        @function_tool
        def add_seafloor_rock_plateau(
            room_id: str,
            center_x: float,
            center_y: float,
            radius: float,
            height: float,
        ) -> Result:
            """Raise a flat-topped rock plateau in a room's seafloor.

            Use as an anchor point for large structural corals or boulders.
            Coordinates are room-local meters (origin at the room's min
            corner).

            Args:
                room_id: Room to sculpt.
                center_x: Plateau center X in meters.
                center_y: Plateau center Y in meters.
                radius: Radius of the flat plateau top in meters.
                height: Height of the plateau above its base in meters.

            Returns:
                Result indicating success or failure.
            """
            return self._add_seafloor_rock_plateau_impl(
                room_id, center_x, center_y, radius, height
            )

        @function_tool
        def find_seafloor_anchor_zones(
            room_id: str,
            search_radius: float,
            max_slope: float,
            min_height: float = -1e9,
        ) -> AnchorZonesResult:
            """Find locally flat, elevated spots suitable for anchoring large
            reef structures (boulders, massive coral) on a room's seafloor.

            Args:
                room_id: Room to analyze.
                search_radius: Radius in meters over which flatness is
                    scored, and minimum separation enforced between returned
                    zones.
                max_slope: Maximum acceptable local slope (rise/run).
                min_height: Minimum mean height to qualify (require rock,
                    not trench/sand). Leave at the default for no filter.

            Returns:
                AnchorZonesResult with candidate zones, best (flattest,
                then highest) first.
            """
            return self._find_seafloor_anchor_zones_impl(
                room_id, search_radius, max_slope, min_height
            )

        return {
            "carve_seafloor_trench": carve_seafloor_trench,
            "add_seafloor_rock_plateau": add_seafloor_rock_plateau,
            "find_seafloor_anchor_zones": find_seafloor_anchor_zones,
        }

    def _carve_seafloor_trench_impl(
        self,
        room_id: str,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        width: float,
        depth: float,
    ) -> Result:
        console_logger.info(
            f"Tool called: carve_seafloor_trench(room_id={room_id}, "
            f"start=({start_x}, {start_y}), end=({end_x}, {end_y}), "
            f"width={width}, depth={depth})"
        )
        grid = self.layout.get_or_init_seafloor_grid(room_id, cell_size=self.cell_size)
        if grid is None:
            return self._fail(f"Room '{room_id}' not found or not yet placed.")

        try:
            new_grid = carve_trench(
                grid,
                start_xy=np.array([start_x, start_y]),
                end_xy=np.array([end_x, end_y]),
                width=width,
                depth=depth,
            )
        except ValueError as e:
            return self._fail(str(e))

        self.layout.seafloor_grids[room_id] = new_grid
        if self.layout.invalidate_room_geometry(room_id):
            console_logger.debug(f"Invalidated geometry for room: {room_id}")

        return Result(success=True, message=f"Carved trench into '{room_id}' seafloor.")

    def _add_seafloor_rock_plateau_impl(
        self,
        room_id: str,
        center_x: float,
        center_y: float,
        radius: float,
        height: float,
    ) -> Result:
        console_logger.info(
            f"Tool called: add_seafloor_rock_plateau(room_id={room_id}, "
            f"center=({center_x}, {center_y}), radius={radius}, height={height})"
        )
        grid = self.layout.get_or_init_seafloor_grid(room_id, cell_size=self.cell_size)
        if grid is None:
            return self._fail(f"Room '{room_id}' not found or not yet placed.")

        try:
            new_grid = add_rock_plateau(
                grid,
                center_xy=np.array([center_x, center_y]),
                radius=radius,
                height=height,
            )
        except ValueError as e:
            return self._fail(str(e))

        self.layout.seafloor_grids[room_id] = new_grid
        if self.layout.invalidate_room_geometry(room_id):
            console_logger.debug(f"Invalidated geometry for room: {room_id}")

        return Result(
            success=True, message=f"Added rock plateau to '{room_id}' seafloor."
        )

    def _find_seafloor_anchor_zones_impl(
        self,
        room_id: str,
        search_radius: float,
        max_slope: float,
        min_height: float,
    ) -> AnchorZonesResult:
        console_logger.info(
            f"Tool called: find_seafloor_anchor_zones(room_id={room_id}, "
            f"search_radius={search_radius}, max_slope={max_slope}, "
            f"min_height={min_height})"
        )
        grid = self.layout.get_or_init_seafloor_grid(room_id, cell_size=self.cell_size)
        if grid is None:
            return AnchorZonesResult(
                success=False, message=f"Room '{room_id}' not found or not yet placed."
            )

        # -1e9 is the "no filter" sentinel (OpenAI function schemas don't
        # support Optional[float] defaults cleanly); matches the empty-string
        # sentinel convention used elsewhere in this tool surface, e.g.
        # FloorPlanTools.set_room_materials's floor_material_id="".
        effective_min_height = None if min_height <= -1e9 else min_height
        try:
            zones = find_anchor_zones(
                grid,
                search_radius=search_radius,
                max_slope=max_slope,
                min_height=effective_min_height,
            )
        except ValueError as e:
            return AnchorZonesResult(success=False, message=str(e))

        zone_infos = [
            AnchorZoneInfo(
                center_x=float(z.center[0]),
                center_y=float(z.center[1]),
                mean_height=z.mean_height,
                max_slope=z.max_slope,
            )
            for z in zones
        ]
        return AnchorZonesResult(
            success=True,
            message=f"Found {len(zone_infos)} anchor zone(s).",
            zones=zone_infos,
        )
