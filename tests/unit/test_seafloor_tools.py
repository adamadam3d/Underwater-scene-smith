"""Tests for seafloor terrain sculpting tools (SeafloorTools)."""

import unittest

import numpy as np

from reefsmith.agent_utils.house import HouseLayout, PlacedRoom, RoomSpec
from reefsmith.seabed_agents.tools.seafloor_tools import SeafloorTools


class TestSeafloorTools(unittest.TestCase):
    """Test carving, sculpting, and querying a room's seafloor terrain."""

    def setUp(self) -> None:
        self.layout = HouseLayout()
        self.layout.room_specs = [RoomSpec(room_id="reef_main", width=8.0, length=10.0)]
        self.layout.placed_rooms = [
            PlacedRoom(
                room_id="reef_main", position=(0.0, 0.0), width=10.0, depth=8.0, walls=[]
            )
        ]
        self.tools = SeafloorTools(layout=self.layout, cell_size=0.5)

    def test_tool_closures_registered(self) -> None:
        """All three tools are exposed by name."""
        assert set(self.tools.tools.keys()) == {
            "carve_seafloor_trench",
            "add_seafloor_rock_plateau",
            "find_seafloor_anchor_zones",
        }

    def test_carve_trench_lowers_seafloor_and_invalidates_geometry(self) -> None:
        """Carving a trench updates layout.seafloor_grids and invalidates any
        cached RoomGeometry for that room."""
        self.layout.room_geometries["reef_main"] = object()  # Stand-in cached geometry.

        result = self.tools._carve_seafloor_trench_impl(
            "reef_main", start_x=0.0, start_y=4.0, end_x=10.0, end_y=4.0,
            width=1.0, depth=1.5,
        )

        assert result.success, result.message
        grid = self.layout.seafloor_grids["reef_main"]
        assert grid.height_at(np.array([5.0, 4.0])) < -1.0
        assert "reef_main" not in self.layout.room_geometries

    def test_add_rock_plateau_raises_seafloor(self) -> None:
        """Adding a rock plateau raises terrain near its center."""
        result = self.tools._add_seafloor_rock_plateau_impl(
            "reef_main", center_x=2.0, center_y=2.0, radius=1.0, height=0.8
        )

        assert result.success, result.message
        grid = self.layout.seafloor_grids["reef_main"]
        assert grid.height_at(np.array([2.0, 2.0])) > 0.5

    def test_find_anchor_zones_returns_candidates_on_flat_terrain(self) -> None:
        """Flat (unsculpted) terrain has plenty of suitable anchor zones."""
        result = self.tools._find_seafloor_anchor_zones_impl(
            "reef_main", search_radius=1.0, max_slope=0.5, min_height=-1e9
        )

        assert result.success, result.message
        assert len(result.zones) > 0

    def test_unknown_room_fails_gracefully(self) -> None:
        """Operating on a room that doesn't exist/isn't placed fails cleanly,
        not with an exception."""
        trench_result = self.tools._carve_seafloor_trench_impl(
            "nonexistent", start_x=0.0, start_y=0.0, end_x=1.0, end_y=0.0,
            width=1.0, depth=1.0,
        )
        assert not trench_result.success

        plateau_result = self.tools._add_seafloor_rock_plateau_impl(
            "nonexistent", center_x=0.0, center_y=0.0, radius=1.0, height=1.0
        )
        assert not plateau_result.success

        zones_result = self.tools._find_seafloor_anchor_zones_impl(
            "nonexistent", search_radius=1.0, max_slope=0.5, min_height=-1e9
        )
        assert not zones_result.success

    def test_invalid_parameters_fail_without_raising(self) -> None:
        """Bad geometry parameters surface as Result(success=False), not a
        raised exception (seafloor_topography raises ValueError internally)."""
        result = self.tools._carve_seafloor_trench_impl(
            "reef_main", start_x=0.0, start_y=0.0, end_x=1.0, end_y=0.0,
            width=0.0, depth=1.0,
        )
        assert not result.success
        assert "width" in result.message.lower()

    def test_resize_resets_sculpted_terrain_to_flat(self) -> None:
        """Resizing a room (dimension mismatch) resets its terrain, since the
        old heightfield no longer matches the new bounds."""
        self.tools._add_seafloor_rock_plateau_impl(
            "reef_main", center_x=5.0, center_y=4.0, radius=1.0, height=1.0
        )
        self.layout.placed_rooms[0] = PlacedRoom(
            room_id="reef_main", position=(0.0, 0.0), width=6.0, depth=5.0, walls=[]
        )

        result = self.tools._find_seafloor_anchor_zones_impl(
            "reef_main", search_radius=1.0, max_slope=0.5, min_height=-1e9
        )
        assert result.success
        grid = self.layout.seafloor_grids["reef_main"]
        assert np.allclose(grid.heights, 0.0)


if __name__ == "__main__":
    unittest.main()
