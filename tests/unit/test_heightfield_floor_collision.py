"""Tests for StatefulSeabedAgent._add_heightfield_floor_collision.

NOT YET RUN by the author - written without pydrake/bpy available in this
sandbox. StatefulSeabedAgent's import chain requires the full real
environment (pydrake, bpy via BlenderServer, torch/open_clip via materials
retrieval) just to import, so this file can only run where those are
installed. It only exercises _add_heightfield_floor_collision, a
@staticmethod that builds SDF XML directly from numpy - no BlenderServer or
live agent instance is needed to run these tests.

These tests check the structural properties this module's author could
verify by reasoning (box count, positive heights, bottom/top range) but NOT
actual Drake contact behavior - see the method's docstring for known risks
(box count/performance, flat-top-per-cell approximation error) that still
need checking against a real MultibodyPlant.
"""

import unittest

import lxml.etree as ET
import numpy as np

from reefsmith.agent_utils.seafloor_topography import add_rock_plateau, flat_seafloor
from reefsmith.seabed_agents.stateful_seabed_agent import StatefulSeabedAgent


class TestAddHeightfieldFloorCollision(unittest.TestCase):
    """Tests for the per-cell box collision approximation of sculpted terrain."""

    def _make_sculpted_grid(self):
        grid = flat_seafloor(4.0, 4.0, cell_size=1.0, base_height=-1.0)
        return add_rock_plateau(grid, np.array([2.0, 2.0]), radius=1.0, height=0.5)

    def test_generates_one_collision_box_per_cell(self) -> None:
        grid = self._make_sculpted_grid()
        link = ET.Element("link", name="test_link")

        StatefulSeabedAgent._add_heightfield_floor_collision(
            link, grid=grid, thickness=0.3
        )

        collisions = link.findall("collision")
        nx, ny = grid.shape
        expected_count = (nx - 1) * (ny - 1)
        assert len(collisions) == expected_count

    def test_every_box_has_positive_height(self) -> None:
        """Each box's Z size must be positive - a zero/negative box size
        would be invalid SDF geometry and likely crash Drake's parser."""
        grid = self._make_sculpted_grid()
        link = ET.Element("link", name="test_link")

        StatefulSeabedAgent._add_heightfield_floor_collision(
            link, grid=grid, thickness=0.3
        )

        for collision in link.findall("collision"):
            size_text = collision.find("geometry/box/size").text
            _, _, height = (float(v) for v in size_text.split())
            assert height > 0, f"non-positive collision box height: {size_text}"

    def test_boxes_span_the_full_bottom_to_top_range(self) -> None:
        """The lowest box's bottom face should sit exactly thickness below
        the grid's minimum height, and no box should rise above the grid's
        true maximum height."""
        grid = self._make_sculpted_grid()
        link = ET.Element("link", name="test_link")
        thickness = 0.3

        StatefulSeabedAgent._add_heightfield_floor_collision(
            link, grid=grid, thickness=thickness
        )

        bottoms = []
        tops = []
        for collision in link.findall("collision"):
            size_text = collision.find("geometry/box/size").text
            _, _, height = (float(v) for v in size_text.split())
            pose_text = collision.find("pose").text
            _, _, center_z = (float(v) for v in pose_text.split()[:3])
            bottoms.append(center_z - height / 2.0)
            tops.append(center_z + height / 2.0)

        expected_bottom = float(grid.heights.min()) - thickness
        assert abs(min(bottoms) - expected_bottom) < 1e-6
        # Each box's top is the AVERAGE of its 4 corner heights (see the
        # method's docstring, risk #2), so it should never exceed the
        # grid's true maximum height.
        assert max(tops) <= float(grid.heights.max()) + 1e-6

    def test_flat_grid_still_produces_valid_boxes(self) -> None:
        """A flat grid (degenerate case - callers should normally route flat
        grids to _add_floor_collision instead) shouldn't crash or produce
        zero-height boxes."""
        grid = flat_seafloor(2.0, 2.0, cell_size=1.0, base_height=0.0)
        link = ET.Element("link", name="test_link")

        StatefulSeabedAgent._add_heightfield_floor_collision(
            link, grid=grid, thickness=0.1
        )

        for collision in link.findall("collision"):
            size_text = collision.find("geometry/box/size").text
            _, _, height = (float(v) for v in size_text.split())
            assert abs(height - 0.1) < 1e-9


if __name__ == "__main__":
    unittest.main()
