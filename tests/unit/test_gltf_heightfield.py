"""Tests for create_heightfield_floor_gltf (heightfield floor mesh generation).

NOT YET RUN by the author of this file - written without a full pydrake/bpy
environment available to visually or physically verify the mesh these tests
build. These tests do check the specific risks flagged in
create_heightfield_floor_gltf's docstring (side-wall winding, watertightness,
bounds/centering math) using pure trimesh/numpy, which doesn't require
pydrake or bpy - please run these before trusting that function.
"""

import tempfile
import unittest

from pathlib import Path

import numpy as np

from reefsmith.agent_utils.seafloor_topography import (
    add_rock_plateau,
    carve_trench,
    flat_seafloor,
)
from reefsmith.utils.gltf_generation import create_heightfield_floor_gltf
from reefsmith.utils.material import Material


def _load_mesh(path: Path):
    """Load a saved GLTF file as a single trimesh.Trimesh, unwrapping a
    Scene if trimesh returns one for this file."""
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    if hasattr(mesh, "geometry"):
        mesh = list(mesh.geometry.values())[0]
    return mesh


class TestCreateHeightfieldFloorGltf(unittest.TestCase):
    """Tests for create_heightfield_floor_gltf."""

    def setUp(self) -> None:
        self.material = Material.from_path(Path("materials/Wood094_1K-JPG"))
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.tmp_dir.name) / "floor.gltf"

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_flat_grid_produces_valid_watertight_mesh(self) -> None:
        """A flat (unsculpted) grid is the degenerate case of the heightfield
        mesh builder - it should still be a valid, closed solid."""
        grid = flat_seafloor(4.0, 4.0, cell_size=1.0)

        create_heightfield_floor_gltf(
            grid=grid,
            thickness=0.1,
            material=self.material,
            output_path=self.output_path,
        )

        assert self.output_path.exists()
        mesh = _load_mesh(self.output_path)
        assert mesh.is_watertight, "flat heightfield floor mesh is not watertight"
        assert (
            mesh.is_winding_consistent
        ), "flat heightfield floor mesh winding is inconsistent"

    def test_sculpted_grid_produces_valid_watertight_mesh(self) -> None:
        """A grid with both a rock plateau and a trench - the case this
        function exists for - should still produce a valid closed solid.

        This is the most important test in this file: it directly checks
        the hand-derived side-wall winding this module's author flagged as
        unverified.
        """
        grid = flat_seafloor(10.0, 8.0, cell_size=0.5, base_height=-2.0)
        grid = add_rock_plateau(grid, np.array([7.0, 6.0]), radius=1.5, height=1.0)
        grid = carve_trench(
            grid, np.array([0.0, 4.0]), np.array([10.0, 4.0]), width=1.0, depth=1.0
        )

        create_heightfield_floor_gltf(
            grid=grid,
            thickness=0.2,
            material=self.material,
            output_path=self.output_path,
        )

        mesh = _load_mesh(self.output_path)
        assert mesh.is_watertight, "sculpted heightfield floor mesh is not watertight"
        assert (
            mesh.is_winding_consistent
        ), "sculpted heightfield floor mesh winding is inconsistent"

    def test_mesh_bounds_match_grid_and_thickness(self) -> None:
        """Top surface follows the grid's own reported heights; bottom is
        `thickness` below the grid's lowest point. Checked against the same
        grid object's .heights.min()/.max(), not a hand-derived number, so
        this doesn't depend on independently re-deriving the plateau math.
        """
        grid = flat_seafloor(4.0, 4.0, cell_size=1.0, base_height=-1.0)
        grid = add_rock_plateau(grid, np.array([2.0, 2.0]), radius=1.0, height=0.5)

        create_heightfield_floor_gltf(
            grid=grid,
            thickness=0.3,
            material=self.material,
            output_path=self.output_path,
        )

        mesh = _load_mesh(self.output_path)

        # GLTF is Y-up; Drake Z-up elevation maps to GLTF Y (zup_to_yup_transform).
        expected_bottom = float(grid.heights.min()) - 0.3
        expected_top = float(grid.heights.max())
        assert abs(float(mesh.bounds[0][1]) - expected_bottom) < 1e-4
        assert abs(float(mesh.bounds[1][1]) - expected_top) < 1e-4

    def test_recenters_grid_at_center_xy(self) -> None:
        """center_x/center_y shift the mesh's footprint, matching
        create_floor_gltf's center-based box convention.

        GLTF X == Drake X; GLTF Z == -Drake Y (zup_to_yup_transform).
        """
        grid = flat_seafloor(4.0, 6.0, cell_size=1.0)

        create_heightfield_floor_gltf(
            grid=grid,
            thickness=0.1,
            material=self.material,
            output_path=self.output_path,
            center_x=10.0,
            center_y=-5.0,
        )

        mesh = _load_mesh(self.output_path)

        assert abs(float(mesh.bounds[0][0]) - (10.0 - 2.0)) < 1e-4
        assert abs(float(mesh.bounds[1][0]) - (10.0 + 2.0)) < 1e-4
        assert abs(float(mesh.bounds[0][2]) - (-(-5.0 + 3.0))) < 1e-4
        assert abs(float(mesh.bounds[1][2]) - (-(-5.0 - 3.0))) < 1e-4


if __name__ == "__main__":
    unittest.main()
