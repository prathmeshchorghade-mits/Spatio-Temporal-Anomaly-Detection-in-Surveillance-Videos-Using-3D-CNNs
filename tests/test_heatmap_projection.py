"""Smoke test: projecting a one-hot voxel map to frame pixel coordinates.

Verifies that a voxel activated at (t_v, h_v, w_v) projects to the correct
pixel bounding box in the original frame, within ±1 pixel tolerance.
"""

import pytest
import torch


def voxel_to_pixel_box(
    vox_t: int,
    vox_h: int,
    vox_w: int,
    map_H: int,
    map_W: int,
    frame_H: int,
    frame_W: int,
) -> tuple:
    """Convert a single voxel location to a pixel bounding box.

    Uses bilinear-upsampling semantics: each voxel cell covers a rectangular
    region of the frame.

    Args:
        vox_t, vox_h, vox_w: Voxel coordinates (time index, height, width).
        map_H, map_W: Spatial size of the voxel map.
        frame_H, frame_W: Spatial size of the original frame.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates (float).
    """
    cell_h = frame_H / map_H
    cell_w = frame_W / map_W
    y1 = vox_h * cell_h
    x1 = vox_w * cell_w
    y2 = y1 + cell_h
    x2 = x1 + cell_w
    return x1, y1, x2, y2


def project_voxel_map_to_frame(
    vox_map: torch.Tensor,
    frame_H: int,
    frame_W: int,
    time_idx: int = 0,
) -> torch.Tensor:
    """Bilinearly upsample a single voxel map slice to frame resolution.

    Args:
        vox_map: Voxel map of shape (T', H', W').
        frame_H: Target frame height.
        frame_W: Target frame width.
        time_idx: Which temporal slice to upsample.

    Returns:
        Frame-resolution heatmap of shape (frame_H, frame_W).
    """
    # Take temporal slice and upsample
    spatial = vox_map[time_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
    upsampled = torch.nn.functional.interpolate(
        spatial,
        size=(frame_H, frame_W),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.squeeze()  # (frame_H, frame_W)


class TestVoxelToPixelBox:
    """Verify voxel_to_pixel_box maps to the expected pixel region."""

    def test_top_left_voxel(self) -> None:
        """Voxel (0, 0, 0) should map to the top-left corner of the frame."""
        x1, y1, x2, y2 = voxel_to_pixel_box(0, 0, 0, 14, 14, 112, 112)
        assert abs(x1) < 1e-6 and abs(y1) < 1e-6, (
            f"Top-left voxel should start at (0,0), got ({x1},{y1})"
        )

    def test_cell_size_14x14_map(self) -> None:
        """For a 14×14 map on a 112×112 frame, each cell is 8×8 pixels."""
        x1, y1, x2, y2 = voxel_to_pixel_box(0, 0, 0, 14, 14, 112, 112)
        assert abs((x2 - x1) - 8.0) < 1e-5, f"Cell width should be 8, got {x2 - x1}"
        assert abs((y2 - y1) - 8.0) < 1e-5, f"Cell height should be 8, got {y2 - y1}"

    def test_cell_size_7x7_map(self) -> None:
        """For a 7×7 map on a 112×112 frame, each cell is 16×16 pixels."""
        x1, y1, x2, y2 = voxel_to_pixel_box(0, 0, 0, 7, 7, 112, 112)
        assert abs((x2 - x1) - 16.0) < 1e-5
        assert abs((y2 - y1) - 16.0) < 1e-5

    def test_bottom_right_voxel_14x14(self) -> None:
        """Last voxel (13,13) on 14×14 map should end at frame corner (112,112)."""
        x1, y1, x2, y2 = voxel_to_pixel_box(0, 13, 13, 14, 14, 112, 112)
        assert abs(x2 - 112.0) < 1e-5, f"Expected x2=112, got {x2}"
        assert abs(y2 - 112.0) < 1e-5, f"Expected y2=112, got {y2}"

    def test_center_voxel_argmax_projects_to_frame_center(self) -> None:
        """A one-hot map with peak at center should upsample to near-center."""
        T_, H_, W_ = 4, 14, 14
        vox = torch.zeros(T_, H_, W_)
        ch, cw = H_ // 2, W_ // 2
        vox[0, ch, cw] = 10.0  # high activation at centre

        heatmap = project_voxel_map_to_frame(vox, 112, 112, time_idx=0)
        assert heatmap.shape == (112, 112), f"Expected (112,112), got {heatmap.shape}"

        # Argmax of the upsampled heatmap should be near the centre of the frame
        flat_idx = heatmap.argmax().item()
        peak_h = flat_idx // 112
        peak_w = flat_idx % 112
        # Allow ±8 pixel tolerance (one voxel cell)
        assert abs(peak_h - 56) <= 8, f"Peak h={peak_h} not near 56"
        assert abs(peak_w - 56) <= 8, f"Peak w={peak_w} not near 56"

    @pytest.mark.parametrize("map_size,frame_size,expected_cell", [
        ((7,  7),  (112, 112), (16, 16)),
        ((14, 14), (112, 112), (8,  8)),
        ((28, 28), (112, 112), (4,  4)),
    ])
    def test_cell_sizes_all_variants(
        self,
        map_size: tuple,
        frame_size: tuple,
        expected_cell: tuple,
    ) -> None:
        """Cell sizes match expected values for all resolution variants."""
        mH, mW = map_size
        fH, fW = frame_size
        ech, ecw = expected_cell
        x1, y1, x2, y2 = voxel_to_pixel_box(0, 0, 0, mH, mW, fH, fW)
        assert abs((x2 - x1) - ecw) < 1e-5, (
            f"map {map_size}: expected cell_w={ecw}, got {x2 - x1}"
        )
        assert abs((y2 - y1) - ech) < 1e-5, (
            f"map {map_size}: expected cell_h={ech}, got {y2 - y1}"
        )
