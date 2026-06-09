from __future__ import annotations

from pathlib import Path

import numpy as np

from .colors import ERROR_COLOR, GT_COLOR, frame_color_plan
from .geometry import cell_center_zyx, make_all_rings
from .io import load_tif, read_cells, read_gt, resolve_frame_tif


def open_review_viewer(
    frame: int,
    cells_csv: Path,
    gt_csv: Path | None = None,
    run_dir: Path | None = None,
    real_tif: Path | None = None,
    synth_tif: Path | None = None,
):
    import napari

    if run_dir is not None:
        real_tif = real_tif or resolve_frame_tif(run_dir, frame, "real")
        synth_tif = synth_tif or resolve_frame_tif(run_dir, frame, "synth")

    cells = read_cells(cells_csv, frame)
    gt = read_gt(gt_csv, frame) if gt_csv is not None else []
    colors, missing_labels, _, extra_names = frame_color_plan(cells, gt, frame)

    viewer = napari.Viewer(title=f"f{frame} CellUniverse review", ndisplay=3)
    if real_tif is not None and real_tif.is_file():
        viewer.add_image(load_tif(real_tif), name=f"{frame}_real", colormap="gray", opacity=1.0)
    if synth_tif is not None and synth_tif.is_file():
        viewer.add_image(load_tif(synth_tif), name=f"{frame}_synth", colormap="cyan", opacity=0.22)

    rings, owners = make_all_rings(cells, segments=72, rings_per_axis=2, radius_scale=1.0)
    if rings:
        viewer.add_shapes(
            rings,
            shape_type="path",
            name="prediction ellipsoid outlines",
            edge_color=[colors.get(owner, (0.0, 0.9, 0.2, 0.70)) for owner in owners],
            edge_width=0.16,
            opacity=1.0,
        )

    pred_points = np.asarray([cell_center_zyx(cell) for cell in cells], dtype=float)
    if len(pred_points):
        viewer.add_points(
            pred_points,
            name="prediction centers",
            size=3.0,
            face_color=[colors.get(cell.name, (0.0, 0.9, 0.2, 0.70)) for cell in cells],
            border_color=[ERROR_COLOR if cell.name in extra_names else (0.05, 0.05, 0.05, 0.65) for cell in cells],
            border_width=0.15,
            out_of_slice_display=True,
        )

    gt_points = np.asarray([[row.z, row.y, row.x] for row in gt], dtype=float)
    if len(gt_points):
        viewer.add_points(
            gt_points,
            name="fixed GT centers",
            size=3.2,
            symbol="square",
            face_color=[ERROR_COLOR if row.label_id in missing_labels else GT_COLOR for row in gt],
            border_color=(0.0, 0.0, 0.0, 0.35),
            border_width=0.2,
            out_of_slice_display=True,
        )

    return viewer
