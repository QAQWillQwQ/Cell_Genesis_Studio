from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import cell_center_zyx, make_all_rings
from .io import load_tif, read_cells, read_gt, write_tif
from .models import CellRecord, GtRecord, OverlayOptions
from .colors import frame_color_plan


def hex_to_rgb(color: str) -> np.ndarray:
    color = color.lstrip("#")
    return np.asarray([int(color[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def rgba_to_rgb(color: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(color[:3], dtype=np.float32) * 255.0


def normalize_gray(stack: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    data = np.asarray(stack, dtype=np.float32)
    low, high = np.percentile(data, [low_pct, high_pct])
    if high <= low:
        high = float(data.max()) if data.size else 1.0
        low = float(data.min()) if data.size else 0.0
    scaled = (data - low) / max(1e-6, high - low)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def gray_to_rgb(stack: np.ndarray, options: OverlayOptions) -> np.ndarray:
    if not options.draw_base:
        return np.zeros(stack.shape + (3,), dtype=np.uint8)
    gray = normalize_gray(stack, options.normalize_low_percentile, options.normalize_high_percentile)
    base_max = max(0, min(255, int(options.base_max_intensity)))
    if base_max < 255:
        gray = np.clip(gray.astype(np.float32) * (base_max / 255.0), 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def blend_voxel(rgb: np.ndarray, z: int, y: int, x: int, color: np.ndarray, opacity: float) -> None:
    if z < 0 or y < 0 or x < 0 or z >= rgb.shape[0] or y >= rgb.shape[1] or x >= rgb.shape[2]:
        return
    base = rgb[z, y, x].astype(np.float32)
    rgb[z, y, x] = np.clip(base * (1.0 - opacity) + color * opacity, 0, 255).astype(np.uint8)


def draw_ball(rgb: np.ndarray, center_zyx: np.ndarray, color: np.ndarray, radius: int, opacity: float) -> None:
    z0, y0, x0 = [int(round(v)) for v in center_zyx]
    r = max(0, int(radius))
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dz * dz + dy * dy + dx * dx <= r * r:
                    blend_voxel(rgb, z0 + dz, y0 + dy, x0 + dx, color, opacity)


def draw_polyline(rgb: np.ndarray, points_zyx: np.ndarray, color: np.ndarray, width: int, opacity: float) -> None:
    if len(points_zyx) < 2:
        return
    width = max(0, int(width))
    for left, right in zip(points_zyx[:-1], points_zyx[1:]):
        distance = float(np.linalg.norm(right - left))
        steps = max(1, int(np.ceil(distance * 1.8)))
        for step in range(steps + 1):
            point = left * (1.0 - step / steps) + right * (step / steps)
            draw_ball(rgb, point, color, width, opacity)


def gt_center_zyx(row: GtRecord) -> np.ndarray:
    return np.asarray([row.z, row.y, row.x], dtype=float)


def bake_overlay_stack(
    base_stack: np.ndarray,
    cells: list[CellRecord],
    gt: list[GtRecord] | None = None,
    options: OverlayOptions | None = None,
    color_map: dict[str, tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    options = options or OverlayOptions()
    rgb = gray_to_rgb(base_stack, options)
    colors = color_map or {cell.name: (0.0, 0.9, 0.2, 0.70) for cell in cells}

    if options.draw_rings:
        rings, owners = make_all_rings(cells, options.ring_segments, options.rings_per_axis, options.radius_scale)
        for ring, owner in zip(rings, owners):
            draw_polyline(
                rgb,
                ring,
                rgba_to_rgb(colors.get(owner, (0.0, 0.9, 0.2, 0.70))),
                options.ring_width,
                options.pred_opacity,
            )

    if options.draw_pred_centers:
        for cell in cells:
            draw_ball(
                rgb,
                cell_center_zyx(cell),
                rgba_to_rgb(colors.get(cell.name, (0.0, 0.9, 0.2, 0.70))),
                options.center_radius,
                options.pred_opacity,
            )

    if gt and options.draw_gt_centers:
        for row in gt:
            draw_ball(rgb, gt_center_zyx(row), hex_to_rgb(options.gt_color), options.center_radius + 1, options.gt_opacity)
    return rgb


def bake_overlay_tif(
    frame: int,
    base_tif: Path,
    cells_csv: Path,
    output_tif: Path,
    gt_csv: Path | None = None,
    options: OverlayOptions | None = None,
) -> Path:
    base_stack = load_tif(base_tif)
    cells = read_cells(cells_csv, frame)
    gt = read_gt(gt_csv, frame) if gt_csv is not None else None
    color_map = frame_color_plan(cells, gt or [], frame)[0]
    overlay = bake_overlay_stack(base_stack, cells, gt, options, color_map)
    write_tif(output_tif, overlay)
    return output_tif
