from __future__ import annotations

import csv
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .colors import ERROR_COLOR, GT_COLOR, frame_color_plan
from .geometry import cell_center_zyx, make_all_rings
from .io import load_tif, read_cells, read_gt, resolve_frame_tif
from .matching import gt_xyz
from .models import CellRecord, GtRecord


@dataclass(frozen=True)
class FrameSource:
    frame: int
    cells_csv: Path
    run_dir: Path
    real_tif: Path
    synth_tif: Path | None


@dataclass(frozen=True)
class Camera:
    yaw: float
    pitch: float
    roll: float = 0.0


@dataclass(frozen=True)
class Projection:
    center_xyz: np.ndarray
    scale: float
    panel_size: tuple[int, int]


@dataclass(frozen=True)
class PreparedFrame:
    source: FrameSource
    projection: Projection
    points_xyz: np.ndarray
    values: np.ndarray
    cells: list[CellRecord]
    gt: list[GtRecord]
    colors: dict[str, tuple[float, float, float, float]]
    missing_labels: set[str]
    audit_names: set[str]


def read_manifest_sources(
    manifest: Path,
    first_frame: int,
    last_frame: int,
) -> dict[int, Path]:
    sources: dict[int, Path] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if first_frame <= frame <= last_frame:
                source = row.get("source_cells_csv", "").strip()
                if source:
                    sources[frame] = Path(source)
    return sources


def resolve_sources(
    manifest: Path,
    first_frame: int,
    last_frame: int,
) -> list[FrameSource]:
    cells_by_frame = read_manifest_sources(manifest, first_frame, last_frame)
    sources: list[FrameSource] = []
    for frame in range(first_frame, last_frame + 1):
        cells_csv = cells_by_frame.get(frame)
        if cells_csv is None:
            raise FileNotFoundError(f"No cells.csv source found for frame {frame} in {manifest}")
        run_dir = cells_csv.parent
        real_tif = resolve_frame_tif(run_dir, frame, "real")
        synth_tif = resolve_frame_tif(run_dir, frame, "synth")
        if real_tif is None:
            raise FileNotFoundError(f"No real tif found for frame {frame} under {run_dir}")
        sources.append(
            FrameSource(
                frame=frame,
                cells_csv=cells_csv,
                run_dir=run_dir,
                real_tif=real_tif,
                synth_tif=synth_tif,
            )
        )
    return sources


def normalize_stack(stack: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.7) -> np.ndarray:
    data = np.asarray(stack, dtype=np.float32)
    low, high = np.percentile(data, [low_pct, high_pct])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low = float(data.min())
        high = float(data.max())
    if high <= low:
        high = low + 1.0
    return np.clip((data - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def sample_volume_points(
    stack_u8: np.ndarray,
    max_points: int = 140_000,
    threshold_pct: float = 78.0,
) -> tuple[np.ndarray, np.ndarray]:
    threshold = float(np.percentile(stack_u8, threshold_pct))
    mask = stack_u8 >= max(1.0, threshold)
    indices = np.flatnonzero(mask)
    if len(indices) > max_points:
        selected = np.linspace(0, len(indices) - 1, max_points, dtype=np.int64)
        indices = indices[selected]
    z, y, x = np.unravel_index(indices, stack_u8.shape)
    points_xyz = np.column_stack([x, y, z]).astype(np.float32)
    values = stack_u8.ravel()[indices].astype(np.uint8)
    return points_xyz, values


def rotation_matrix(camera: Camera) -> np.ndarray:
    cy, sy = math.cos(camera.yaw), math.sin(camera.yaw)
    cp, sp = math.cos(camera.pitch), math.sin(camera.pitch)
    cr, sr = math.cos(camera.roll), math.sin(camera.roll)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rr = np.asarray([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rr @ rx @ rz


def project_points(points_xyz: np.ndarray, projection: Projection, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    rot = rotation_matrix(camera)
    centered = points_xyz.astype(np.float32) - projection.center_xyz.astype(np.float32)
    view = centered @ rot.T
    width, height = projection.panel_size
    xy = view[:, :2] * projection.scale
    px = np.round(xy[:, 0] + width * 0.5).astype(np.int32)
    py = np.round(xy[:, 1] + height * 0.5).astype(np.int32)
    return np.column_stack([px, py]), view[:, 2]


def render_volume_projection(
    points_xyz: np.ndarray,
    values: np.ndarray,
    projection: Projection,
    camera: Camera,
    color: tuple[int, int, int] = (230, 230, 230),
    blur_sigma: float = 0.8,
) -> np.ndarray:
    width, height = projection.panel_size
    image = np.zeros((height, width), dtype=np.uint8)
    pixel_xy, _ = project_points(points_xyz, projection, camera)
    x = pixel_xy[:, 0]
    y = pixel_xy[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    np.maximum.at(image, (y[inside], x[inside]), values[inside])
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    for channel, value in enumerate(color):
        bgr[:, :, channel] = np.clip(image.astype(np.float32) * (value / 255.0), 0, 255).astype(np.uint8)
    return bgr


def rgba_to_bgr(color: tuple[float, float, float, float]) -> tuple[int, int, int]:
    red, green, blue, _ = color
    return (int(blue * 255), int(green * 255), int(red * 255))


def blend_over(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def draw_rings_and_centers(
    image: np.ndarray,
    cells: list[CellRecord],
    colors: dict[str, tuple[float, float, float, float]],
    projection: Projection,
    camera: Camera,
    names: set[str] | None = None,
    ring_width: int = 1,
    center_radius: int = 2,
) -> None:
    selected = [cell for cell in cells if names is None or cell.name in names]
    rings, owners = make_all_rings(selected, segments=72, rings_per_axis=2, radius_scale=1.0)
    for ring, owner in zip(rings, owners):
        xyz = ring[:, [2, 1, 0]].astype(np.float32)
        pixel_xy, _ = project_points(xyz, projection, camera)
        points = pixel_xy.reshape((-1, 1, 2))
        color = rgba_to_bgr(colors.get(owner, (0.0, 0.9, 0.2, 0.70)))
        cv2.polylines(image, [points], False, color, ring_width, cv2.LINE_AA)

    for cell in selected:
        center = np.asarray([[cell.x, cell.y, cell.z]], dtype=np.float32)
        pixel_xy, _ = project_points(center, projection, camera)
        x, y = int(pixel_xy[0, 0]), int(pixel_xy[0, 1])
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            color = rgba_to_bgr(colors.get(cell.name, (0.0, 0.9, 0.2, 0.70)))
            cv2.circle(image, (x, y), center_radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), max(1, center_radius + 1), (18, 18, 18), 1, cv2.LINE_AA)


def draw_gt_points(
    image: np.ndarray,
    gt: list[GtRecord],
    missing_labels: set[str],
    projection: Projection,
    camera: Camera,
    radius: int = 2,
) -> None:
    if not gt:
        return
    points = np.asarray([gt_xyz(row) for row in gt], dtype=np.float32)
    pixel_xy, _ = project_points(points, projection, camera)
    for row, xy in zip(gt, pixel_xy):
        x, y = int(xy[0]), int(xy[1])
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        color = rgba_to_bgr(ERROR_COLOR if row.label_id in missing_labels else GT_COLOR)
        cv2.rectangle(image, (x - radius, y - radius), (x + radius, y + radius), color, -1, cv2.LINE_AA)


def draw_panel_title(panel: np.ndarray, title: str, frame: int, dataset_frame: int) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (48, 48, 48), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 34), (10, 10, 10), -1, cv2.LINE_AA)
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    label = f"frame {dataset_frame:03d}"
    size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    cv2.putText(panel, label, (panel.shape[1] - size[0] - 12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)


def panel_with_title(content: np.ndarray, title: str, output_frame: int, dataset_frame: int) -> np.ndarray:
    panel = content.copy()
    draw_panel_title(panel, title, output_frame, dataset_frame)
    return panel


def compose_four_panel(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (3, 3, 3)
    gap = 12
    margin = 16
    cell_w = (width - 2 * margin - gap) // 2
    cell_h = (height - 2 * margin - gap) // 2
    panels = [top_left, top_right, bottom_left, bottom_right]
    positions = [
        (margin, margin),
        (margin + cell_w + gap, margin),
        (margin, margin + cell_h + gap),
        (margin + cell_w + gap, margin + cell_h + gap),
    ]
    for panel, (x, y) in zip(panels, positions):
        resized = cv2.resize(panel, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[y : y + cell_h, x : x + cell_w] = resized
    return canvas


def prepare_frame(
    source: FrameSource,
    gt_csv: Path,
    panel_size: tuple[int, int],
    max_volume_points: int,
) -> PreparedFrame:
    stack = load_tif(source.real_tif)
    if stack.ndim == 4:
        stack = stack.max(axis=-1)
    stack_u8 = normalize_stack(stack)
    points_xyz, values = sample_volume_points(stack_u8, max_volume_points)
    center_xyz = np.asarray([stack.shape[2] * 0.5, stack.shape[1] * 0.5, stack.shape[0] * 0.5], dtype=np.float32)
    scale = min(panel_size[0] / (stack.shape[2] * 1.12), panel_size[1] / (stack.shape[1] * 1.12))
    projection = Projection(center_xyz=center_xyz, scale=scale, panel_size=panel_size)

    cells = read_cells(source.cells_csv, source.frame)
    gt = read_gt(gt_csv, source.frame)
    colors, missing_labels, split_groups, extra_names = frame_color_plan(cells, gt, source.frame)
    split_names = {name for names in split_groups.values() for name in names}
    audit_names = split_names | extra_names
    return PreparedFrame(
        source=source,
        projection=projection,
        points_xyz=points_xyz,
        values=values,
        cells=cells,
        gt=gt,
        colors=colors,
        missing_labels=missing_labels,
        audit_names=audit_names,
    )


def render_prepared_views(prepared: PreparedFrame, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    real = render_volume_projection(
        prepared.points_xyz,
        prepared.values,
        prepared.projection,
        camera,
        color=(228, 228, 228),
        blur_sigma=0.8,
    )
    annotated = real.copy()
    draw_rings_and_centers(
        annotated,
        prepared.cells,
        prepared.colors,
        prepared.projection,
        camera,
        ring_width=1,
        center_radius=2,
    )

    geometry = np.zeros_like(real)
    draw_rings_and_centers(
        geometry,
        prepared.cells,
        prepared.colors,
        prepared.projection,
        camera,
        ring_width=1,
        center_radius=2,
    )

    audit = (real.astype(np.float32) * 0.55).astype(np.uint8)
    draw_rings_and_centers(
        audit,
        prepared.cells,
        prepared.colors,
        prepared.projection,
        camera,
        names=prepared.audit_names if prepared.audit_names else None,
        ring_width=1,
        center_radius=2,
    )
    draw_gt_points(audit, prepared.gt, prepared.missing_labels, prepared.projection, camera, radius=2)
    return real, annotated, geometry, audit


def camera_for_index(index: int, total: int) -> Camera:
    if total <= 1:
        phase = 0.0
    else:
        phase = index / float(total - 1)
    yaw = math.radians(-18.0 + 54.0 * phase)
    pitch = math.radians(18.0 + 7.0 * math.sin(2.0 * math.pi * phase))
    return Camera(yaw=yaw, pitch=pitch)


def make_mp4(
    first_frame: int,
    last_frame: int,
    manifest: Path,
    gt_csv: Path,
    output: Path,
    fps: float = 8.0,
    subframes_per_frame: int = 6,
    width: int = 1920,
    height: int = 1080,
    max_volume_points: int = 120_000,
    keep_frames: bool = False,
) -> Path:
    sources = resolve_sources(manifest, first_frame, last_frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_size = ((width - 44) // 2, (height - 44) // 2)
    total_output_frames = len(sources) * subframes_per_frame
    start_time = time.time()
    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.png"):
        old_frame.unlink()

    output_index = 0
    for source in sources:
        frame_start = time.time()
        prepared = prepare_frame(source, gt_csv, panel_size, max_volume_points)
        prepare_elapsed = time.time() - frame_start
        for local in range(subframes_per_frame):
            camera = camera_for_index(output_index, max(1, total_output_frames))
            views = render_prepared_views(prepared, camera)
            panels = [
                panel_with_title(views[0], "Real rotating volume", output_index, source.frame),
                panel_with_title(views[1], "Real plus algorithm outlines", output_index, source.frame),
                panel_with_title(views[2], "Algorithm geometry only", output_index, source.frame),
                panel_with_title(views[3], "Split and GT audit", output_index, source.frame),
            ]
            canvas_bgr = compose_four_panel(*panels, width=width, height=height)
            frame_path = frame_dir / f"frame_{output_index:06d}.png"
            if not cv2.imwrite(str(frame_path), canvas_bgr):
                raise IOError(f"Failed to write frame image: {frame_path}")
            output_index += 1
        print(
            f"[MP4] frame {source.frame:03d} rendered "
            f"{subframes_per_frame} views in {time.time() - frame_start:.1f}s "
            f"(prepare {prepare_elapsed:.1f}s)"
        )

    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-movflags",
        "+faststart",
        "-crf",
        "18",
        "-preset",
        "medium",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed to encode MP4")

    if not keep_frames:
        for frame_path in frame_dir.glob("frame_*.png"):
            frame_path.unlink()
        try:
            frame_dir.rmdir()
        except OSError:
            pass
    print(f"[MP4] saved {output} in {time.time() - start_time:.1f}s")
    return output
