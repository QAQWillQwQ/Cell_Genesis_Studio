from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CellRecord:
    file: str
    name: str
    x: float
    y: float
    z: float
    a: float
    b: float
    c: float
    theta_x: float
    theta_y: float
    theta_z: float
    raw: dict[str, str]


@dataclass(frozen=True)
class GtRecord:
    frame: int
    label_id: str
    parent_label: str
    x: float
    y: float
    z: float
    raw: dict[str, str]


@dataclass(frozen=True)
class FrameInputs:
    frame: int
    cells_csv: Path
    gt_csv: Path | None = None
    real_tif: Path | None = None
    synth_tif: Path | None = None
    run_dir: Path | None = None


@dataclass(frozen=True)
class OverlayOptions:
    ring_segments: int = 96
    rings_per_axis: int = 2
    radius_scale: float = 1.0
    ring_width: int = 1
    center_radius: int = 2
    pred_opacity: float = 0.88
    gt_opacity: float = 1.0
    draw_pred_centers: bool = True
    draw_gt_centers: bool = True
    draw_rings: bool = True
    draw_base: bool = True
    base_max_intensity: int = 160
    gt_color: str = "#ffffff"
    normalize_low_percentile: float = 1.0
    normalize_high_percentile: float = 99.5
