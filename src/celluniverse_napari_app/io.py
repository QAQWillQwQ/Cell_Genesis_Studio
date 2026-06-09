from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile

from .models import CellRecord, GtRecord


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _radius(row: dict[str, str], primary: str, legacy: str, default: float) -> float:
    value = _float(row, primary, float("nan"))
    if np.isfinite(value) and value > 0:
        return value
    value = _float(row, legacy, float("nan"))
    if np.isfinite(value) and value > 0:
        return value
    return default


def read_cells(cells_csv: Path, frame: int) -> list[CellRecord]:
    rows: list[CellRecord] = []
    expected = f"t{frame:03d}.tif"
    with cells_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("file") != expected:
                continue
            if row.get("isTrash", "0").strip().lower() in {"1", "true", "yes"}:
                continue
            a = _radius(row, "aRadius", "majorRadius", 10.0)
            b = _radius(row, "bRadius", "bRadius", a)
            c = _radius(row, "cRadius", "minorRadius", b)
            rows.append(
                CellRecord(
                    file=row.get("file", ""),
                    name=row.get("name", ""),
                    x=_float(row, "x"),
                    y=_float(row, "y"),
                    z=_float(row, "z"),
                    a=a,
                    b=b,
                    c=c,
                    theta_x=_float(row, "theta_x"),
                    theta_y=_float(row, "theta_y"),
                    theta_z=_float(row, "theta_z"),
                    raw=dict(row),
                )
            )
    return rows


def read_gt(gt_csv: Path, frame: int) -> list[GtRecord]:
    rows: list[GtRecord] = []
    with gt_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                row_frame = int(float(row["frame"]))
            except (KeyError, ValueError):
                continue
            if row_frame != frame:
                continue
            rows.append(
                GtRecord(
                    frame=row_frame,
                    label_id=row.get("label_id", ""),
                    parent_label=row.get("parent_label", ""),
                    x=_float(row, "x"),
                    y=_float(row, "y"),
                    z=_float(row, "z_interp", _float(row, "z")),
                    raw=dict(row),
                )
            )
    return rows


def resolve_frame_tif(run_dir: Path, frame: int, kind: str) -> Path | None:
    candidates = [
        run_dir / f"{frame}_{kind}.tif",
        run_dir / "tiff" / kind / f"{frame}.tif",
        run_dir / kind / f"{frame}.tif",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_tif(path: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(path))


def write_tif(path: Path, stack: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if stack.ndim == 4 and stack.shape[-1] in (3, 4):
        tifffile.imwrite(path, stack, photometric="rgb")
    else:
        tifffile.imwrite(path, stack)
