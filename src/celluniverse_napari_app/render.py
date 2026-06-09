from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import rotation_matrix
from .io import read_cells, write_tif
from .models import CellRecord


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("shape must be z,y,x")
    return parts[0], parts[1], parts[2]


def render_synth_stack(
    cells: list[CellRecord],
    shape_zyx: tuple[int, int, int] = (239, 512, 708),
    value: int = 250,
) -> np.ndarray:
    stack = np.zeros(shape_zyx, dtype=np.uint8)
    z_count, y_count, x_count = shape_zyx
    value = int(max(0, min(255, value)))

    for cell in cells:
        radius = max(cell.a, cell.b, cell.c) + 2.0
        x0 = max(0, int(np.floor(cell.x - radius)))
        x1 = min(x_count - 1, int(np.ceil(cell.x + radius)))
        y0 = max(0, int(np.floor(cell.y - radius)))
        y1 = min(y_count - 1, int(np.ceil(cell.y + radius)))
        z0 = max(0, int(np.floor(cell.z - radius)))
        z1 = min(z_count - 1, int(np.ceil(cell.z + radius)))
        if x1 < x0 or y1 < y0 or z1 < z0:
            continue

        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1 + 1, dtype=np.float32),
            np.arange(y0, y1 + 1, dtype=np.float32),
            np.arange(x0, x1 + 1, dtype=np.float32),
            indexing="ij",
        )
        dx = xx - cell.x
        dy = yy - cell.y
        dz = zz - cell.z
        rot = rotation_matrix(cell)
        local_x = dx * rot[0, 0] + dy * rot[1, 0] + dz * rot[2, 0]
        local_y = dx * rot[0, 1] + dy * rot[1, 1] + dz * rot[2, 1]
        local_z = dx * rot[0, 2] + dy * rot[1, 2] + dz * rot[2, 2]
        inside = (
            (local_x / max(cell.a, 1.0)) ** 2
            + (local_y / max(cell.b, 1.0)) ** 2
            + (local_z / max(cell.c, 1.0)) ** 2
        ) <= 1.0
        view = stack[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
        view[inside] = np.maximum(view[inside], value)
    return stack


def render_synth_tif(
    frame: int,
    cells_csv: Path,
    output_tif: Path,
    shape_zyx: tuple[int, int, int] = (239, 512, 708),
    value: int = 250,
) -> Path:
    cells = read_cells(cells_csv, frame)
    stack = render_synth_stack(cells, shape_zyx=shape_zyx, value=value)
    write_tif(output_tif, stack)
    return output_tif
