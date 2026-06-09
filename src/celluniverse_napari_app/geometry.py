from __future__ import annotations

import math

import numpy as np

from .models import CellRecord


def rotation_matrix(cell: CellRecord) -> np.ndarray:
    tx, ty, tz = cell.theta_x, cell.theta_y, cell.theta_z
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    cz, sz = math.cos(tz), math.sin(tz)
    return np.asarray(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=float,
    )


def xyz_to_zyx(points_xyz: np.ndarray) -> np.ndarray:
    return points_xyz[:, [2, 1, 0]]


def cell_center_zyx(cell: CellRecord) -> np.ndarray:
    return np.asarray([cell.z, cell.y, cell.x], dtype=float)


def make_cell_rings(
    cell: CellRecord,
    segments: int = 96,
    rings_per_axis: int = 2,
    radius_scale: float = 1.0,
) -> list[np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, max(12, segments), endpoint=True)
    if rings_per_axis <= 1:
        offsets = [0.0]
    elif rings_per_axis == 2:
        offsets = [-0.38, 0.38]
    else:
        offsets = np.linspace(-0.62, 0.62, rings_per_axis)

    center_xyz = np.asarray([cell.x, cell.y, cell.z], dtype=float)
    rx = max(1.0, cell.a * radius_scale)
    ry = max(1.0, cell.b * radius_scale)
    rz = max(1.0, cell.c * radius_scale)
    rot = rotation_matrix(cell)
    rings: list[np.ndarray] = []

    for offset in offsets:
        scale = float(np.sqrt(max(0.0, 1.0 - offset * offset)))
        local_xy = np.column_stack(
            [rx * scale * np.cos(angles), ry * scale * np.sin(angles), np.full_like(angles, rz * offset)]
        )
        local_xz = np.column_stack(
            [rx * scale * np.cos(angles), np.full_like(angles, ry * offset), rz * scale * np.sin(angles)]
        )
        local_yz = np.column_stack(
            [np.full_like(angles, rx * offset), ry * scale * np.cos(angles), rz * scale * np.sin(angles)]
        )
        rings.append(xyz_to_zyx(center_xyz + local_xy @ rot.T))
        rings.append(xyz_to_zyx(center_xyz + local_xz @ rot.T))
        rings.append(xyz_to_zyx(center_xyz + local_yz @ rot.T))
    return rings


def make_all_rings(
    cells: list[CellRecord],
    segments: int = 96,
    rings_per_axis: int = 2,
    radius_scale: float = 1.0,
) -> tuple[list[np.ndarray], list[str]]:
    rings: list[np.ndarray] = []
    owners: list[str] = []
    for cell in cells:
        for ring in make_cell_rings(cell, segments, rings_per_axis, radius_scale):
            rings.append(ring)
            owners.append(cell.name)
    return rings, owners
