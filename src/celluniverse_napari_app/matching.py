from __future__ import annotations

from collections import deque

import numpy as np

from .models import CellRecord, GtRecord


def cell_xyz(cell: CellRecord) -> np.ndarray:
    return np.asarray([cell.x, cell.y, cell.z], dtype=float)


def gt_xyz(row: GtRecord) -> np.ndarray:
    return np.asarray([row.x, row.y, row.z], dtype=float)


def match_cells_to_gt(
    cells: list[CellRecord],
    gt: list[GtRecord],
    threshold: float,
) -> tuple[dict[str, str], set[str], set[str]]:
    adjacency: list[list[int]] = []
    for cell in cells:
        point = cell_xyz(cell)
        adjacency.append(
            [
                gt_index
                for gt_index, gt_row in enumerate(gt)
                if float(np.linalg.norm(point - gt_xyz(gt_row))) <= threshold
            ]
        )

    pair_cell = [-1] * len(cells)
    pair_gt = [-1] * len(gt)
    dist = [0] * len(cells)

    def bfs() -> bool:
        queue: deque[int] = deque()
        found = False
        for index, partner in enumerate(pair_cell):
            if partner == -1:
                dist[index] = 0
                queue.append(index)
            else:
                dist[index] = -1
        while queue:
            index = queue.popleft()
            for gt_index in adjacency[index]:
                next_cell = pair_gt[gt_index]
                if next_cell == -1:
                    found = True
                elif dist[next_cell] == -1:
                    dist[next_cell] = dist[index] + 1
                    queue.append(next_cell)
        return found

    def dfs(index: int) -> bool:
        for gt_index in adjacency[index]:
            next_cell = pair_gt[gt_index]
            if next_cell == -1 or (dist[next_cell] == dist[index] + 1 and dfs(next_cell)):
                pair_cell[index] = gt_index
                pair_gt[gt_index] = index
                return True
        dist[index] = -1
        return False

    while bfs():
        for index, partner in enumerate(pair_cell):
            if partner == -1:
                dfs(index)

    cell_to_gt: dict[str, str] = {}
    extra_names: set[str] = set()
    missing_labels: set[str] = set()
    for cell_index, gt_index in enumerate(pair_cell):
        if gt_index == -1:
            extra_names.add(cells[cell_index].name)
        else:
            cell_to_gt[cells[cell_index].name] = gt[gt_index].label_id
    for gt_index, cell_index in enumerate(pair_gt):
        if cell_index == -1:
            missing_labels.add(gt[gt_index].label_id)
    return cell_to_gt, extra_names, missing_labels


def infer_gt_split_groups(gt: list[GtRecord], frame: int) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for row in gt:
        parent = row.parent_label.strip()
        if parent in {"", "0", "-1", "nan", "None"}:
            continue
        raw_start = row.raw.get("start_frame", "")
        try:
            start_frame = int(float(raw_start))
        except ValueError:
            continue
        if start_frame == frame:
            groups.setdefault(parent, set()).add(row.label_id)
    return groups


def split_pred_groups_from_gt(
    cells: list[CellRecord],
    gt: list[GtRecord],
    split_gt_groups: dict[str, set[str]],
    max_distance: float,
) -> dict[str, set[str]]:
    if not split_gt_groups:
        return {}
    label_to_group: dict[str, str] = {}
    for group_name, labels in split_gt_groups.items():
        for label in labels:
            label_to_group[label] = group_name

    result = {group_name: set() for group_name in split_gt_groups}
    used_names: set[str] = set()
    for gt_row in gt:
        group_name = label_to_group.get(gt_row.label_id)
        if group_name is None:
            continue
        gt_point = gt_xyz(gt_row)
        best_name = ""
        best_distance = float("inf")
        for cell in cells:
            if cell.name in used_names:
                continue
            distance = float(np.linalg.norm(cell_xyz(cell) - gt_point))
            if distance < best_distance:
                best_name = cell.name
                best_distance = distance
        if best_name and best_distance <= max_distance:
            result[group_name].add(best_name)
            used_names.add(best_name)
    return {group_name: names for group_name, names in result.items() if names}
