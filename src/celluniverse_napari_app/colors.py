from __future__ import annotations

import colorsys

import numpy as np

from .matching import cell_xyz
from .models import CellRecord, GtRecord


RGBA = tuple[float, float, float, float]


SPLIT_COLORS: list[RGBA] = [
    (0.58, 0.16, 1.0, 0.92),  # purple
    (0.00, 0.78, 0.92, 0.92),  # blue green
    (0.00, 0.20, 0.80, 0.92),  # deep blue
]

ERROR_COLOR: RGBA = (1.0, 0.0, 0.0, 0.60)
GT_COLOR: RGBA = (0.0, 0.0, 0.0, 0.60)


def green_palette(color_count: int) -> list[RGBA]:
    values = np.linspace(0.40, 1.0, max(3, color_count))
    colors: list[RGBA] = []
    for value in values:
        red, green, blue = colorsys.hsv_to_rgb(0.36, 0.88, float(value))
        colors.append((red, green, blue, 0.70))
    return colors


def build_neighbor_palette(
    cells: list[CellRecord],
    color_count: int = 9,
    neighbor_distance: float = 42.0,
) -> dict[str, int]:
    color_count = max(3, color_count)
    names = [cell.name for cell in cells]
    positions = {cell.name: cell_xyz(cell) for cell in cells}
    neighbors: dict[str, list[tuple[str, float]]] = {name: [] for name in names}

    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            distance = float(np.linalg.norm(positions[left] - positions[right]))
            if distance <= neighbor_distance:
                weight = max(0.05, 1.0 - distance / max(1.0, neighbor_distance))
                neighbors[left].append((right, weight))
                neighbors[right].append((left, weight))

    assignment: dict[str, int] = {}
    color_usage = [0] * color_count
    remaining = set(names)

    def similarity(left: int, right: int) -> float:
        if color_count <= 1:
            return 1.0
        diff = abs(left - right) / float(color_count - 1)
        return (1.0 - diff) ** 2

    while remaining:
        name = max(
            remaining,
            key=lambda item: (
                sum(1 for other, _ in neighbors[item] if other in assignment),
                sum(weight for other, weight in neighbors[item] if other in assignment),
                sum(weight for _, weight in neighbors[item]),
                item,
            ),
        )
        target = (len(assignment) * 0.61803398875) % 1.0
        target_index = int(round(target * (color_count - 1)))

        def penalty(index: int) -> tuple[float, int, int]:
            neighbor_penalty = 0.0
            for other, weight in neighbors[name]:
                if other not in assignment:
                    continue
                other_index = assignment[other]
                neighbor_penalty += weight * (4.0 if index == other_index else similarity(index, other_index))
            return (neighbor_penalty + 0.06 * color_usage[index], abs(index - target_index), index)

        selected = min(range(color_count), key=penalty)
        assignment[name] = selected
        color_usage[selected] += 1
        remaining.remove(name)
    return assignment


def split_group_centroids(
    cells: list[CellRecord],
    gt: list[GtRecord],
    split_pred_groups: dict[str, set[str]],
    split_gt_groups: dict[str, set[str]],
) -> dict[str, np.ndarray]:
    pred_by_name = {cell.name: cell_xyz(cell) for cell in cells}
    gt_by_label = {
        row.label_id: np.asarray([row.x, row.y, row.z], dtype=float)
        for row in gt
    }
    centroids: dict[str, np.ndarray] = {}
    for group_name in sorted(set(split_pred_groups) | set(split_gt_groups)):
        points = [
            pred_by_name[name]
            for name in split_pred_groups.get(group_name, set())
            if name in pred_by_name
        ]
        if not points:
            points = [
                gt_by_label[label]
                for label in split_gt_groups.get(group_name, set())
                if label in gt_by_label
            ]
        if points:
            centroids[group_name] = np.mean(np.asarray(points), axis=0)
    return centroids


def assign_split_group_colors(
    group_order: list[str],
    centroids: dict[str, np.ndarray],
    neighbor_distance: float = 64.0,
) -> dict[str, RGBA]:
    if not group_order:
        return {}
    neighbors: dict[str, list[tuple[str, float]]] = {group: [] for group in group_order}
    for index, left in enumerate(group_order):
        if left not in centroids:
            continue
        for right in group_order[index + 1 :]:
            if right not in centroids:
                continue
            distance = float(np.linalg.norm(centroids[left] - centroids[right]))
            if distance <= neighbor_distance:
                weight = max(0.05, 1.0 - distance / max(1.0, neighbor_distance))
                neighbors[left].append((right, weight))
                neighbors[right].append((left, weight))

    assignment: dict[str, int] = {}
    usage = [0] * len(SPLIT_COLORS)
    remaining = set(group_order)
    while remaining:
        group = max(
            remaining,
            key=lambda item: (
                sum(1 for other, _ in neighbors[item] if other in assignment),
                sum(weight for _, weight in neighbors[item]),
                item,
            ),
        )

        def penalty(color_index: int) -> tuple[float, int, int]:
            near_penalty = 0.0
            for other, weight in neighbors[group]:
                if other in assignment and assignment[other] == color_index:
                    near_penalty += 12.0 * weight
            return (near_penalty + 0.08 * usage[color_index], color_index, usage[color_index])

        selected = min(range(len(SPLIT_COLORS)), key=penalty)
        assignment[group] = selected
        usage[selected] += 1
        remaining.remove(group)
    return {group: SPLIT_COLORS[assignment[group]] for group in group_order}


def frame_color_plan(
    cells: list[CellRecord],
    gt: list[GtRecord],
    frame: int,
    match_distance: float = 25.0,
    split_match_distance: float = 25.0,
    green_levels: int = 9,
    neighbor_distance: float = 42.0,
) -> tuple[dict[str, RGBA], set[str], dict[str, set[str]], set[str]]:
    from .matching import infer_gt_split_groups, match_cells_to_gt, split_pred_groups_from_gt

    _, extra_names, missing_labels = match_cells_to_gt(cells, gt, match_distance) if gt else ({}, set(), set())
    split_gt_groups = infer_gt_split_groups(gt, frame) if gt else {}
    split_pred_groups = split_pred_groups_from_gt(cells, gt, split_gt_groups, split_match_distance) if gt else {}
    split_name_to_group = {
        name: group_name
        for group_name, names in split_pred_groups.items()
        for name in names
    }
    group_order = sorted(split_gt_groups)
    split_colors = assign_split_group_colors(
        group_order,
        split_group_centroids(cells, gt, split_pred_groups, split_gt_groups),
        neighbor_distance * 1.6,
    )
    greens = green_palette(green_levels)
    green_assignment = build_neighbor_palette(cells, green_levels, neighbor_distance)

    colors: dict[str, RGBA] = {}
    for cell in cells:
        if cell.name in extra_names:
            colors[cell.name] = ERROR_COLOR
        elif cell.name in split_name_to_group:
            colors[cell.name] = split_colors.get(split_name_to_group[cell.name], SPLIT_COLORS[0])
        else:
            colors[cell.name] = greens[green_assignment.get(cell.name, 0) % len(greens)]
    return colors, missing_labels, split_pred_groups, extra_names
