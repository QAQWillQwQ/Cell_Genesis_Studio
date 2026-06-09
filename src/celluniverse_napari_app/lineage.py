from __future__ import annotations

import colorsys
import hashlib

from .models import CellRecord


def lineage_parent(name: str) -> str | None:
    if "_" in name:
        prefix, code = name.rsplit("_", 1)
        if len(code) > 1 and code[-1] in "01":
            return prefix + "_" + code[:-1]
    if len(name) > 1 and name[-1] in "01":
        return name[:-1]
    return None


def lineage_root(name: str) -> str:
    current = name
    while True:
        parent = lineage_parent(current)
        if parent is None:
            return current
        current = parent


def color_for_key(key: str, saturation: float = 0.78, value: float = 1.0) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    hue = int(digest[:8], 16) / 0xFFFFFFFF
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return "#{:02x}{:02x}{:02x}".format(int(red * 255), int(green * 255), int(blue * 255))


def lineage_color_map(cells: list[CellRecord]) -> dict[str, str]:
    return {cell.name: color_for_key(lineage_root(cell.name)) for cell in cells}
