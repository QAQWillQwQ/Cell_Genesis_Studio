from __future__ import annotations

import csv
from pathlib import Path


def read_manifest(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["frame"])] = row
    return rows


def verified_frames(path: Path) -> list[int]:
    rows = read_manifest(path)
    allowed = {"VERIFIED_BY_FIXED_GT_25PX", "VERIFIED_WITH_GT_DELAY_NOTE"}
    return [frame for frame, row in sorted(rows.items()) if row.get("review_status") in allowed]
