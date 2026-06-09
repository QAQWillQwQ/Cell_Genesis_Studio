from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np


def write_movie(frames: list[np.ndarray], output: Path, fps: float = 6.0) -> Path:
    if not frames:
        raise ValueError("Cannot write a movie with no frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        iio.imwrite(output, frames, duration=1000.0 / max(1e-6, fps), loop=0)
    else:
        iio.imwrite(output, frames, fps=fps)
    return output
