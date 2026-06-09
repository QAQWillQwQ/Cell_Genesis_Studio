from __future__ import annotations

import argparse
from pathlib import Path

from .models import OverlayOptions
from .movie_pipeline import make_mp4
from .overlays import bake_overlay_tif
from .render import parse_shape, render_synth_tif
from .viewer import open_review_viewer


DEFAULT_FIXED_GT = Path(
    "/Users/wangyiding/CellUniverse/C++/config/embryo/ground_truth/embryo_FixedGroundTruth.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CellUniverse Napari review tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("app", help="Open the local PyQt manual initial and video export app")

    review = sub.add_parser("review", help="Open one frame in Napari")
    review.add_argument("--frame", type=int, required=True)
    review.add_argument("--cells", type=Path, required=True)
    review.add_argument("--gt", type=Path, default=DEFAULT_FIXED_GT)
    review.add_argument("--run", type=Path)
    review.add_argument("--real-tif", type=Path)
    review.add_argument("--synth-tif", type=Path)

    bake = sub.add_parser("bake-overlay", help="Write a separate RGB overlay tif")
    bake.add_argument("--frame", type=int, required=True)
    bake.add_argument("--base-tif", type=Path, required=True)
    bake.add_argument("--cells", type=Path, required=True)
    bake.add_argument("--gt", type=Path, default=DEFAULT_FIXED_GT)
    bake.add_argument("--output", type=Path, required=True)
    bake.add_argument("--rings-per-axis", type=int, default=2)
    bake.add_argument("--ring-width", type=int, default=1)
    bake.add_argument("--center-radius", type=int, default=2)
    bake.add_argument("--base-max", type=int, default=160)
    bake.add_argument("--pred-opacity", type=float, default=0.95)
    bake.add_argument("--gt-opacity", type=float, default=0.85)
    bake.add_argument("--gt-color", default="#ffffff")
    bake.add_argument("--no-rings", action="store_true")
    bake.add_argument("--no-pred-centers", action="store_true")
    bake.add_argument("--no-gt-centers", action="store_true")
    bake.add_argument("--no-base", action="store_true")

    render = sub.add_parser("render-synth", help="Render a synth tif from cells.csv ellipsoids")
    render.add_argument("--frame", type=int, required=True)
    render.add_argument("--cells", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--shape", default="239,512,708", help="Output stack shape as z,y,x")
    render.add_argument("--value", type=int, default=250)

    movie = sub.add_parser("make-mp4", help="Render an integrated four panel rotating MP4")
    movie.add_argument("--first-frame", type=int, required=True)
    movie.add_argument("--last-frame", type=int, required=True)
    movie.add_argument(
        "--manifest",
        type=Path,
        default=Path("/Volumes/T9/🦠Cell Universe/🟣Output/Visualization/_audit_20260602/review_manifest_fixed_gt_0_174_20260602.csv"),
    )
    movie.add_argument("--gt", type=Path, default=DEFAULT_FIXED_GT)
    movie.add_argument("--output", type=Path, required=True)
    movie.add_argument("--fps", type=float, default=8.0)
    movie.add_argument("--subframes-per-frame", type=int, default=6)
    movie.add_argument("--width", type=int, default=1920)
    movie.add_argument("--height", type=int, default=1080)
    movie.add_argument("--max-volume-points", type=int, default=120000)
    movie.add_argument("--keep-frames", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "app":
        from .local_app import run_app

        return run_app()

    if args.command == "review":
        open_review_viewer(
            frame=args.frame,
            cells_csv=args.cells,
            gt_csv=args.gt,
            run_dir=args.run,
            real_tif=args.real_tif,
            synth_tif=args.synth_tif,
        )
        import napari

        napari.run()
        return 0

    if args.command == "bake-overlay":
        options = OverlayOptions(
            rings_per_axis=args.rings_per_axis,
            ring_width=args.ring_width,
            center_radius=args.center_radius,
            base_max_intensity=args.base_max,
            pred_opacity=args.pred_opacity,
            gt_opacity=args.gt_opacity,
            gt_color=args.gt_color,
            draw_rings=not args.no_rings,
            draw_pred_centers=not args.no_pred_centers,
            draw_gt_centers=not args.no_gt_centers,
            draw_base=not args.no_base,
        )
        output = bake_overlay_tif(args.frame, args.base_tif, args.cells, args.output, args.gt, options)
        print(output)
        return 0

    if args.command == "render-synth":
        output = render_synth_tif(
            frame=args.frame,
            cells_csv=args.cells,
            output_tif=args.output,
            shape_zyx=parse_shape(args.shape),
            value=args.value,
        )
        print(output)
        return 0

    if args.command == "make-mp4":
        output = make_mp4(
            first_frame=args.first_frame,
            last_frame=args.last_frame,
            manifest=args.manifest,
            gt_csv=args.gt,
            output=args.output,
            fps=args.fps,
            subframes_per_frame=args.subframes_per_frame,
            width=args.width,
            height=args.height,
            max_volume_points=args.max_volume_points,
            keep_frames=args.keep_frames,
        )
        print(output)
        return 0

    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
