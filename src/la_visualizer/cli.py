from __future__ import annotations

import argparse
from pathlib import Path

from la_visualizer.plots import (
    plot_homogeneous_translation,
    plot_letter_n,
    plot_rotation,
    save_or_show,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="la-visualizer",
        description="Generate linear algebra visuals from the lavisuals notebook.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rotation = subparsers.add_parser("rotation", help="visualize 2D vector rotation")
    rotation.add_argument("--vector", nargs=2, type=float, default=(3.0, 1.0), metavar=("X", "Y"))
    rotation.add_argument("--angles", nargs="+", type=float, default=[45.0])
    rotation.add_argument("--output", type=Path, default=Path("outputs/rotation.png"))
    rotation.add_argument("--show", action="store_true")

    letter_n = subparsers.add_parser("letter-n", help="visualize letter N transformations")
    letter_n.add_argument(
        "--transform",
        choices=["original", "shear-fixed-x", "shear-fixed-y", "rotate", "shear-scaled-y"],
        default="original",
    )
    letter_n.add_argument("--angle", type=float, default=30.0)
    letter_n.add_argument("--output", type=Path, default=Path("outputs/letter_n.png"))
    letter_n.add_argument("--show", action="store_true")

    homogeneous = subparsers.add_parser(
        "homogeneous",
        help="visualize homogeneous-coordinate translation",
    )
    homogeneous.add_argument("--translate", nargs=2, type=float, default=(7.0, 4.0), metavar=("DX", "DY"))
    homogeneous.add_argument("--output", type=Path, default=Path("outputs/homogeneous.png"))
    homogeneous.add_argument("--show", action="store_true")

    all_parser = subparsers.add_parser("all", help="generate all notebook-inspired visuals")
    all_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    all_parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "rotation":
        fig = plot_rotation(tuple(args.vector), args.angles)
        save_or_show(fig, args.output, args.show)
        return 0

    if args.command == "letter-n":
        fig = plot_letter_n(args.transform, args.angle)
        save_or_show(fig, args.output, args.show)
        return 0

    if args.command == "homogeneous":
        fig = plot_homogeneous_translation(tuple(args.translate))
        save_or_show(fig, args.output, args.show)
        return 0

    output_dir = args.output_dir
    figures = {
        "rotation.png": plot_rotation((3.0, 1.0), [45.0]),
        "letter_n_original.png": plot_letter_n("original"),
        "letter_n_shear_fixed_x.png": plot_letter_n("shear-fixed-x"),
        "letter_n_shear_fixed_y.png": plot_letter_n("shear-fixed-y"),
        "letter_n_rotated.png": plot_letter_n("rotate"),
        "letter_n_shear_scaled_y.png": plot_letter_n("shear-scaled-y"),
        "homogeneous.png": plot_homogeneous_translation(),
    }
    for filename, fig in figures.items():
        save_or_show(fig, output_dir / filename, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
