"""Command-line entry point for the Stage-1 folder pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment scanner captures and phone photographs recursively.")
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument(
        "--review", choices=("none", "flagged", "all"), default="flagged",
        help="manual editor policy; default opens only structurally flagged masks")
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--pdf-dpi", type=int, default=300)
    parser.add_argument("--reprocess", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_pipeline(
        args.input_folder,
        args.output_folder,
        review=args.review,
        cpu_threads=args.cpu_threads,
        pdf_dpi=args.pdf_dpi,
        reprocess=args.reprocess,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

