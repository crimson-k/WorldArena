"""Command-line entry point for the eight retained metrics."""

from __future__ import annotations

import argparse
import sys

from .aggregate import aggregate
from .constants import SUPPORTED_METRICS, normalize_metrics
from .manifest import prepare
from .runner import run_evaluate, run_jepa


def _metrics(value: str) -> list[str]:
    parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
    try:
        return normalize_metrics(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare PNG/JPEG frames")
    prepare_parser.add_argument("--summary", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--metrics", type=_metrics, default=list(SUPPORTED_METRICS))

    evaluate_parser = subparsers.add_parser("evaluate", help="Run non-JEPA metrics")
    evaluate_parser.add_argument("--manifest", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--metrics", type=_metrics, default=list(SUPPORTED_METRICS))

    jepa_parser = subparsers.add_parser("jepa", help="Run JEPA on two MP4 directories")
    jepa_parser.add_argument("--real-dir", required=True)
    jepa_parser.add_argument("--gen-dir", required=True)
    jepa_parser.add_argument("--output-dir", required=True)
    jepa_parser.add_argument("--config", required=True)
    jepa_parser.add_argument("--jepa-python", default=sys.executable)

    aggregate_parser = subparsers.add_parser("aggregate", help="Build result JSON/CSV")
    aggregate_parser.add_argument("--manifest", required=True)
    aggregate_parser.add_argument("--output-dir", required=True)
    aggregate_parser.add_argument("--metrics", type=_metrics, default=list(SUPPORTED_METRICS))

    all_parser = subparsers.add_parser("all", help="Prepare, evaluate, JEPA, aggregate")
    all_parser.add_argument("--summary", required=True)
    all_parser.add_argument("--output-dir", required=True)
    all_parser.add_argument("--config", required=True)
    all_parser.add_argument("--metrics", type=_metrics, default=list(SUPPORTED_METRICS))
    all_parser.add_argument("--jepa-real-dir")
    all_parser.add_argument("--jepa-gen-dir")
    all_parser.add_argument("--jepa-python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.summary, args.output_dir, args.metrics))
        return 0
    if args.command == "evaluate":
        metrics = [metric for metric in args.metrics if metric != "jepa_similarity"]
        if metrics:
            run_evaluate(args.manifest, args.output_dir, metrics, args.config)
        return 0
    if args.command == "jepa":
        run_jepa(
            args.real_dir,
            args.gen_dir,
            args.output_dir,
            args.config,
            args.jepa_python,
        )
        return 0
    if args.command == "aggregate":
        print(aggregate(args.manifest, args.output_dir, args.metrics))
        return 0
    if args.command == "all":
        manifest = prepare(args.summary, args.output_dir, args.metrics)
        metrics = [metric for metric in args.metrics if metric != "jepa_similarity"]
        if metrics:
            run_evaluate(manifest, args.output_dir, metrics, args.config)
        if "jepa_similarity" in args.metrics:
            if not args.jepa_real_dir or not args.jepa_gen_dir:
                raise ValueError(
                    "--jepa-real-dir and --jepa-gen-dir are required when JEPA is selected"
                )
            run_jepa(
                args.jepa_real_dir,
                args.jepa_gen_dir,
                args.output_dir,
                args.config,
                args.jepa_python,
            )
        print(aggregate(manifest, args.output_dir, args.metrics))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
