"""Command-line entry point for the eight retained metrics."""

from __future__ import annotations

import argparse
import os
import sys

from .aggregate import aggregate
from .constants import SUPPORTED_METRICS, normalize_metrics
from .manifest import prepare
from .runner import run_distributed_evaluate, run_evaluate, run_jepa


def _metrics(value: str) -> list[str]:
    parsed = [item.strip().lower() for item in value.split(",") if item.strip()]
    try:
        return normalize_metrics(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _gpus(value: str) -> list[int]:
    try:
        gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must be comma-separated integers") from exc
    if not gpu_ids or any(gpu < 0 for gpu in gpu_ids):
        raise argparse.ArgumentTypeError("--gpus requires one or more non-negative IDs")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("--gpus contains duplicate IDs")
    return gpu_ids


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


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
    evaluate_parser.add_argument(
        "--gpus",
        type=_gpus,
        help="GPU IDs used to shard every selected non-JEPA metric, e.g. 0,1,2,3",
    )
    evaluate_parser.add_argument(
        "--processes-per-gpu",
        type=_positive_int,
        default=1,
        help="Worker processes launched on each selected GPU (default: 1)",
    )

    jepa_parser = subparsers.add_parser(
        "jepa", help="Run JEPA on two MP4 directories or stacked videos"
    )
    jepa_parser.add_argument("--real-dir")
    jepa_parser.add_argument("--gen-dir")
    jepa_parser.add_argument("--stacked-summary")
    jepa_parser.add_argument("--output-dir", required=True)
    jepa_parser.add_argument("--config", required=True)
    jepa_parser.add_argument("--jepa-python", default=sys.executable)
    jepa_parser.add_argument("--gpu", type=int, help="Single GPU ID used by JEPA")

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
    all_parser.add_argument("--jepa-stacked-summary")
    all_parser.add_argument("--jepa-python", default=sys.executable)
    all_parser.add_argument(
        "--gpus",
        type=_gpus,
        help="GPU IDs used to shard all non-JEPA metrics; JEPA uses the first ID",
    )
    all_parser.add_argument(
        "--processes-per-gpu",
        type=_positive_int,
        default=1,
        help="Workers per selected GPU for non-JEPA metrics (default: 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        print(prepare(args.summary, args.output_dir, args.metrics))
        return 0
    if args.command == "evaluate":
        if not args.gpus and args.processes_per_gpu != 1:
            raise ValueError("--processes-per-gpu requires --gpus")
        metrics = [metric for metric in args.metrics if metric != "jepa_similarity"]
        if metrics:
            if args.gpus:
                run_distributed_evaluate(
                    args.manifest,
                    args.output_dir,
                    metrics,
                    args.config,
                    args.gpus,
                    args.processes_per_gpu,
                )
            else:
                run_evaluate(args.manifest, args.output_dir, metrics, args.config)
        return 0
    if args.command == "jepa":
        run_jepa(
            args.real_dir,
            args.gen_dir,
            args.output_dir,
            args.config,
            args.jepa_python,
            args.gpu,
            args.stacked_summary,
        )
        return 0
    if args.command == "aggregate":
        print(aggregate(args.manifest, args.output_dir, args.metrics))
        return 0
    if args.command == "all":
        if not args.gpus and args.processes_per_gpu != 1:
            raise ValueError("--processes-per-gpu requires --gpus")
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError(
                "Do not start the 'all' command with torchrun; use 'all --gpus ...' "
                "so prepare and aggregate remain single-process"
            )
        manifest = prepare(args.summary, args.output_dir, args.metrics)
        metrics = [metric for metric in args.metrics if metric != "jepa_similarity"]
        if metrics:
            if args.gpus:
                run_distributed_evaluate(
                    manifest,
                    args.output_dir,
                    metrics,
                    args.config,
                    args.gpus,
                    args.processes_per_gpu,
                )
            else:
                run_evaluate(manifest, args.output_dir, metrics, args.config)
        if "jepa_similarity" in args.metrics:
            has_directories = bool(args.jepa_real_dir and args.jepa_gen_dir)
            if not args.jepa_stacked_summary and not has_directories:
                raise ValueError(
                    "JEPA requires --jepa-stacked-summary or both "
                    "--jepa-real-dir and --jepa-gen-dir"
                )
            run_jepa(
                None if args.jepa_stacked_summary else args.jepa_real_dir,
                None if args.jepa_stacked_summary else args.jepa_gen_dir,
                args.output_dir,
                args.config,
                args.jepa_python,
                args.gpus[0] if args.gpus else None,
                args.jepa_stacked_summary,
            )
        print(aggregate(manifest, args.output_dir, args.metrics))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
