"""Build a deterministic evaluation summary for vertically stacked task videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_summary(input_root: Path) -> list[dict[str, str]]:
    videos = sorted(input_root.glob("*/*.mp4"))
    if not videos:
        raise FileNotFoundError(f"No task MP4 files found under: {input_root}")

    rows = []
    seen = set()
    for video in videos:
        task = video.parent.name
        sample_id = f"{task}__{video.stem}"
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID: {sample_id}")
        seen.add(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "stacked_video": str(video.resolve()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Fail unless exactly this many MP4 files are found",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Keep only the first N sorted samples (useful for smoke tests)",
    )
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    rows = build_summary(input_root)
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} videos, found {len(rows)} in {input_root}"
        )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Wrote {len(rows)} samples to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
