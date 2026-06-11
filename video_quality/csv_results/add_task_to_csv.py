import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


EPISODE_PATTERN = re.compile(r"episode[_-]?0*(\d+)", re.IGNORECASE)


def extract_episode_index(video_id: str) -> int | None:
    match = EPISODE_PATTERN.search(str(video_id))
    if match is None:
        return None
    return int(match.group(1))


def load_episode_tasks(jsonl_path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if "episode_index" not in item:
                raise KeyError(f"JSONL line {line_no} is missing 'episode_index'.")
            if "task" not in item:
                raise KeyError(f"JSONL line {line_no} is missing 'task'.")

            episode_index = int(item["episode_index"])
            if episode_index in tasks:
                raise ValueError(f"Duplicate episode_index in JSONL: {episode_index}")
            tasks[episode_index] = str(item["task"])

    return tasks


def choose_video_id(rows: list[dict[str, str]], episode_index: int) -> str:
    canonical = f"episode{episode_index}"
    for row in rows:
        if row.get("Video_ID") == canonical:
            return canonical
    return rows[0].get("Video_ID", canonical)


def merge_episode_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    episode_index: int,
) -> dict[str, str]:
    merged = {fieldname: "" for fieldname in fieldnames}
    conflicts: list[str] = []

    for row in rows:
        for fieldname in fieldnames:
            value = row.get(fieldname, "")
            if value == "":
                continue
            if merged[fieldname] in ("", value):
                merged[fieldname] = value
            elif fieldname not in ("Video_ID",):
                conflicts.append(fieldname)

    if conflicts:
        conflict_text = ", ".join(sorted(set(conflicts)))
        raise ValueError(
            f"Conflicting non-empty values for episode {episode_index}: {conflict_text}"
        )

    merged["Video_ID"] = choose_video_id(rows, episode_index)
    return merged


def merge_add_task_and_sort(csv_path: Path, jsonl_path: Path, output_path: Path) -> Path:
    tasks = load_episode_tasks(jsonl_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header.")
        if "Video_ID" not in reader.fieldnames:
            raise KeyError("CSV file is missing required column: Video_ID")

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    grouped_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    skipped_video_ids: list[str] = []

    for row in rows:
        episode_index = extract_episode_index(row.get("Video_ID", ""))
        if episode_index is None:
            skipped_video_ids.append(row.get("Video_ID", ""))
            continue
        grouped_rows[episode_index].append(row)

    missing_tasks = sorted(set(grouped_rows) - set(tasks))
    if missing_tasks:
        preview = ", ".join(map(str, missing_tasks[:10]))
        raise KeyError(f"Missing task labels for episode_index: {preview}")

    missing_csv_rows = sorted(set(tasks) - set(grouped_rows))
    if missing_csv_rows:
        preview = ", ".join(map(str, missing_csv_rows[:10]))
        raise KeyError(f"Missing CSV rows for episode_index: {preview}")

    episode_indices = sorted(grouped_rows)
    task_indices = sorted(tasks)

    output_fieldnames = list(fieldnames)
    if "task" not in output_fieldnames:
        video_id_pos = output_fieldnames.index("Video_ID")
        output_fieldnames.insert(video_id_pos + 1, "task")

    merged_rows = []
    for episode_index in episode_indices:
        merged = merge_episode_rows(grouped_rows[episode_index], fieldnames, episode_index)
        merged["task"] = tasks[episode_index]
        merged_rows.append((tasks[episode_index], episode_index, merged))

    merged_rows.sort(key=lambda item: (item[0].casefold(), item[1]))
    sorted_rows = [row for _, _, row in merged_rows]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"Input rows: {len(rows)}")
    print(f"Merged episode rows: {len(sorted_rows)}")
    print(f"CSV episode_index range: {episode_indices[0]}..{episode_indices[-1]}")
    print(f"JSONL episode_index range: {task_indices[0]}..{task_indices[-1]}")
    if skipped_video_ids:
        print(f"Skipped non-episode rows: {len(skipped_video_ids)}")
    print(f"Saved: {output_path}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge split per-episode metric rows, add task labels from JSONL, "
            "then sort by task and numeric episode index."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to the aggregated CSV file.")
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to the JSONL file containing episode_index and task.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path. Defaults to '<input>_merged_with_task_sorted.csv'.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.csv_path.with_name(
            f"{args.csv_path.stem}_merged_with_task_sorted{args.csv_path.suffix}"
        )

    merge_add_task_and_sort(args.csv_path, args.jsonl_path, output_path)


if __name__ == "__main__":
    main()
