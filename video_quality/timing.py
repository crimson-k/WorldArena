"""Rank-zero timing logs for long-running evaluation stages."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

from .manifest import atomic_write_json


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class MetricTimingLogger:
    """Write a readable log and an atomically updated structured timing file."""

    def __init__(
        self,
        output_dir: str | Path,
        manifest: str | Path,
        metrics: list[str],
        world_size: int,
    ) -> None:
        log_root = Path(output_dir).expanduser().resolve() / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        self.text_path = log_root / "metric_timings.log"
        self.json_path = log_root / "metric_timings.json"
        self.run_started = time.perf_counter()
        self.active_stage: tuple[list[str], str, float] | None = None
        self.payload = {
            "status": "running",
            "manifest": str(Path(manifest).expanduser().resolve()),
            "metrics": metrics,
            "world_size": int(world_size),
            "run_started_at": _now(),
            "run_finished_at": None,
            "total_elapsed_seconds": None,
            "total_elapsed": None,
            "stages": [],
        }
        self.text_path.write_text("", encoding="utf-8")
        self._append(
            f"[{self.payload['run_started_at']}] RUN START "
            f"world_size={world_size} metrics={','.join(metrics)}"
        )
        self._write_json()

    def _append(self, message: str) -> None:
        with self.text_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(message, flush=True)

    def _write_json(self) -> None:
        atomic_write_json(self.json_path, self.payload)

    def start_stage(self, metrics: list[str]) -> None:
        if self.active_stage is not None:
            raise RuntimeError("A metric timing stage is already active")
        started_at = _now()
        self.active_stage = (list(metrics), started_at, time.perf_counter())
        self._append(f"[{started_at}] START {'+'.join(metrics)}")

    def finish_stage(self) -> None:
        if self.active_stage is None:
            raise RuntimeError("No metric timing stage is active")
        metrics, started_at, started = self.active_stage
        elapsed = time.perf_counter() - started
        finished_at = _now()
        self.payload["stages"].append(
            {
                "metrics": metrics,
                "started_at": started_at,
                "finished_at": finished_at,
                "elapsed_seconds": round(elapsed, 3),
                "elapsed": format_duration(elapsed),
            }
        )
        self._append(
            f"[{finished_at}] END {'+'.join(metrics)} "
            f"elapsed={format_duration(elapsed)} ({elapsed:.3f}s)"
        )
        self.active_stage = None
        self._write_json()

    def finish_run(self) -> None:
        elapsed = time.perf_counter() - self.run_started
        finished_at = _now()
        self.payload.update(
            {
                "status": "completed",
                "run_finished_at": finished_at,
                "total_elapsed_seconds": round(elapsed, 3),
                "total_elapsed": format_duration(elapsed),
            }
        )
        self._append(
            f"[{finished_at}] RUN END elapsed={format_duration(elapsed)} "
            f"({elapsed:.3f}s)"
        )
        self._write_json()

    def fail_run(self, error: BaseException) -> None:
        elapsed = time.perf_counter() - self.run_started
        finished_at = _now()
        self.payload.update(
            {
                "status": "failed",
                "run_finished_at": finished_at,
                "total_elapsed_seconds": round(elapsed, 3),
                "total_elapsed": format_duration(elapsed),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        self._append(
            f"[{finished_at}] RUN FAILED elapsed={format_duration(elapsed)} "
            f"error={type(error).__name__}: {error}"
        )
        self._write_json()
