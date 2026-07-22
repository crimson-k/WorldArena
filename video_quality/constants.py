"""Stable public names used by the focused evaluation pipeline."""

SUPPORTED_METRICS = (
    "psnr",
    "ssim",
    "aesthetic_quality",
    "image_quality",
    "jepa_similarity",
    "subject_consistency",
    "trajectory_accuracy",
    "depth_accuracy",
)

METRIC_COLUMNS = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "aesthetic_quality": "Aesthetic Quality",
    "image_quality": "Image Quality",
    "jepa_similarity": "JEPA Similarity",
    "subject_consistency": "Subject Consistency",
    "trajectory_accuracy": "Trajectory Accuracy",
    "depth_accuracy": "Depth Accuracy",
}

JPEG_METRICS = frozenset(
    {
        "aesthetic_quality",
        "image_quality",
        "subject_consistency",
        "trajectory_accuracy",
        "depth_accuracy",
    }
)

WORLD_ARENA_BOUNDS = {
    "trajectory_accuracy": {"min": 0.0, "max": 40.8540, "invert": False},
    "depth_accuracy": {"min": 0.2228, "max": 4.3711, "invert": True},
}


def normalize_worldarena(metric: str, value: float) -> float:
    """Apply the pinned WorldArena leaderboard normalization."""
    bounds = WORLD_ARENA_BOUNDS[metric]
    normalized = (float(value) - bounds["min"]) / (bounds["max"] - bounds["min"])
    normalized = max(0.0, min(1.0, normalized))
    return 1.0 - normalized if bounds["invert"] else normalized


def normalize_metrics(metrics) -> list[str]:
    values = [str(metric).strip().lower() for metric in metrics if str(metric).strip()]
    if not values:
        raise ValueError("metric list cannot be empty")
    unknown = sorted(set(values) - set(SUPPORTED_METRICS))
    if unknown:
        raise ValueError(
            f"unsupported metrics: {', '.join(unknown)}; allowed: {', '.join(SUPPORTED_METRICS)}"
        )
    if len(values) != len(set(values)):
        raise ValueError("metric list contains duplicates")
    selected = set(values)
    return [metric for metric in SUPPORTED_METRICS if metric in selected]
