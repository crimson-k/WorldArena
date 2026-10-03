"""Evaluate each completed checkpoint from the two running stage-2 experiments."""

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
BWM_ROOT = Path("/data1/fangxuebin/boundless-world-model")
DATASET = BWM_ROOT / "converted_dataset_bwm_6tasks_256x192"
OUTPUT_ROOT = PROJECT_ROOT / "evaluation_runs/stage2_checkpoint_monitor"
EXPORT_ROOT = Path("/tmp/worldarena_stage2_checkpoint_exports")
BWM_PYTHON = "/home/fangxuebin/.conda/envs/BWM/bin/python"
EVAL_PYTHON = "/home/fangxuebin/.conda/envs/WorldArena/bin/python"
JEPA_PYTHON = "/home/fangxuebin/.conda/envs/WorldArena_JEPA/bin/python"
MODEL_PATH = "/data1/common_model/modelscope/Wan-AI/Wan2.2-TI2V-5B/"
RUNS = {
    "stage2_actionshuffle_e30_block15_w0p01": BWM_ROOT / "outputs/training/stage2_actionshuffle_e30_block15_w0p01",
    "stage2_actionshuffle_e30_block15_w0p5_restart_20261001": BWM_ROOT / "outputs/training/stage2_actionshuffle_e30_block15_w0p5_restart_20261001",
    "stage2_actionshuffle_e30_block15_w0p01_restart_20261001": BWM_ROOT / "outputs/training/stage2_actionshuffle_e30_block15_w0p01_restart_20261001",
}
GPU_ORDER = (4, 5, 6, 7, 0, 1, 2, 3)
EPISODE_PATTERN = re.compile(r"episode[_-]?(\d+)\.mp4$")
METRICS = "psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,depth_accuracy,subject_consistency,trajectory_accuracy"
NON_JEPA = [metric for metric in METRICS.split(",") if metric != "jepa_similarity"]


def expected_episodes():
    with (DATASET / "metadata_test.jsonl").open() as stream:
        episodes = {
            (row["task"], int(row["source_episode_index"]))
            for row in (json.loads(line) for line in stream if line.strip())
        }
    assert len(episodes) == 60, f"Expected 60 validation episodes, got {len(episodes)}"
    return episodes


def checkpoints():
    for name, run_dir in RUNS.items():
        for metadata in sorted((run_dir / "states").glob("step-*.json")):
            step = int(metadata.stem.removeprefix("step-"))
            state = run_dir / "states" / metadata.stem / "pytorch_model_fsdp_0"
            if state.is_dir():
                yield name, step, state


def gpu_memory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return {int(parts[0]): (int(parts[1]), int(parts[2]))
            for line in output.splitlines() if (parts := [part.strip() for part in line.split(",")])}


def training_gpus():
    protected = set()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "scripts/train.py" not in command or not any(str(path) in command for path in RUNS.values()):
                continue
            for item in (proc / "environ").read_bytes().split(b"\0"):
                if item.startswith(b"CUDA_VISIBLE_DEVICES="):
                    protected.update(int(gpu) for gpu in item.split(b"=", 1)[1].split(b",") if gpu)
        except (OSError, ValueError):
            continue
    return protected


def generated_episodes(directory):
    found = set()
    for video in directory.glob("*/*.mp4"):
        match = EPISODE_PATTERN.fullmatch(video.name)
        if match and video.stat().st_size:
            found.add((video.parent.name, int(match.group(1))))
    return found


def result_count(path):
    with path.open(newline="") as stream:
        return sum(row["sample_id"] != "AVERAGE" for row in csv.DictReader(stream))


def stage_paths(name, step, stage):
    root = OUTPUT_ROOT / name / f"step-{step}"
    return root, root / "stages" / f"{stage}.done", root / "stages" / f"{stage}.failed"


def observed_memory(stage, fallback):
    peaks = []
    for marker in OUTPUT_ROOT.glob(f"*/step-*/stages/{stage}.done"):
        peak = json.loads(marker.read_text()).get("peak_extra_mib", 0)
        if peak > 0:
            peaks.append(peak)
    return max(peaks) if peaks else fallback


def recoverable(output, stage):
    evaluation = output if stage == "full" else output / "evaluation_left"
    return (evaluation / "run_manifest.json").is_file() and all(
        (evaluation / "results/metrics" / f"{metric}.json").is_file()
        for metric in NON_JEPA
    )


def recover_evaluation(output, stage, gpu):
    def finish(evaluation):
        if not (evaluation / "results/jepa/results.json").is_file():
            subprocess.run([
                EVAL_PYTHON, "-m", "video_quality.cli", "jepa",
                "--real-dir", str(evaluation / "cache/jepa_pairs/gt"),
                "--gen-dir", str(evaluation / "cache/jepa_pairs/generated"),
                "--output-dir", str(evaluation),
                "--config", str(PROJECT_ROOT / "video_quality/config/config.yaml"),
                "--jepa-python", JEPA_PYTHON, "--gpu", str(gpu),
            ], cwd=PROJECT_ROOT, check=True)
        subprocess.run([
            EVAL_PYTHON, "-m", "video_quality.cli", "aggregate",
            "--manifest", str(evaluation / "run_manifest.json"),
            "--output-dir", str(evaluation), "--metrics", METRICS,
        ], cwd=PROJECT_ROOT, check=True)

    if stage == "full":
        finish(output)
        return

    left = output / "evaluation_left"
    right = output / "evaluation_right"
    finish(left)
    if not (right / "results/results.csv").is_file():
        subprocess.run([
            EVAL_PYTHON, "-m", "video_quality.stack_and_evaluate",
            "--generated-root", str(output / "generated_crops_right"),
            "--gt-root", str(output / "gt_crops_right"),
            "--output-dir", str(right), "--trim-gt-to-generated",
            "--jepa-python", JEPA_PYTHON, "--gpus", str(gpu),
            "--processes-per-gpu", "1",
        ], cwd=PROJECT_ROOT, check=True)
    from interaction_mask.crop_and_evaluate import write_average_results
    write_average_results(output, {"left": left / "results/results.csv",
                                   "right": right / "results/results.csv"})


def command_for(name, step, state, stage, gpu):
    root, _, _ = stage_paths(name, step, stage)
    infer_dir = root / "generated"
    if stage == "export":
        export_dir = EXPORT_ROOT / name / f"step-{step}"
        temporary = export_dir.with_name(export_dir.name + ".pending")
        shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(export_dir, ignore_errors=True)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        return [str(Path(BWM_PYTHON).with_name("accelerate")), "merge-weights",
                str(state), str(temporary)], None
    if stage == "infer":
        shutil.rmtree(infer_dir, ignore_errors=True)
        infer_dir.mkdir(parents=True)
        model = EXPORT_ROOT / name / f"step-{step}" / "model.safetensors"
        command = [BWM_PYTHON, "scripts/infer_distributed.py",
                   "--config", "configs/infer/infer.yaml", "--model_paths", MODEL_PATH,
                   "--ckpt_path", str(model), "--dataset_base_path", str(DATASET),
                   "--dataset_metadata_path", str(DATASET / "metadata_test.jsonl"),
                   "--action_stat_path", str(DATASET / "stat.json"),
                   "--output_path", str(infer_dir), "--max_samples", "60"]
        return command, {"CUDA_VISIBLE_DEVICES": str(gpu)}
    output = root / stage
    if recoverable(output, stage):
        return [sys.executable, str(Path(__file__).resolve()),
                "--recover-output", str(output), "--recover-stage", stage,
                "--recover-gpu", str(gpu)], None
    shutil.rmtree(output, ignore_errors=True)
    if stage == "full":
        return [EVAL_PYTHON, "-m", "video_quality.stack_and_evaluate",
                "--generated-root", str(infer_dir), "--gt-root", str(DATASET / "videos"),
                "--output-dir", str(output), "--trim-gt-to-generated",
                "--jepa-python", JEPA_PYTHON,
                "--gpus", str(gpu), "--processes-per-gpu", "1"], None
    return [sys.executable, str(PROJECT_ROOT / "interaction_mask/crop_and_evaluate.py"),
            "--generated-root", str(infer_dir), "--gt-root", str(DATASET),
            "--robotwin-dataset", "/data1/common_data/RoboTwin2.0/dataset",
            "--output-dir", str(output), "--width", "160", "--height", "120",
            "--trim-gt-to-generated", "--gpus", str(gpu),
            "--processes-per-gpu", "1", "--eval-python", EVAL_PYTHON,
            "--jepa-python", JEPA_PYTHON], None


def validate_stage(name, step, stage, episodes):
    root, _, _ = stage_paths(name, step, stage)
    if stage == "export":
        temporary = EXPORT_ROOT / name / f"step-{step}.pending"
        model = temporary / "model.safetensors"
        if not model.is_file() or not model.stat().st_size:
            raise RuntimeError(f"Exported model is missing: {model}")
        temporary.rename(temporary.with_name(f"step-{step}"))
    elif stage == "infer":
        found = generated_episodes(root / "generated")
        if found != episodes:
            raise RuntimeError(f"Expected 60 generated episodes; found {len(found)}, missing={sorted(episodes - found)}")
        shutil.rmtree(EXPORT_ROOT / name / f"step-{step}")
    elif stage == "full":
        count = result_count(root / "full/results/results.csv")
        if count != len(episodes):
            raise RuntimeError(f"Full evaluation has {count} results, expected {len(episodes)}")
    else:
        count = result_count(root / "crop/results.csv")
        if count != len(episodes):
            raise RuntimeError(f"Crop evaluation has {count} results, expected {len(episodes)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--infer-memory-mib", type=int, default=45000)
    parser.add_argument("--eval-memory-mib", type=int, default=36000)
    parser.add_argument("--reserve-mib", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recover-output", type=Path)
    parser.add_argument("--recover-stage", choices=("full", "crop"))
    parser.add_argument("--recover-gpu", type=int)
    args = parser.parse_args()
    if args.recover_output:
        recover_evaluation(args.recover_output, args.recover_stage, args.recover_gpu)
        return
    episodes = expected_episodes()
    if args.dry_run:
        for name, step, state in checkpoints():
            print(f"{name} step-{step}: {state}")
        print(f"Validation episodes: {len(episodes)}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT_ROOT / "monitor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = {}
    measured = {"infer": observed_memory("infer", args.infer_memory_mib),
                "full": observed_memory("full", args.eval_memory_mib),
                "crop": observed_memory("crop", args.eval_memory_mib)}
    print(f"Monitoring {', '.join(RUNS)}; outputs={OUTPUT_ROOT}", flush=True)
    while True:
        memory = gpu_memory()
        for key, job in list(jobs.items()):
            gpu = job["gpu"]
            if gpu is not None:
                job["peak"] = max(job["peak"], memory[gpu][0] - job["start_used"])
            code = job["process"].poll()
            if code is None:
                continue
            name, step, stage = key
            root, done, failed = stage_paths(name, step, stage)
            try:
                if code:
                    raise RuntimeError(f"Process exited with status {code}; see {root / 'logs' / (stage + '.log')}")
                validate_stage(name, step, stage, episodes)
                done.write_text(json.dumps({"finished_at": time.time(), "gpu": gpu,
                                            "peak_extra_mib": job["peak"]}) + "\n")
                failed.unlink(missing_ok=True)
                if gpu is not None:
                    measured[stage] = max(measured[stage], job["peak"])
                print(f"[done] {name} step-{step} {stage} gpu={gpu} peak_extra={job['peak']} MiB", flush=True)
            except Exception as exc:
                failed.write_text(f"{time.ctime()}: {exc}\n")
                print(f"[failed] {name} step-{step} {stage}: {exc}", flush=True)
            del jobs[key]

        busy_gpus = {job["gpu"] for job in jobs.values() if job["gpu"] is not None}
        protected = training_gpus()
        for name, step, state in checkpoints():
            for stage in ("export", "infer", "full", "crop"):
                key = (name, step, stage)
                root, done, failed = stage_paths(name, step, stage)
                infer_done = stage_paths(name, step, "infer")[1].exists()
                model = EXPORT_ROOT / name / f"step-{step}" / "model.safetensors"
                if stage == "export" and done.exists() and not infer_done and not model.is_file():
                    done.unlink()
                if done.exists() or failed.exists() or key in jobs:
                    continue
                export_done = stage_paths(name, step, "export")[1].exists()
                if stage == "export" and (infer_done or export_done and model.is_file()):
                    continue
                if stage == "infer" and (not export_done or not model.is_file()):
                    continue
                if stage in ("full", "crop") and not infer_done:
                    continue
                if stage == "export" and any(item[2] == "export" for item in jobs):
                    continue
                gpu = None
                if stage != "export":
                    required = measured[stage] + args.reserve_mib
                    gpu = next((candidate for candidate in GPU_ORDER
                                if candidate not in busy_gpus and candidate not in protected
                                and candidate in memory and memory[candidate][1] >= required), None)
                    if gpu is None:
                        continue
                root.joinpath("stages").mkdir(parents=True, exist_ok=True)
                root.joinpath("logs").mkdir(exist_ok=True)
                command, extra_env = command_for(name, step, state, stage, gpu)
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                if extra_env:
                    environment.update(extra_env)
                if stage in ("full", "crop"):
                    environment["PATH"] = str(Path(EVAL_PYTHON).parent) + os.pathsep + environment["PATH"]
                log_path = root / "logs" / f"{stage}.log"
                with log_path.open("a") as log:
                    log.write(f"\n[{time.ctime()}] {' '.join(command)}\n")
                    log.flush()
                    process = subprocess.Popen(command, cwd=BWM_ROOT if stage in ("export", "infer") else PROJECT_ROOT,
                                               env=environment, stdout=log, stderr=subprocess.STDOUT)
                jobs[key] = {"process": process, "gpu": gpu,
                             "start_used": memory[gpu][0] if gpu is not None else 0, "peak": 0}
                if gpu is not None:
                    busy_gpus.add(gpu)
                print(f"[start] {name} step-{step} {stage} gpu={gpu} log={log_path}", flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
