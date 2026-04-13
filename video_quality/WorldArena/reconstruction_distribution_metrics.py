import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d, interpolate
from tqdm import tqdm

from .distributed import get_rank
from .utils import load_dimension_info

FID_DIMS = 2048
DEFAULT_BATCH_SIZE = 64
DEFAULT_FVD_INPUT_RES = 224
DEFAULT_FVD_CHUNK_SIZE = 16


def _ensure_torch_hub_checkpoint(local_ckpt: str):
    if not local_ckpt:
        return
    src = Path(local_ckpt)
    if not src.is_file():
        return

    hub_dir = Path(torch.hub.get_dir())
    if not hub_dir.exists() or not os.access(hub_dir, os.W_OK):
        fallback = Path("/tmp/worldarena_torch_hub")
        fallback.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(fallback)
        torch.hub.set_dir(str(fallback))
        hub_dir = fallback

    dst = hub_dir / "checkpoints" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        import shutil
        shutil.copy2(src, dst)


def _read_video_frames(video_path: str) -> List[np.ndarray]:
    if os.path.isdir(video_path):
        frame_paths = []
        for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
            frame_paths.extend(glob for glob in Path(video_path).glob(f"frame_*.{ext}"))
        frame_paths = sorted(frame_paths)
        frames = []
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _align_frame_pair(gt_frame: np.ndarray, pred_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gt_frame.shape[:2]
    if pred_frame.shape[:2] != (h, w):
        pred_frame = cv2.resize(pred_frame, (w, h), interpolation=cv2.INTER_CUBIC)
    return gt_frame, pred_frame


def _resolve_gt_video_path(pred_video_path: str, gt_root: str) -> Optional[str]:
    parts = Path(pred_video_path).parts
    if len(parts) < 4:
        return None
    task_id = parts[-4]
    episode_id = parts[-3]

    gt_dir = os.path.join(gt_root, task_id, episode_id, "video")
    if os.path.isdir(gt_dir):
        return gt_dir

    for ext in (".mp4", ".avi", ".mov"):
        gt_file = os.path.join(gt_root, task_id, episode_id, f"video{ext}")
        if os.path.isfile(gt_file):
            return gt_file
    return None


def _list_video_pairs(json_dir: str, dimension: str, gt_root: str) -> List[Tuple[str, str]]:
    video_list = load_dimension_info(json_dir, dimension=dimension)
    if not isinstance(video_list, list):
        return []
    pairs = []
    for pred_video_path in video_list:
        gt_video_path = _resolve_gt_video_path(pred_video_path, gt_root)
        if gt_video_path is None:
            continue
        pairs.append((pred_video_path, gt_video_path))
    return pairs


def _compute_stats(features: np.ndarray):
    if features.shape[0] <= 0:
        raise RuntimeError("No features to compute statistics")
    mu = np.mean(features, axis=0)
    if features.shape[0] == 1:
        sigma = np.eye(features.shape[1], dtype=np.float64) * 1e-6
    else:
        sigma = np.cov(features, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)


def _extract_inception_features(frames: np.ndarray, model, device: torch.device, batch_size: int):
    if frames.shape[0] <= 0:
        return np.empty((0, FID_DIMS), dtype=np.float32)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float()
    activations = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start:start + batch_size].to(device)
            pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            activations.append(pred)
    if not activations:
        return np.empty((0, FID_DIMS), dtype=np.float32)
    return np.concatenate(activations, axis=0)


def _extract_i3d_features(
    frames: np.ndarray,
    i3d_model,
    device: torch.device,
    chunk_size: int,
    fvd_input_res: int = DEFAULT_FVD_INPUT_RES,
):
    features = []
    if frames.shape[0] < chunk_size:
        return features
    for start in range(0, frames.shape[0], chunk_size):
        chunk = frames[start:start + chunk_size]
        if chunk.shape[0] != chunk_size:
            continue
        tensor = torch.from_numpy(chunk).permute(0, 3, 1, 2).contiguous().float()
        if tensor.shape[-2] != fvd_input_res or tensor.shape[-1] != fvd_input_res:
            tensor = interpolate(
                tensor, size=(fvd_input_res, fvd_input_res), mode="bilinear", align_corners=False
            )
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        tensor = 2.0 * tensor - 1.0
        with torch.no_grad():
            feat = i3d_model(tensor, rescale=False, resize=False, return_features=True)
        features.append(feat.squeeze(0).cpu().numpy())
    return features


def compute_mse(json_dir, submodules_list, **kwargs):
    gt_root = kwargs.get("gt_path")
    if not gt_root:
        raise ValueError("mse requires gt_path in evaluate kwargs")

    pairs = _list_video_pairs(json_dir, dimension="mse", gt_root=gt_root)
    scores = []
    video_results = []

    for pred_video_path, gt_video_path in tqdm(pairs, disable=get_rank() > 0):
        pred_frames = _read_video_frames(pred_video_path)
        gt_frames = _read_video_frames(gt_video_path)
        usable = min(len(pred_frames), len(gt_frames))
        if usable <= 0:
            continue

        mse_sum = 0.0
        for idx in range(usable):
            gt_frame, pred_frame = _align_frame_pair(gt_frames[idx], pred_frames[idx])
            gt_frame = gt_frame.astype(np.float32) / 255.0
            pred_frame = pred_frame.astype(np.float32) / 255.0
            mse_sum += float(np.mean((pred_frame - gt_frame) ** 2))

        score = mse_sum / usable
        video_results.append({"video_path": pred_video_path, "video_results": score})
        scores.append(score)

    avg_score = float(np.mean(scores)) if scores else -1.0
    return avg_score, video_results


def compute_lpips(json_dir, submodules_list, **kwargs):
    try:
        import lpips
    except ImportError as exc:
        raise ImportError("lpips package is required for lpips metric") from exc

    gt_root = kwargs.get("gt_path")
    if not gt_root:
        raise ValueError("lpips requires gt_path in evaluate kwargs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = int(kwargs.get("lpips_batch_size", DEFAULT_BATCH_SIZE))

    alexnet_ckpt = ""
    if isinstance(submodules_list, dict):
        alexnet_ckpt = submodules_list.get("alexnet", "") or submodules_list.get("model", "")
    _ensure_torch_hub_checkpoint(alexnet_ckpt)

    model = lpips.LPIPS(net="alex").to(device).eval()
    pairs = _list_video_pairs(json_dir, dimension="lpips", gt_root=gt_root)
    scores = []
    video_results = []

    for pred_video_path, gt_video_path in tqdm(pairs, disable=get_rank() > 0):
        pred_frames = _read_video_frames(pred_video_path)
        gt_frames = _read_video_frames(gt_video_path)
        usable = min(len(pred_frames), len(gt_frames))
        if usable <= 0:
            continue

        lpips_sum = 0.0
        frame_count = 0
        with torch.no_grad():
            for start in range(0, usable, batch_size):
                gt_batch_list = []
                pred_batch_list = []
                end = min(start + batch_size, usable)
                for idx in range(start, end):
                    gt_frame, pred_frame = _align_frame_pair(gt_frames[idx], pred_frames[idx])
                    gt_tensor = torch.from_numpy(gt_frame).permute(2, 0, 1).contiguous().float() / 255.0
                    pred_tensor = torch.from_numpy(pred_frame).permute(2, 0, 1).contiguous().float() / 255.0
                    gt_batch_list.append(gt_tensor)
                    pred_batch_list.append(pred_tensor)

                gt_batch = torch.stack(gt_batch_list, dim=0).to(device)
                pred_batch = torch.stack(pred_batch_list, dim=0).to(device)
                lpips_batch = model(pred_batch, gt_batch, normalize=True)
                lpips_sum += float(lpips_batch.sum().item())
                frame_count += int(lpips_batch.shape[0])

        score = (lpips_sum / frame_count) if frame_count > 0 else -1.0
        video_results.append({"video_path": pred_video_path, "video_results": score})
        if score >= 0.0:
            scores.append(score)

    avg_score = float(np.mean(scores)) if scores else -1.0
    return avg_score, video_results


def compute_fid(json_dir, submodules_list, **kwargs):
    try:
        from pytorch_fid.inception import InceptionV3
    except ImportError as exc:
        raise ImportError("pytorch-fid package is required for fid metric") from exc

    gt_root = kwargs.get("gt_path")
    if not gt_root:
        raise ValueError("fid requires gt_path in evaluate kwargs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = int(kwargs.get("fid_batch_size", DEFAULT_BATCH_SIZE))

    inception_ckpt = ""
    if isinstance(submodules_list, dict):
        inception_ckpt = submodules_list.get("inception", "") or submodules_list.get("model", "")
    _ensure_torch_hub_checkpoint(inception_ckpt)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
    model = InceptionV3([block_idx]).to(device).eval()

    pairs = _list_video_pairs(json_dir, dimension="fid", gt_root=gt_root)
    gt_acts = []
    pred_acts = []
    used_video_paths = []

    for pred_video_path, gt_video_path in tqdm(pairs, desc="Preparing FID features ...", disable=get_rank() > 0):
        pred_frames = _read_video_frames(pred_video_path)
        gt_frames = _read_video_frames(gt_video_path)
        usable = min(len(pred_frames), len(gt_frames))
        if usable <= 0:
            continue

        gt_arr = np.stack(gt_frames[:usable], axis=0).astype(np.float32) / 255.0
        pred_arr = np.stack(
            [_align_frame_pair(gt_frames[idx], pred_frames[idx])[1] for idx in range(usable)],
            axis=0,
        ).astype(np.float32) / 255.0

        gt_feat = _extract_inception_features(gt_arr, model, device, batch_size=batch_size)
        pred_feat = _extract_inception_features(pred_arr, model, device, batch_size=batch_size)
        if gt_feat.shape[0] > 0 and pred_feat.shape[0] > 0:
            gt_acts.append(gt_feat)
            pred_acts.append(pred_feat)
            used_video_paths.append(pred_video_path)

    if not gt_acts or not pred_acts:
        return -1.0, [{"video_path": p, "video_results": -1.0} for p, _ in pairs]

    gt_acts = np.concatenate(gt_acts, axis=0)
    pred_acts = np.concatenate(pred_acts, axis=0)
    mu1, sigma1 = _compute_stats(gt_acts)
    mu2, sigma2 = _compute_stats(pred_acts)
    score = _frechet_distance(mu1, sigma1, mu2, sigma2)
    video_results = [{"video_path": p, "video_results": score} for p in used_video_paths]
    return score, video_results


def compute_fvd(json_dir, submodules_list, **kwargs):
    gt_root = kwargs.get("gt_path")
    if not gt_root:
        raise ValueError("fvd requires gt_path in evaluate kwargs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    i3d_path = ""
    chunk_size = DEFAULT_FVD_CHUNK_SIZE
    if isinstance(submodules_list, dict):
        i3d_path = submodules_list.get("i3d", "") or submodules_list.get("model", "")
        chunk_size = int(submodules_list.get("chunk_size", chunk_size))
    if kwargs.get("fvd_chunk_size") is not None:
        chunk_size = int(kwargs["fvd_chunk_size"])

    if not i3d_path:
        candidate_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_downloaded", "i3d_torchscript.pt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_downloaded", "i3d_torchscript.pt"),
        ]
        i3d_path = next((p for p in candidate_paths if os.path.isfile(p)), "")
    if not i3d_path or not os.path.isfile(i3d_path):
        raise FileNotFoundError(f"fvd requires i3d torchscript model, got: {i3d_path}")

    i3d_model = torch.jit.load(i3d_path, map_location=device).eval()
    pairs = _list_video_pairs(json_dir, dimension="fvd", gt_root=gt_root)
    gt_feats = []
    pred_feats = []
    used_video_paths = []

    for pred_video_path, gt_video_path in tqdm(pairs, desc="Preparing FVD features ...", disable=get_rank() > 0):
        pred_frames = _read_video_frames(pred_video_path)
        gt_frames = _read_video_frames(gt_video_path)
        usable = min(len(pred_frames), len(gt_frames))
        if usable <= 0:
            continue

        gt_arr = np.stack(gt_frames[:usable], axis=0).astype(np.float32) / 255.0
        pred_arr = np.stack(
            [_align_frame_pair(gt_frames[idx], pred_frames[idx])[1] for idx in range(usable)],
            axis=0,
        ).astype(np.float32) / 255.0

        gt_video_feats = _extract_i3d_features(gt_arr, i3d_model, device, chunk_size=chunk_size)
        pred_video_feats = _extract_i3d_features(pred_arr, i3d_model, device, chunk_size=chunk_size)
        if gt_video_feats and pred_video_feats:
            gt_feats.extend(gt_video_feats)
            pred_feats.extend(pred_video_feats)
            used_video_paths.append(pred_video_path)

    if not gt_feats or not pred_feats:
        return -1.0, [{"video_path": p, "video_results": -1.0} for p, _ in pairs]

    gt_feats = np.asarray(gt_feats, dtype=np.float64)
    pred_feats = np.asarray(pred_feats, dtype=np.float64)
    mu1, sigma1 = _compute_stats(gt_feats)
    mu2, sigma2 = _compute_stats(pred_feats)
    score = _frechet_distance(mu1, sigma1, mu2, sigma2)
    video_results = [{"video_path": p, "video_results": score} for p in used_video_paths]
    return score, video_results
