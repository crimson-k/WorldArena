import glob
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
from PIL import Image
from tqdm import tqdm


def cal_ssim(gt_img, pd_img):
    return structural_similarity(gt_img, pd_img, channel_axis=-1)


def list_frame_images(folder):
    image_list = []
    for ext in ("png", "jpg", "jpeg"):
        image_list += glob.glob(os.path.join(folder, f"frame_*.{ext}"))
    image_list.sort()
    return image_list


def compute_basic_metrics(gt_path, pd_path, metric_names=["psnr", "ssim"]):

    metric_funcs = dict({
        "psnr": peak_signal_noise_ratio,
        "ssim": cal_ssim,
    })
    res = dict()
    for _ in metric_names:
        assert(_ in metric_funcs)
        res.update({_: dict()})
    
    for task_id in sorted(os.listdir(pd_path)):
        task_path = os.path.join(pd_path, task_id)
        
        for _ in metric_names:
            res[_][task_id] = {}

        for episode_id in tqdm(sorted(os.listdir(task_path))):
            if episode_id.endswith(('.png', '.json')): 
                continue
            
            for _ in metric_names:
                res[_][task_id][episode_id] = {}

            gt_video_dir = os.path.join(gt_path, task_id, episode_id, "video")
            gt_image_list = list_frame_images(gt_video_dir)
            n_frames = len(gt_image_list)
            if n_frames == 0:
                raise FileNotFoundError(f"No GT frames found in {gt_video_dir}")

            gid_list = sorted(os.listdir(os.path.join(task_path, episode_id)))
            for gid in gid_list:
                pd_video_dir = os.path.join(task_path, episode_id, gid, "video")
                pd_image_list = list_frame_images(pd_video_dir)
                if len(pd_image_list) != n_frames:
                    raise ValueError(
                        f"Frame count mismatch for {task_id}/{episode_id}/{gid}: "
                        f"pred={len(pd_image_list)}, gt={n_frames}, "
                        f"pred_dir={pd_video_dir}, gt_dir={gt_video_dir}"
                    )
                cur_metrics = {_: 0.0 for _ in metric_names}
                for pd_img, gt_img in zip(pd_image_list, gt_image_list):
                    pd_img = np.asanyarray(Image.open(pd_img))
                    gt_img = np.asanyarray(Image.open(gt_img))
                    if pd_img.shape != gt_img.shape:
                        pd_img = cv2.resize(pd_img, dsize=tuple(gt_img.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)
                    for _ in metric_names:
                        cur_metrics[_] += metric_funcs[_](gt_img, pd_img)
                for _ in metric_names:
                    res[_][task_id][episode_id].update({gid: cur_metrics[_]/n_frames})
    
    return res
