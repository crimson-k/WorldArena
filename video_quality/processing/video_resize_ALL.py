import os
import cv2
import decord
import numpy as np
import yaml


def resize_and_save_images(image_folder, target_size=(640, 480)):
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    for fname in image_files:
        img_path = os.path.join(image_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read {img_path}")
            continue
        if img.shape[:2] != tuple(target_size[::-1]):
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(img_path, img_resized)
        else:
            f"Keep: {img_path}"


def process_video_or_images(input_base, is_gt=False):
    if not os.path.exists(input_base):
        return
        
    for task in sorted(os.listdir(input_base)):
        task_path = os.path.join(input_base, task)
        if not os.path.isdir(task_path):
            continue
            
        for episode in os.listdir(task_path):
            if episode.endswith(('.png', '.json')):
                continue
            episode_path = os.path.join(task_path, episode)
            
            if is_gt:
                # 真实视频 (GT) 的路径结构：episode/video
                video_path = os.path.join(episode_path, "video")
                if os.path.exists(video_path):
                    print(f"[GT Image Folder] Processing {video_path}")
                    resize_and_save_images(video_path)
            else:
                # 生成视频的路径结构：episode/gid/video
                for gid in sorted(os.listdir(episode_path)):
                    video_path = os.path.join(episode_path, gid, "video")
                    if os.path.exists(video_path):
                        print(f"[Image Folder] Processing {video_path}")
                        resize_and_save_images(video_path)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--resize_gt', action='store_true', help='Also resize ground truth images')

    args = parser.parse_args()
    config = load_config(args.config_path)

    # 1. 处理生成的模型视频
    data_base = config['data']['val_base']
    process_video_or_images(data_base, is_gt=False)

    # 2. 如果传了 --resize_gt 参数，则处理真实视频 (gt_dataset)
    if args.resize_gt:
        gt_base = config['data'].get('gt_path')  
        if gt_base and os.path.exists(gt_base):
            print(f">>> Resizing Ground Truth dataset: {gt_base}")
            process_video_or_images(gt_base, is_gt=True)
        else:
            print(f"Warning: gt_path not found in config or path does not exist: {gt_base}")