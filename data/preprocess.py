"""
Fitify ML — Data Preprocessing
Scans the dataset, builds label maps, and creates train/val splits.
"""

import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2


def scan_dataset(data_dir: str = "data") -> Dict[str, List[str]]:
    """
    Scan the verified_data directory to discover all exercises and their videos.

    Returns:
        Dict mapping exercise_name -> list of video file paths
    """
    exercise_videos = defaultdict(list)
    search_dirs = [
        os.path.join(data_dir, "verified_data", "data_btc_10s"),
        os.path.join(data_dir, "verified_data", "data_crawl_10s"),
    ]

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            print(f"⚠️  Directory not found: {search_dir}")
            continue

        for exercise_name in sorted(os.listdir(search_dir)):
            exercise_path = os.path.join(search_dir, exercise_name)
            if not os.path.isdir(exercise_path):
                continue

            for video_file in os.listdir(exercise_path):
                if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(exercise_path, video_file)
                    exercise_videos[exercise_name].append(video_path)

    return dict(exercise_videos)


def validate_videos(video_paths: List[str]) -> List[str]:
    """
    Validate that video files can be opened and have at least 1 frame.

    Returns:
        List of valid video paths
    """
    valid = []
    invalid_count = 0

    for path in video_paths:
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    valid.append(path)
                else:
                    invalid_count += 1
            else:
                invalid_count += 1
            cap.release()
        except Exception:
            invalid_count += 1

    if invalid_count > 0:
        print(f"   ⚠️  Skipped {invalid_count} unreadable video(s)")

    return valid


def create_label_map(exercise_videos: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Create a mapping from exercise name to integer label.

    Returns:
        Dict mapping exercise_name -> label_id
    """
    exercises = sorted(exercise_videos.keys())
    label_map = {name: idx for idx, name in enumerate(exercises)}
    return label_map


def create_splits(
    exercise_videos: Dict[str, List[str]],
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Create stratified train/validation splits.

    Returns:
        (train_samples, val_samples) where each sample is
        {"video_path": str, "label": str}
    """
    random.seed(seed)
    train_samples = []
    val_samples = []

    for exercise_name, videos in exercise_videos.items():
        shuffled = videos.copy()
        random.shuffle(shuffled)

        split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
        train_vids = shuffled[:split_idx]
        val_vids = shuffled[split_idx:]

        for v in train_vids:
            train_samples.append({"video_path": v, "label": exercise_name})
        for v in val_vids:
            val_samples.append({"video_path": v, "label": exercise_name})

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def preprocess(data_dir: str = "data", val_ratio: float = 0.2):
    """
    Full preprocessing pipeline: scan, validate, build labels, split.
    """
    print("=" * 60)
    print("  Fitify ML — Data Preprocessing")
    print("=" * 60)
    print()

    # Step 1: Scan
    print("🔍 Scanning dataset...")
    exercise_videos = scan_dataset(data_dir)

    if not exercise_videos:
        print("❌ No exercises found! Make sure the dataset is in the 'data/' folder.")
        print("   Expected structure:")
        print("     data/verified_data/data_btc_10s/<exercise_name>/*.mp4")
        print("     data/verified_data/data_crawl_10s/<exercise_name>/*.mp4")
        return

    total_videos = sum(len(v) for v in exercise_videos.values())
    print(f"   Found {len(exercise_videos)} exercises, {total_videos} total videos")
    print()

    # Step 2: Validate
    print("✅ Validating videos...")
    validated_exercises = {}
    for name, videos in exercise_videos.items():
        valid = validate_videos(videos)
        if valid:
            validated_exercises[name] = valid
            print(f"   {name}: {len(valid)} valid videos")

    print()

    # Step 3: Label map
    print("🏷️  Building label map...")
    label_map = create_label_map(validated_exercises)
    label_map_path = os.path.join(data_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"   Saved to {label_map_path}")

    # Also save inverse map
    id_to_label = {v: k for k, v in label_map.items()}
    id_to_label_path = os.path.join(data_dir, "id_to_label.json")
    with open(id_to_label_path, "w") as f:
        json.dump(id_to_label, f, indent=2)
    print(f"   Saved inverse map to {id_to_label_path}")
    print()

    # Step 4: Train/Val splits
    print(f"📊 Creating train/val split ({int((1-val_ratio)*100)}/{int(val_ratio*100)})...")
    train_samples, val_samples = create_splits(validated_exercises, val_ratio)

    splits = {
        "train": train_samples,
        "val": val_samples,
    }
    splits_path = os.path.join(data_dir, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"   Train: {len(train_samples)} samples")
    print(f"   Val:   {len(val_samples)} samples")
    print(f"   Saved to {splits_path}")
    print()

    # Summary
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Exercises:  {len(label_map)}")
    print(f"  Train set:  {len(train_samples)} videos")
    print(f"  Val set:    {len(val_samples)} videos")
    print(f"  Label map:  {label_map_path}")
    print(f"  Splits:     {splits_path}")
    print("=" * 60)

    return label_map, splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Fitify ML dataset")
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root data directory (default: data)"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    args = parser.parse_args()

    preprocess(args.data_dir, args.val_ratio)
