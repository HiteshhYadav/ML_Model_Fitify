"""
Fitify ML — Demo Predictor (Pose-Landmark Model)
Loads the trained Bi-GRU pose classifier (best.pt), extracts BlazePose landmarks from a video,
normalizes the sequence, and predicts the exercise type with confidence scores.
"""

import os
import random
import json
import argparse
import numpy as np
import cv2
import torch
import mediapipe as mp

import tkinter as tk
from tkinter import filedialog

from models.nets import build_model
from models.dataset import normalize_sequence

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def select_video_via_dialog():
    """Open a file explorer dialog to select a video file."""
    root = tk.Tk()
    root.withdraw()  # Hide main Tk window
    root.attributes("-topmost", True)  # Bring dialog to the front
    
    file_path = filedialog.askopenfilename(
        title="Select Workout Video for Analysis",
        filetypes=[
            ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()
    return file_path



def sample_frame_indices(total, num):
    if total <= 0:
        return []
    if total <= num:
        return list(range(total))
    return list(np.linspace(0, total - 1, num).astype(int))


def extract_landmarks_from_video(video_path, pose, num_frames=32):
    """Extract (T, 33, 4) landmark array from video in pixel coordinates."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps if total > 0 else 0.0

    idxs = sample_frame_indices(total, num_frames)
    frames = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        res = pose.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            lm = np.array([[p.x * width, p.y * height, p.z * width, p.visibility]
                           for p in res.pose_landmarks.landmark], dtype=np.float32)
        else:
            lm = np.zeros((33, 4), dtype=np.float32)
        frames.append(lm)
    cap.release()

    if not frames:
        return None, duration

    arr = np.stack(frames, axis=0)
    # Pad/truncate to exactly num_frames
    if arr.shape[0] < num_frames:
        pad = np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:num_frames], duration


def get_random_video(dataset_dir, allowed_classes):
    """Find a random video in the dataset directory that belongs to the allowed classes."""
    video_files = []
    normalized_allowed = {c.strip().lower().replace("_", " ").replace("-", " ") for c in allowed_classes}
    for root, _, files in os.walk(dataset_dir):
        # Check if the folder name (or parent path) contains one of the allowed classes
        dir_name = os.path.basename(root).strip().lower().replace("_", " ").replace("-", " ")
        if dir_name not in normalized_allowed:
            continue
        for file in files:
            if os.path.splitext(file)[1].lower() in VIDEO_EXT:
                video_files.append(os.path.join(root, file))

    if not video_files:
        raise FileNotFoundError(f"No videos of supported classes found under {dataset_dir}")
    return random.choice(video_files)



def main():
    parser = argparse.ArgumentParser(description="Test Fitify ML pose-landmark classifier on a video")
    parser.add_argument("--video", type=str, default=None, help="Path to specific video file (if None, picks a random one)")
    parser.add_argument("--dataset", type=str, default="data/verified_data", help="Dataset directory to pick a random video from")
    parser.add_argument("--ckpt", type=str, default="checkpoints", help="Directory containing best.pt and label maps")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames model expects")
    args = parser.parse_args()

    print("=" * 60)
    print("  Fitify ML — Pose Landmark Classifier Test")
    print("=" * 60)

    # 1. Check if checkpoint files exist
    ckpt_path = os.path.join(args.ckpt, "best.pt")
    label_map_path = os.path.join("data", "label_map.json")
    id_to_label_path = os.path.join("data", "id_to_label.json")

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint file not found at {ckpt_path}")
        return
    if not os.path.exists(id_to_label_path):
        print(f"❌ id_to_label.json file not found at {id_to_label_path}")
        return

    # 2. Load label map
    with open(id_to_label_path, "r") as fh:
        id_to_label = json.load(fh)
    num_classes = len(id_to_label)

    # 3. Determine video path
    if args.video:
        video_path = args.video
        print(f"📁 Using specified video: {video_path}")
    else:
        print("📂 Opening File Explorer to select a video... (Close or cancel dialog to pick a random video instead)")
        video_path = select_video_via_dialog()
        if video_path:
            print(f"📁 Selected video: {video_path}")
        else:
            print("⚠️ File Explorer canceled. Selecting a random video from the dataset instead...")
            try:
                video_path = get_random_video(args.dataset, id_to_label.values())
                print(f"🎲 Randomly selected video: {video_path}")
            except Exception as e:
                print(f"❌ Error finding video: {e}")
                return



    # 4. Load the Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    model = build_model("gru", num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model weights from {ckpt_path}")

    # 5. Initialize MediaPipe
    print("✨ Extracting BlazePose landmarks from video...")
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    raw_landmarks, duration = extract_landmarks_from_video(video_path, pose, args.num_frames)

    if raw_landmarks is None:
        print("❌ Landmark extraction failed. Could not process any frames.")
        return

    # 6. Normalize sequence
    normalized_seq = normalize_sequence(raw_landmarks)  # Shape (T, 99)

    # 7. Model prediction
    input_tensor = torch.from_numpy(normalized_seq).unsqueeze(0).to(device)  # (1, T, 99)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)[0]

    # 8. Sort and output results
    top_indices = torch.argsort(probs, descending=True)

    print()
    print("=" * 60)
    print("  CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  Video Duration: {duration:.2f} seconds")
    print(f"  Actual exercise (from folder): {os.path.basename(os.path.dirname(video_path)).upper()}")
    print("-" * 60)
    print("  Top Predictions:")
    for rank, idx in enumerate(top_indices[:3], 1):
        label_id = str(idx.item())
        label_name = id_to_label[label_id]
        probability = probs[idx].item()
        indicator = "🎯 (BEST MATCH)" if rank == 1 else "   "
        print(f"  {rank}. {label_name:<22} {probability*100:6.2f}% {indicator}")
    print("=" * 60)


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()

