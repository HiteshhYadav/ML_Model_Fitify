"""
Fitify ML — Stage 1: Landmark Extraction
Runs BlazePose over every clip in the chosen exercise folders and saves a
fixed-length landmark sequence per clip. Clips below a visibility threshold
are dropped so garbage never reaches training.

Adapted for nested verified_data structure:
  data/verified_data/data_btc_10s/<exercise>/*.mp4
  data/verified_data/data_crawl_10s/<exercise>/*.mp4

Output: data/landmarks.npz  with arrays
    X     : (N, T, 33, 4)  float32   raw landmarks (x, y, z, visibility)
    y     : (N,)           int64     class id
    files : (N,)           str       source filename (for debugging)
and data/label_map.json / data/id_to_label.json
"""

import os, json, argparse, collections
import numpy as np
import cv2
import mediapipe as mp

# The 12 locked exercises for Fitify.
# Folder names are matched case-insensitively / whitespace-trimmed.
CHOSEN = [
    "squat", "deadlift", "romanian deadlift", "push-up", "pull up",
    "shoulder press", "hammer curl", "lateral raise", "plank",
    "leg raises", "russian twist", "hip thrust",
]
norm = lambda s: s.strip().lower().replace("_", " ").replace("-", " ")
CHOSEN_N = {norm(c) for c in CHOSEN}

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def sample_frame_indices(total, num):
    if total <= 0:
        return []
    if total <= num:
        return list(range(total))
    return list(np.linspace(0, total - 1, num).astype(int))


def extract_clip(fp, pose, num_frames):
    """Return (T,33,4) array or None. T == num_frames (padded/truncated)."""
    cap = cv2.VideoCapture(fp)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
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
        return None
    arr = np.stack(frames, axis=0)                       # (t,33,4)
    # pad/truncate to exactly num_frames
    if arr.shape[0] < num_frames:
        pad = np.repeat(arr[-1:], num_frames - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:num_frames]


def collect_videos(dataset_root):
    """
    Walk dataset_root and collect (filepath, canonical_class_name) for all
    chosen exercises. Handles both flat and one-level-nested structures.

    Supported layouts:
      - <root>/<exercise>/*.mp4           (flat)
      - <root>/<subdataset>/<exercise>/*.mp4  (nested, like verified_data)
    """
    video_pairs = []  # list of (filepath, canon_cls)

    for entry in os.listdir(dataset_root):
        entry_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(entry_path):
            continue

        # Check if this entry itself is a chosen exercise folder (flat layout)
        if norm(entry) in CHOSEN_N:
            cls = norm(entry)
            for fname in os.listdir(entry_path):
                if os.path.splitext(fname)[1].lower() in VIDEO_EXT:
                    video_pairs.append((os.path.join(entry_path, fname), cls))
        else:
            # Try nested layout: entry is a sub-dataset folder
            for sub_entry in os.listdir(entry_path):
                sub_path = os.path.join(entry_path, sub_entry)
                if os.path.isdir(sub_path) and norm(sub_entry) in CHOSEN_N:
                    cls = norm(sub_entry)
                    for fname in os.listdir(sub_path):
                        if os.path.splitext(fname)[1].lower() in VIDEO_EXT:
                            video_pairs.append((os.path.join(sub_path, fname), cls))

    return video_pairs


def main(args):
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=args.complexity,
    )

    print(f"Scanning dataset root: {args.dataset}")
    video_pairs = collect_videos(args.dataset)
    if not video_pairs:
        raise SystemExit(f"No videos found under {args.dataset} for the chosen exercises.")

    # Build stable label map from classes present in the data
    classes_present = sorted({cls for _, cls in video_pairs})
    label_map   = {name: i for i, name in enumerate(classes_present)}
    id_to_label = {str(i): name for name, i in label_map.items()}
    print(f"Found {len(classes_present)} classes: {classes_present}")
    print(f"Total clips to process: {len(video_pairs)}\n")

    X, y, files_out = [], [], []
    per_class_kept = collections.Counter()
    per_class_drop = collections.Counter()

    for fp, cls in video_pairs:
        arr = extract_clip(fp, pose, args.num_frames)
        if arr is None:
            per_class_drop[cls] += 1
            continue
        mean_vis = float(arr[..., 3].mean())
        if mean_vis < args.vis_thresh:
            per_class_drop[cls] += 1
            continue
        X.append(arr)
        y.append(label_map[cls])
        files_out.append(os.path.basename(fp))
        per_class_kept[cls] += 1
        if sum(per_class_kept.values()) % 50 == 0:
            print(f"  processed {sum(per_class_kept.values())} clips so far...", flush=True)

    if not X:
        raise SystemExit("No clips passed the visibility threshold! Try lowering --vis-thresh.")

    X_arr   = np.stack(X).astype(np.float32)
    y_arr   = np.array(y, dtype=np.int64)
    f_arr   = np.array(files_out)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "landmarks.npz"),
                        X=X_arr, y=y_arr, files=f_arr)
    with open(os.path.join(args.out, "label_map.json"), "w") as fh:
        json.dump(label_map, fh, indent=2)
    with open(os.path.join(args.out, "id_to_label.json"), "w") as fh:
        json.dump(id_to_label, fh, indent=2)

    print("\n=== extraction summary ===")
    print(f"{'exercise':26}{'kept':>6}{'dropped':>9}")
    for cls in classes_present:
        print(f"{cls:26}{per_class_kept[cls]:>6}{per_class_drop[cls]:>9}")
    print(f"\nX shape: {X_arr.shape}  |  saved to {args.out}/landmarks.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    required=True,  help="path to dataset root (e.g. data/verified_data)")
    ap.add_argument("--out",        default="data",  help="output directory for landmarks.npz")
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--vis-thresh", type=float, default=0.6,
                    help="minimum mean visibility to keep a clip (default 0.6)")
    ap.add_argument("--complexity", type=int,  default=1, choices=[0, 1, 2],
                    help="BlazePose model complexity (0=fast, 1=default, 2=heavy)")
    main(ap.parse_args())
