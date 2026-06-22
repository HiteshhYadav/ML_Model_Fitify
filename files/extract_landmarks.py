"""
Fitify ML — Stage 1: Landmark Extraction
Runs BlazePose over every clip in the chosen exercise folders and saves a
fixed-length landmark sequence per clip. Clips below a visibility threshold
are dropped so garbage never reaches training.

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

# The 12 locked exercises. Folder names are matched case-insensitively /
# whitespace-trimmed against the dataset's actual folder names.
CHOSEN = [
    "squat", "deadlift", "romanian deadlift", "push-up", "pull up",
    "shoulder press", "hammer curl", "lateral raise", "plank",
    "leg raises", "russian twist", "hip thrust",
]
norm = lambda s: s.strip().lower().replace("_", " ")
CHOSEN_N = {norm(c) for c in CHOSEN}

# Folder-name aliases (dataset uses some odd casings/spellings)
ALIASES = {"pull up": "pull up", "pull-up": "pull up", "pull_up": "pull up"}

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
            lm = np.array([[p.x, p.y, p.z, p.visibility]
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


def main(args):
    pose = mp.solutions.pose.Pose(static_image_mode=True,
                                  model_complexity=args.complexity)

    # build label map from the folders actually present
    present = []
    for entry in sorted(os.listdir(args.dataset)):
        if norm(entry) in CHOSEN_N and os.path.isdir(os.path.join(args.dataset, entry)):
            present.append(entry)
    if not present:
        raise SystemExit(f"No chosen folders found under {args.dataset}")

    # canonical label = normalized name -> stable id
    canon = sorted({norm(e) for e in present})
    label_map = {name: i for i, name in enumerate(canon)}      # name -> id
    id_to_label = {str(i): name for name, i in label_map.items()}

    X, y, files = [], [], []
    per_class_kept = collections.Counter()
    per_class_drop = collections.Counter()

    for folder in present:
        cls = norm(folder)
        cid = label_map[cls]
        fdir = os.path.join(args.dataset, folder)
        vids = [f for f in os.listdir(fdir)
                if os.path.splitext(f)[1].lower() in VIDEO_EXT]
        print(f"[{cls}] {len(vids)} clips ...", flush=True)
        for f in vids:
            arr = extract_clip(os.path.join(fdir, f), pose, args.num_frames)
            if arr is None:
                per_class_drop[cls] += 1
                continue
            mean_vis = float(arr[..., 3].mean())
            if mean_vis < args.vis_thresh:
                per_class_drop[cls] += 1
                continue
            X.append(arr); y.append(cid); files.append(f)
            per_class_kept[cls] += 1

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    files = np.array(files)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "landmarks.npz"),
                        X=X, y=y, files=files)
    with open(os.path.join(args.out, "label_map.json"), "w") as fh:
        json.dump(label_map, fh, indent=2)
    with open(os.path.join(args.out, "id_to_label.json"), "w") as fh:
        json.dump(id_to_label, fh, indent=2)

    print("\n=== extraction summary ===")
    print(f"{'exercise':22}{'kept':>6}{'dropped':>9}")
    for cls in canon:
        print(f"{cls:22}{per_class_kept[cls]:>6}{per_class_drop[cls]:>9}")
    print(f"\nX shape: {X.shape}  |  saved to {args.out}/landmarks.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to kagglehub dataset root")
    ap.add_argument("--out", default="data")
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--vis-thresh", type=float, default=0.7)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    main(ap.parse_args())
