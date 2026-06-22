# pip install mediapipe opencv-python numpy   (run once in terminal if not already)
import os, csv, cv2, numpy as np, collections
import mediapipe as mp

path = r"C:\Users\yadav\.cache\kagglehub\datasets\philosopher0808\gym-workoutexercises-video\versions\1"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SAMPLE_FRAMES = 15
VIS_THRESH = 0.7

# only score these folders
KEEP = {
    "barbell biceps curl", "plank", "bench press", "push-up", "squat",
    "deadlift", "incline bench press", "hammer curl", "leg raises", "pull Up",
    "lateral raise", "hip thrust", "romanian deadlift", "russian twist",
    "tricep dips", "shoulder press",
}
norm = lambda s: s.strip().lower()
KEEP_N = {norm(k) for k in KEEP}

mp_pose = mp.solutions.pose
rows = []

def score_clip(fp):
    cap = cv2.VideoCapture(fp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release(); return None
    idxs = np.linspace(0, total - 1, min(SAMPLE_FRAMES, total)).astype(int)
    vis = []
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok: continue
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                vis.append(np.mean([lm.visibility for lm in res.pose_landmarks.landmark]))
    cap.release()
    if not vis: return 0.0, 0.0
    return float(np.mean(vis)), len(vis) / len(idxs)

for root, _, files in os.walk(path):
    cls = os.path.basename(root)
    if norm(cls) not in KEEP_N:
        continue
    for f in files:
        if os.path.splitext(f)[1].lower() in VIDEO_EXT:
            r = score_clip(os.path.join(root, f))
            if r is None: continue
            mv, det = r
            rows.append((cls, f, round(mv, 3), round(det, 2)))

agg = collections.defaultdict(list)
for cls, f, mv, det in rows:
    agg[cls].append(mv)

print(f"{'exercise':24} {'n':>4} {'mean_vis':>9} {'%keep>=0.7':>11}")
for cls in sorted(agg, key=lambda c: -np.mean(agg[c])):
    arr = np.array(agg[cls])
    print(f"{cls:24} {len(arr):>4} {arr.mean():>9.3f} {(arr>=VIS_THRESH).mean()*100:>10.0f}%")

with open("clip_scores.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["exercise","file","mean_vis","detect_rate"]); w.writerows(rows)
print("\nWrote clip_scores.csv |", len(rows), "clips scored")
