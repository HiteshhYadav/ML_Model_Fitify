# Fitify ML — Pose-Landmark Exercise Classifier

Path B pipeline: BlazePose landmarks → normalized sequences → small temporal
classifier (GRU / 1D-CNN). Tiny (~0.4–2 MB), portable (ONNX → TFLite/server),
and the same landmark stream feeds your angle-based form-feedback module.

## The 12 locked exercises
squat, deadlift, romanian deadlift, push-up, pull up, shoulder press,
hammer curl, lateral raise, plank, leg raises, russian twist, hip thrust

(Dropped bench press, incline bench, tricep dips, barbell biceps curl —
horizontal/occluded → poor landmark visibility, confirmed by the scorer.)

## Pipeline

```
scripts/extract_landmarks.py   Stage 1: BlazePose over clips -> data/landmarks.npz
models/dataset.py              Normalization (hip-center + torso-scale) + split
models/nets.py                 GRUClassifier (default) / CNN1DClassifier
scripts/train.py               Stage 4: weighted loss, early stop, test + CM, ONNX
```

## Run

```bash
pip install mediapipe opencv-python torch numpy scikit-learn onnx onnxscript

# 1. extract landmarks from the kagglehub dataset path
python scripts/extract_landmarks.py --dataset "<KAGGLEHUB_PATH>" --num-frames 32 --vis-thresh 0.7

# 2. train (GRU default). idc-about-time -> raise epochs; early stopping protects you
python scripts/train.py --model gru --epochs 120 --patience 20

# featherweight alternative
python scripts/train.py --model cnn --epochs 120 --patience 20
```

Outputs: `checkpoints/best.pt`, `checkpoints/fitify_pose.onnx`,
`results/test_report.txt`, `results/confusion_matrix.npy`, `results/history.json`.

## Key design notes
- **Normalization is the load-bearing step.** Each frame is re-origined to the
  hip-center and scaled by torso length → invariant to body size, camera
  distance, and frame position. Done in `models/dataset.normalize_sequence`.
- **Class imbalance** (67→195 clips) handled by inverse-frequency weighted
  cross-entropy + label smoothing.
- **Held-out test set** is split once (seeded) and only touched after training.
- **Offline / upload-based**: clips are processed as whole sequences, not
  streamed — matches the "upload a video → get analysis" UX.

## Deploy
- ONNX runs as-is on a server (onnxruntime).
- For on-device: `onnx2tf` or the TF→TFLite path; model is already small enough
  that INT8 quantization gets you well under 1 MB.

## Report baseline (optional, recommended)
Train a small frame-CNN or VideoMAE-base once on the same split purely to get
one comparison row: "pose-landmark model matched the video baseline at ~1/70th
the size." One paragraph of academic credit, not a maintained codepath.
