"""
Fitify ML — Dataset & Normalization
Loads extracted landmarks, applies size/position-invariant normalization,
builds per-frame feature vectors, and provides a seeded stratified split.

Normalization (the part that matters most):
  1. re-origin every joint to the hip-center (midpoint of L/R hip)
  2. scale by torso length (hip-center -> shoulder-center distance)
This makes features invariant to where the person is in the frame, how far
from the camera, and their absolute body size.

MediaPipe Pose landmark indices used:
  11 L-shoulder, 12 R-shoulder, 23 L-hip, 24 R-hip
"""

import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
EPS = 1e-6


def normalize_sequence(seq):
    """
    seq: (T, 33, 4) raw landmarks (x,y,z,visibility)
    returns: (T, 33*3) normalized xyz, hip-centered & torso-scaled, per frame.
    Visibility is dropped from features but used for nothing here (already
    filtered upstream); keep xyz only to stay compact.
    """
    xyz = seq[..., :3].copy()                              # (T,33,3)
    hip_center = (xyz[:, L_HIP] + xyz[:, R_HIP]) / 2.0      # (T,3)
    sh_center = (xyz[:, L_SH] + xyz[:, R_SH]) / 2.0         # (T,3)
    torso = np.linalg.norm(sh_center - hip_center, axis=-1, keepdims=True)  # (T,1)
    torso = np.maximum(torso, EPS)
    xyz = xyz - hip_center[:, None, :]                     # re-origin
    xyz = xyz / torso[:, None, :]                          # scale
    return xyz.reshape(xyz.shape[0], -1).astype(np.float32)  # (T, 99)


class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        feats = normalize_sequence(self.X[i])             # (T,99)
        return torch.from_numpy(feats), self.y[i]


def stratified_split(y, seed=42, val=0.15, test=0.15):
    rng = np.random.default_rng(seed)
    idx_train, idx_val, idx_test = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test)))
        n_val = max(1, int(round(n * val)))
        idx_test += idx[:n_test].tolist()
        idx_val += idx[n_test:n_test + n_val].tolist()
        idx_train += idx[n_test + n_val:].tolist()
    rng.shuffle(idx_train); rng.shuffle(idx_val); rng.shuffle(idx_test)
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def load_splits(data_dir="data", seed=42):
    d = np.load(os.path.join(data_dir, "landmarks.npz"), allow_pickle=True)
    X, y = d["X"], d["y"]
    tr, va, te = stratified_split(y, seed=seed)
    with open(os.path.join(data_dir, "id_to_label.json")) as fh:
        id_to_label = json.load(fh)
    ds = lambda ix: LandmarkDataset(X[ix], y[ix])
    return ds(tr), ds(va), ds(te), y[tr], id_to_label


def class_weights(y_train, num_classes):
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)             # inverse-frequency
    return torch.tensor(w, dtype=torch.float32)
