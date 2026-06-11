"""
Fitify ML — Video Dataset
Custom PyTorch Dataset for loading workout exercise videos.
Compatible with VideoMAE (HuggingFace Transformers).
"""

import os
import json
import random
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class WorkoutVideoDataset(Dataset):
    """
    PyTorch Dataset for gym workout videos.
    Loads videos, samples frames uniformly, and applies VideoMAE transforms.
    """

    def __init__(
        self,
        samples: List[Dict],
        label_map: Dict[str, int],
        num_frames: int = 16,
        image_size: int = 224,
        image_processor=None,
        augment: bool = False,
    ):
        """
        Args:
            samples: List of {"video_path": str, "label": str}
            label_map: Dict mapping exercise_name -> label_id
            num_frames: Number of frames to sample per video (VideoMAE default: 16)
            image_size: Target frame size (VideoMAE default: 224)
            image_processor: HuggingFace VideoMAEImageProcessor instance
            augment: Whether to apply data augmentation
        """
        self.samples = samples
        self.label_map = label_map
        self.num_frames = num_frames
        self.image_size = image_size
        self.image_processor = image_processor
        self.augment = augment
        self.num_classes = len(label_map)

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """
        Uniformly sample frame indices from the video.
        If video has fewer frames than needed, repeat frames.
        """
        if total_frames >= self.num_frames:
            # Uniform sampling
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Repeat frames if video is too short
            indices = np.arange(total_frames)
            while len(indices) < self.num_frames:
                indices = np.concatenate([indices, np.arange(total_frames)])
            indices = indices[:self.num_frames]
            indices.sort()

        return indices.tolist()

    def _load_video_frames(self, video_path: str) -> Optional[List[np.ndarray]]:
        """
        Load and sample frames from a video file using OpenCV.

        Returns:
            List of RGB frames as numpy arrays, or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        indices = self._sample_frame_indices(total_frames)
        indices_set = set(indices)
        frames_dict = {}

        current_idx = 0
        max_idx = max(indices)

        while current_idx <= max_idx:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current_idx in indices_set:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame_rgb = cv2.resize(frame_rgb, (self.image_size, self.image_size))
                frames_dict[current_idx] = frame_rgb
            current_idx += 1

        cap.release()

        # Reconstruct list in correct order
        frames = []
        for idx in indices:
            if idx in frames_dict:
                frames.append(frames_dict[idx])
            else:
                # Use last good frame or black frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return frames

    def _augment_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply simple data augmentation to video frames.
        All frames get the same augmentation to maintain temporal consistency.
        """
        if random.random() < 0.5:
            # Horizontal flip
            frames = [np.fliplr(f).copy() for f in frames]

        if random.random() < 0.3:
            # Brightness adjustment
            factor = random.uniform(0.8, 1.2)
            frames = [np.clip(f * factor, 0, 255).astype(np.uint8) for f in frames]

        if random.random() < 0.3:
            # Random crop and resize (simulating slight zoom)
            h, w = frames[0].shape[:2]
            crop_size = int(min(h, w) * random.uniform(0.85, 1.0))
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            frames = [
                cv2.resize(f[top:top+crop_size, left:left+crop_size], (w, h))
                for f in frames
            ]

        return frames

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        video_path = sample["video_path"]
        label_name = sample["label"]
        label_id = self.label_map[label_name]

        # Load frames
        frames = self._load_video_frames(video_path)

        if frames is None:
            # Fallback: return black frames
            frames = [
                np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                for _ in range(self.num_frames)
            ]

        # Augment
        if self.augment:
            frames = self._augment_frames(frames)

        # Process with VideoMAE processor
        if self.image_processor is not None:
            inputs = self.image_processor(
                list(frames),
                return_tensors="pt"
            )
            pixel_values = inputs["pixel_values"].squeeze(0)  # Remove batch dim
        else:
            # Manual normalization fallback
            frames_tensor = torch.tensor(
                np.stack(frames), dtype=torch.float32
            ).permute(0, 3, 1, 2) / 255.0  # (T, C, H, W)
            pixel_values = frames_tensor

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


def load_dataset_from_splits(
    data_dir: str = "data",
    num_frames: int = 16,
    image_processor=None,
) -> Tuple[WorkoutVideoDataset, WorkoutVideoDataset, Dict[str, int]]:
    """
    Load train and validation datasets from preprocessed splits.

    Returns:
        (train_dataset, val_dataset, label_map)
    """
    # Load label map
    label_map_path = os.path.join(data_dir, "label_map.json")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    # Load splits
    splits_path = os.path.join(data_dir, "splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    train_dataset = WorkoutVideoDataset(
        samples=splits["train"],
        label_map=label_map,
        num_frames=num_frames,
        image_processor=image_processor,
        augment=True,
    )

    val_dataset = WorkoutVideoDataset(
        samples=splits["val"],
        label_map=label_map,
        num_frames=num_frames,
        image_processor=image_processor,
        augment=False,
    )

    print(f"📦 Loaded datasets: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"🏷️  Classes: {len(label_map)}")

    return train_dataset, val_dataset, label_map
