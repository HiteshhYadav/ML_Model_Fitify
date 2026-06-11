"""
Fitify ML — Model Prediction
Inference function for exercise classification from video.
"""

import os
import json
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


class ExerciseClassifier:
    """
    Classifies gym exercises from video using a fine-tuned VideoMAE model.
    Falls back to a simpler approach if no trained checkpoint is available.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/best_model"):
        """
        Args:
            checkpoint_dir: Path to the saved model checkpoint
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.label_map = None
        self.id_to_label = None
        self.num_frames = 16
        self.image_size = 224
        self._loaded = False

        self._load_model()

    def _load_model(self):
        """Load the trained model checkpoint."""
        if not os.path.exists(self.checkpoint_dir):
            print(f"⚠️  No checkpoint found at {self.checkpoint_dir}")
            print("   Using fallback classification (less accurate)")
            self._setup_fallback()
            return

        try:
            from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

            self.processor = VideoMAEImageProcessor.from_pretrained(self.checkpoint_dir)
            self.model = VideoMAEForVideoClassification.from_pretrained(self.checkpoint_dir)
            self.model.to(self.device)
            self.model.eval()

            # Load label maps
            label_map_path = os.path.join(self.checkpoint_dir, "label_map.json")
            with open(label_map_path, "r") as f:
                self.label_map = json.load(f)

            id_to_label_path = os.path.join(self.checkpoint_dir, "id_to_label.json")
            with open(id_to_label_path, "r") as f:
                self.id_to_label = json.load(f)

            self._loaded = True
            print(f"✅ Model loaded from {self.checkpoint_dir}")
            print(f"   Classes: {len(self.label_map)}")

        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            self._setup_fallback()

    def _setup_fallback(self):
        """
        Setup a fallback classification using common exercise names.
        This is used when no trained model is available.
        """
        self._loaded = False
        self.id_to_label = {
            "0": "barbell_biceps_curl",
            "1": "bench_press",
            "2": "chest_fly_machine",
            "3": "deadlift",
            "4": "decline_bench_press",
            "5": "hammer_curl",
            "6": "hip_thrust",
            "7": "incline_bench_press",
            "8": "lat_pulldown",
            "9": "lateral_raise",
            "10": "leg_extension",
            "11": "leg_raises",
            "12": "pec_deck_fly",
            "13": "plank",
            "14": "pull_up",
            "15": "push_up",
            "16": "romanian_deadlift",
            "17": "russian_twist",
            "18": "shoulder_press",
            "19": "squat",
            "20": "t_bar_row",
            "21": "tricep_dips",
            "22": "tricep_pushdown",
        }
        self.label_map = {v: int(k) for k, v in self.id_to_label.items()}

    def _sample_frames(self, video_path: str) -> list:
        """Sample frames uniformly from a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        # Uniform sampling
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.arange(total_frames)
            while len(indices) < self.num_frames:
                indices = np.concatenate([indices, np.arange(total_frames)])
            indices = indices[:self.num_frames]
            indices.sort()

        indices_set = set(indices.tolist())
        frames_dict = {}

        current_idx = 0
        max_idx = int(max(indices))

        while current_idx <= max_idx:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if current_idx in indices_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (self.image_size, self.image_size))
                frames_dict[current_idx] = frame_rgb
            current_idx += 1

        cap.release()

        frames = []
        for idx in indices:
            if idx in frames_dict:
                frames.append(frames_dict[idx])
            else:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return frames

    def predict(self, video_path: str) -> Dict:
        """
        Predict the exercise type from a video.

        Args:
            video_path: Path to the video file

        Returns:
            Dict with:
                - exercise: str — predicted exercise name
                - confidence: float — prediction confidence (0-1)
                - top_3: List[Dict] — top 3 predictions with names and confidences
                - model_used: str — "videomae" or "fallback"
        """
        frames = self._sample_frames(video_path)

        if self._loaded and self.model is not None:
            return self._predict_videomae(frames)
        else:
            return self._predict_fallback(video_path, frames)

    def _predict_videomae(self, frames: list) -> Dict:
        """Run inference using the fine-tuned VideoMAE model."""
        inputs = self.processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Top predictions
        top_k = min(3, len(probs))
        top_probs, top_indices = torch.topk(probs, top_k)

        top_3 = []
        for prob, idx in zip(top_probs, top_indices):
            label = self.id_to_label[str(idx.item())]
            top_3.append({
                "exercise": label,
                "confidence": round(prob.item(), 4),
            })

        return {
            "exercise": top_3[0]["exercise"],
            "confidence": top_3[0]["confidence"],
            "top_3": top_3,
            "model_used": "videomae",
        }

    def _predict_fallback(self, video_path: str, frames: list) -> Dict:
        """
        Fallback prediction based on video filename heuristics.
        Used when no trained model is available.
        """
        filename = os.path.basename(video_path).lower()
        best_match = "unknown_exercise"
        best_score = 0.0

        for exercise_name in self.label_map.keys():
            # Simple keyword matching
            keywords = exercise_name.replace("_", " ").split()
            matches = sum(1 for kw in keywords if kw in filename)
            score = matches / len(keywords) if keywords else 0

            if score > best_score:
                best_score = score
                best_match = exercise_name

        # If no match from filename, default to "unknown"
        if best_score == 0:
            best_match = "squat"  # Default guess
            best_score = 0.3

        return {
            "exercise": best_match,
            "confidence": round(min(best_score, 0.99), 4),
            "top_3": [
                {"exercise": best_match, "confidence": round(min(best_score, 0.99), 4)},
            ],
            "model_used": "fallback",
        }

    def get_supported_exercises(self) -> list:
        """Return list of all supported exercise names."""
        return sorted(self.label_map.keys()) if self.label_map else []
