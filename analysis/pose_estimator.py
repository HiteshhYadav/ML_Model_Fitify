"""
Fitify ML — Pose Estimator
MediaPipe-based pose estimation with joint angle calculations.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe Pose landmark indices
LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
}


class PoseEstimator:
    """
    Extracts body pose landmarks and joint angles from video frames
    using MediaPipe Pose.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
        ema_alpha: float = 0.7,
    ):
        """
        Args:
            min_detection_confidence: MediaPipe detection confidence threshold
            min_tracking_confidence: MediaPipe tracking confidence threshold
            smooth_landmarks: Whether to apply EMA smoothing
            ema_alpha: EMA smoothing factor (higher = less smoothing)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # complexity=2 segfaults on Windows w/ mediapipe 0.10.14
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.smooth_landmarks = smooth_landmarks
        self.ema_alpha = ema_alpha
        self._prev_landmarks = None

    def _calculate_angle(
        self,
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        point_c: Tuple[float, float],
    ) -> float:
        """
        Calculate the angle at point_b formed by points a-b-c.

        Returns:
            Angle in degrees (0-180)
        """
        a = np.array(point_a)
        b = np.array(point_b)
        c = np.array(point_c)

        ba = a - b
        bc = c - b

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        cosine = np.clip(cosine, -1.0, 1.0)
        angle = math.degrees(math.acos(cosine))

        return angle

    def _apply_ema(self, landmarks: Dict) -> Dict:
        """Apply Exponential Moving Average smoothing to landmarks."""
        if self._prev_landmarks is None:
            self._prev_landmarks = landmarks.copy()
            return landmarks

        smoothed = {}
        for key, value in landmarks.items():
            if key in self._prev_landmarks:
                prev = self._prev_landmarks[key]
                smoothed[key] = {
                    "x": self.ema_alpha * value["x"] + (1 - self.ema_alpha) * prev["x"],
                    "y": self.ema_alpha * value["y"] + (1 - self.ema_alpha) * prev["y"],
                    "z": self.ema_alpha * value["z"] + (1 - self.ema_alpha) * prev["z"],
                    "visibility": value["visibility"],
                }
            else:
                smoothed[key] = value

        self._prev_landmarks = smoothed.copy()
        return smoothed

    def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extract pose landmarks from a single frame.

        Args:
            frame: BGR image (numpy array)

        Returns:
            Dict with landmark positions, or None if no pose detected
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks is None:
            return None

        h, w = frame.shape[:2]
        landmarks = {}

        for idx, name in LANDMARK_NAMES.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = {
                "x": lm.x * w,
                "y": lm.y * h,
                "z": lm.z,
                "visibility": lm.visibility,
            }

        if self.smooth_landmarks:
            landmarks = self._apply_ema(landmarks)

        return landmarks

    def calculate_joint_angles(self, landmarks: Dict) -> Dict[str, float]:
        """
        Calculate key joint angles from landmarks.

        Returns:
            Dict mapping angle_name -> angle_in_degrees
        """
        angles = {}

        def get_point(name):
            lm = landmarks.get(name)
            if lm and lm["visibility"] > 0.3:
                return (lm["x"], lm["y"])
            return None

        # Left elbow angle (shoulder-elbow-wrist)
        pts = [get_point("left_shoulder"), get_point("left_elbow"), get_point("left_wrist")]
        if all(pts):
            angles["left_elbow"] = self._calculate_angle(*pts)

        # Right elbow angle
        pts = [get_point("right_shoulder"), get_point("right_elbow"), get_point("right_wrist")]
        if all(pts):
            angles["right_elbow"] = self._calculate_angle(*pts)

        # Left shoulder angle (elbow-shoulder-hip)
        pts = [get_point("left_elbow"), get_point("left_shoulder"), get_point("left_hip")]
        if all(pts):
            angles["left_shoulder"] = self._calculate_angle(*pts)

        # Right shoulder angle
        pts = [get_point("right_elbow"), get_point("right_shoulder"), get_point("right_hip")]
        if all(pts):
            angles["right_shoulder"] = self._calculate_angle(*pts)

        # Left knee angle (hip-knee-ankle)
        pts = [get_point("left_hip"), get_point("left_knee"), get_point("left_ankle")]
        if all(pts):
            angles["left_knee"] = self._calculate_angle(*pts)

        # Right knee angle
        pts = [get_point("right_hip"), get_point("right_knee"), get_point("right_ankle")]
        if all(pts):
            angles["right_knee"] = self._calculate_angle(*pts)

        # Left hip angle (shoulder-hip-knee)
        pts = [get_point("left_shoulder"), get_point("left_hip"), get_point("left_knee")]
        if all(pts):
            angles["left_hip"] = self._calculate_angle(*pts)

        # Right hip angle
        pts = [get_point("right_shoulder"), get_point("right_hip"), get_point("right_knee")]
        if all(pts):
            angles["right_hip"] = self._calculate_angle(*pts)

        # Back angle (approximate: shoulder-hip vertical alignment)
        l_shoulder = get_point("left_shoulder")
        r_shoulder = get_point("right_shoulder")
        l_hip = get_point("left_hip")
        r_hip = get_point("right_hip")

        if all([l_shoulder, r_shoulder, l_hip, r_hip]):
            mid_shoulder = (
                (l_shoulder[0] + r_shoulder[0]) / 2,
                (l_shoulder[1] + r_shoulder[1]) / 2,
            )
            mid_hip = (
                (l_hip[0] + r_hip[0]) / 2,
                (l_hip[1] + r_hip[1]) / 2,
            )
            # Angle from vertical
            dx = mid_shoulder[0] - mid_hip[0]
            dy = mid_shoulder[1] - mid_hip[1]
            back_angle = abs(math.degrees(math.atan2(dx, -dy)))
            angles["back_lean"] = back_angle

        return angles

    def process_video(
        self,
        video_path: str,
        sample_rate: int = 3,
    ) -> List[Dict]:
        """
        Process an entire video and extract pose data for each sampled frame.

        Args:
            video_path: Path to the video file
            sample_rate: Process every Nth frame (default: every 3rd frame)

        Returns:
            List of frame data dicts with landmarks and angles
        """
        self._prev_landmarks = None  # Reset EMA

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        frame_data = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                landmarks = self.extract_landmarks(frame)
                timestamp = frame_idx / fps if fps > 0 else 0

                if landmarks:
                    angles = self.calculate_joint_angles(landmarks)
                    frame_data.append({
                        "frame_idx": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "landmarks": landmarks,
                        "angles": angles,
                        "has_pose": True,
                    })
                else:
                    frame_data.append({
                        "frame_idx": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "landmarks": None,
                        "angles": {},
                        "has_pose": False,
                    })

            frame_idx += 1

        cap.release()

        return frame_data

    def draw_pose(
        self,
        frame: np.ndarray,
        landmarks: Dict,
        angles: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Draw pose landmarks and optional angle annotations on a frame.

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw connections
        connections = [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]

        for start_name, end_name in connections:
            start = landmarks.get(start_name)
            end = landmarks.get(end_name)
            if start and end and start["visibility"] > 0.3 and end["visibility"] > 0.3:
                pt1 = (int(start["x"]), int(start["y"]))
                pt2 = (int(end["x"]), int(end["y"]))
                cv2.line(annotated, pt1, pt2, (0, 255, 128), 2)

        # Draw landmarks
        for name, lm in landmarks.items():
            if lm["visibility"] > 0.3:
                center = (int(lm["x"]), int(lm["y"]))
                cv2.circle(annotated, center, 5, (0, 200, 255), -1)
                cv2.circle(annotated, center, 7, (0, 100, 200), 2)

        # Draw angles
        if angles:
            angle_positions = {
                "left_elbow": "left_elbow",
                "right_elbow": "right_elbow",
                "left_knee": "left_knee",
                "right_knee": "right_knee",
                "left_hip": "left_hip",
                "right_hip": "right_hip",
            }
            for angle_name, landmark_name in angle_positions.items():
                if angle_name in angles and landmark_name in landmarks:
                    lm = landmarks[landmark_name]
                    if lm["visibility"] > 0.3:
                        pos = (int(lm["x"]) + 10, int(lm["y"]) - 10)
                        cv2.putText(
                            annotated,
                            f"{angles[angle_name]:.0f}°",
                            pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

        return annotated

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
