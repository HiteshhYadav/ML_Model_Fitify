"""
Fitify ML — Exercise Analyzer
Orchestrates the full analysis pipeline: classification + pose + form rules.
"""

import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.pose_estimator import PoseEstimator
from analysis.form_rules import get_form_feedback, FormFeedback


class ExerciseAnalyzer:
    """
    Full exercise analysis pipeline:
    1. Classify the exercise type (VideoMAE)
    2. Extract pose landmarks (MediaPipe)
    3. Calculate joint angles
    4. Apply form rules
    5. Count reps
    6. Generate analysis report
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/best_model"):
        """
        Args:
            checkpoint_dir: Path to the trained model checkpoint
        """
        from models.predict import ExerciseClassifier

        self.classifier = ExerciseClassifier(checkpoint_dir)
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
            ema_alpha=0.7,
        )

    def _count_reps(self, angles_list: List[Dict], exercise: str) -> int:
        """
        Count exercise repetitions by detecting oscillation in the primary
        joint angle for the given exercise.
        """
        # Determine primary tracking angle
        primary_angles_map = {
            "squat": ["left_knee", "right_knee"],
            "deadlift": ["left_hip", "right_hip"],
            "romanian_deadlift": ["left_hip", "right_hip"],
            "barbell_biceps_curl": ["left_elbow", "right_elbow"],
            "bicep_curl": ["left_elbow", "right_elbow"],
            "hammer_curl": ["left_elbow", "right_elbow"],
            "push_up": ["left_elbow", "right_elbow"],
            "shoulder_press": ["left_shoulder", "right_shoulder"],
            "lateral_raise": ["left_shoulder", "right_shoulder"],
            "lunge": ["left_knee", "right_knee"],
            "bench_press": ["left_elbow", "right_elbow"],
            "lat_pulldown": ["left_elbow", "right_elbow"],
            "pull_up": ["left_elbow", "right_elbow"],
            "tricep_dips": ["left_elbow", "right_elbow"],
            "tricep_pushdown": ["left_elbow", "right_elbow"],
            "leg_extension": ["left_knee", "right_knee"],
        }

        exercise_key = exercise.lower().replace(" ", "_")
        tracking_keys = primary_angles_map.get(exercise_key, ["left_elbow", "right_elbow"])

        # Collect angle values
        values = []
        for angles in angles_list:
            vals = [angles.get(k) for k in tracking_keys if k in angles]
            if vals:
                values.append(sum(vals) / len(vals))

        if len(values) < 6:
            return 0

        # Smooth the signal
        kernel_size = min(5, len(values) // 3)
        if kernel_size >= 3:
            kernel = np.ones(kernel_size) / kernel_size
            values = np.convolve(values, kernel, mode='valid').tolist()

        if len(values) < 4:
            return 0

        # Find peaks and troughs
        avg_val = sum(values) / len(values)
        threshold = (max(values) - min(values)) * 0.3

        if threshold < 10:
            return 0

        # Count transitions from below-average to above-average
        reps = 0
        was_below = values[0] < avg_val

        for v in values[1:]:
            is_below = v < avg_val
            if was_below and not is_below:
                reps += 1
            was_below = is_below

        return max(0, reps)

    def _calculate_form_timeline(
        self,
        angles_list: List[Dict],
        landmarks_list: List[Dict],
        exercise: str,
        timestamps: List[float],
    ) -> List[Dict]:
        """
        Generate a form quality timeline showing score at intervals.
        """
        if len(angles_list) < 3:
            return []

        timeline = []
        chunk_size = max(1, len(angles_list) // 10)

        for i in range(0, len(angles_list), chunk_size):
            chunk_angles = angles_list[i:i + chunk_size]
            chunk_landmarks = landmarks_list[i:i + chunk_size]

            if not chunk_angles:
                continue

            chunk_feedback = get_form_feedback(exercise, chunk_angles, chunk_landmarks)
            chunk_score = self._calculate_overall_score(chunk_feedback)

            ts = timestamps[i] if i < len(timestamps) else 0
            timeline.append({
                "timestamp": round(ts, 2),
                "score": round(chunk_score, 1),
                "frame_index": i,
            })

        return timeline

    def _calculate_overall_score(self, feedback_list: List[FormFeedback]) -> float:
        """Calculate weighted overall form score from feedback items."""
        if not feedback_list:
            return 75.0

        total_weight = 0
        weighted_score = 0

        for fb in feedback_list:
            weight = fb.severity
            weighted_score += fb.score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 75.0

    def _get_video_info(self, video_path: str) -> Dict:
        """Extract basic video metadata."""
        cap = cv2.VideoCapture(video_path)
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": 0,
        }
        if info["fps"] > 0:
            info["duration"] = round(info["total_frames"] / info["fps"], 2)
        cap.release()
        return info

    def analyze(self, video_path: str) -> Dict:
        """
        Run the full analysis pipeline on a video.

        Args:
            video_path: Path to the video file

        Returns:
            Comprehensive analysis report dict
        """
        start_time = time.time()

        if not os.path.exists(video_path):
            return {"error": f"Video file not found: {video_path}"}

        # 1. Video info
        video_info = self._get_video_info(video_path)

        # 2. Classify exercise
        classification = self.classifier.predict(video_path)

        exercise_name = classification["exercise"]
        exercise_confidence = classification["confidence"]

        # 3. Extract pose data
        frame_data = self.pose_estimator.process_video(video_path, sample_rate=3)

        # Filter frames with poses
        pose_frames = [f for f in frame_data if f["has_pose"]]
        no_pose_frames = len(frame_data) - len(pose_frames)

        if not pose_frames:
            return {
                "exercise": exercise_name,
                "confidence": exercise_confidence,
                "classification": classification,
                "error": "No human pose detected in the video. Please ensure a person is visible.",
                "video_info": video_info,
            }

        angles_list = [f["angles"] for f in pose_frames]
        landmarks_list = [f["landmarks"] for f in pose_frames]
        timestamps = [f["timestamp"] for f in pose_frames]

        # 4. Form analysis
        form_feedback = get_form_feedback(exercise_name, angles_list, landmarks_list)

        # 5. Count reps
        rep_count = self._count_reps(angles_list, exercise_name)

        # 6. Overall score
        overall_score = self._calculate_overall_score(form_feedback)

        # 7. Form timeline
        timeline = self._calculate_form_timeline(
            angles_list, landmarks_list, exercise_name, timestamps
        )

        # 8. Summary statistics
        good_count = sum(1 for f in form_feedback if f.status == "good")
        warning_count = sum(1 for f in form_feedback if f.status == "warning")
        error_count = sum(1 for f in form_feedback if f.status == "error")

        # Grade
        if overall_score >= 85:
            grade = "A"
            grade_message = "Excellent form! Keep it up!"
        elif overall_score >= 70:
            grade = "B"
            grade_message = "Good form with minor areas for improvement."
        elif overall_score >= 55:
            grade = "C"
            grade_message = "Decent form but several areas need attention."
        elif overall_score >= 40:
            grade = "D"
            grade_message = "Poor form. Please review the suggestions carefully."
        else:
            grade = "F"
            grade_message = "Significant form issues. Consider working with a trainer."

        processing_time = time.time() - start_time

        report = {
            "exercise": exercise_name,
            "confidence": round(exercise_confidence, 4),
            "classification": classification,
            "overall_score": round(overall_score, 1),
            "grade": grade,
            "grade_message": grade_message,
            "rep_count": rep_count,
            "form_feedback": [fb.to_dict() for fb in form_feedback],
            "summary": {
                "good_aspects": good_count,
                "warnings": warning_count,
                "errors": error_count,
                "total_aspects": len(form_feedback),
            },
            "timeline": timeline,
            "video_info": video_info,
            "analysis_meta": {
                "frames_analyzed": len(pose_frames),
                "frames_no_pose": no_pose_frames,
                "pose_detection_rate": round(len(pose_frames) / max(1, len(frame_data)) * 100, 1),
                "processing_time_seconds": round(processing_time, 2),
            },
        }

        return report

    def close(self):
        """Release resources."""
        self.pose_estimator.close()
