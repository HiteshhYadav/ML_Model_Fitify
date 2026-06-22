"""
Fitify ML — Demo Predictor
Randomly selects a workout video from the dataset or accepts a specific video,
predicts the exercise type using the trained VideoMAE model, and runs the biomechanical form analysis.
"""

import os
import random
import json
import argparse
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.analyzer import ExerciseAnalyzer


def get_random_video() -> str:
    """Randomly select a video from data/test/ or data/splits.json validation set."""
    # Option 1: Try data/test
    test_dir = os.path.join("data", "test")
    if os.path.exists(test_dir):
        videos = []
        for root, _, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    videos.append(os.path.join(root, file))
        if videos:
            print("🎲 Randomly selected from test set (data/test/)")
            return random.choice(videos)

    # Option 2: Try splits.json
    splits_path = os.path.join("data", "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path, "r") as f:
            splits = json.load(f)
        val_samples = splits.get("val", [])
        if val_samples:
            print("🎲 Randomly selected from validation splits (data/splits.json)")
            sample = random.choice(val_samples)
            return sample["video_path"]

    # Option 3: Fallback warning
    raise FileNotFoundError(
        "Could not find any videos. Make sure the dataset junctions are set up or place a video in the data/test directory."
    )


def main():
    parser = argparse.ArgumentParser(description="Demo prediction script for Fitify ML")
    parser.add_argument("--video", type=str, default=None, help="Path to a specific video file (optional)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model", help="Path to the trained model checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("  Fitify ML — End-to-End Predictor & Analyzer Demo")
    print("=" * 60)
    print()

    # Determine video path
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            sys.exit(1)
        print(f"📁 Using specified video: {video_path}")
    else:
        try:
            video_path = get_random_video()
            print(f"📁 Video: {video_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    print()
    print("🧠 Initializing Analyzer (loading model onto GPU/CPU)...")
    try:
        analyzer = ExerciseAnalyzer(checkpoint_dir=args.checkpoint)
    except Exception as e:
        print(f"❌ Failed to load analyzer: {e}")
        print("💡 Ensure you have a trained model saved in 'checkpoints/best_model'.")
        sys.exit(1)

    print()
    print("🔍 Running classification and biomechanical analysis...")
    report = analyzer.analyze(video_path)

    if "error" in report:
        print(f"❌ Analysis failed: {report['error']}")
        sys.exit(1)

    # Print Report Summary
    print()
    print("=" * 60)
    print("  ANALYSIS REPORT")
    print("=" * 60)
    print(f"  Predicted Exercise:  {report['exercise'].replace('_', ' ').title()}")
    print(f"  Confidence:          {report['confidence'] * 100:.2f}%")
    print(f"  Rep Count:           {report['rep_count']} reps")
    print(f"  Overall Form Score:  {report['overall_score']}/100")
    print(f"  Form Grade:          {report['grade']} ({report['grade_message']})")
    print("-" * 60)
    print("  Form Feedback Suggestions:")
    
    warnings = [fb for fb in report["form_feedback"] if fb["status"] in ("warning", "error")]
    goods = [fb for fb in report["form_feedback"] if fb["status"] == "good"]

    if warnings:
        for fb in warnings:
            status_emoji = "❌" if fb["status"] == "error" else "⚠️"
            print(f"  {status_emoji} [{fb['aspect']}] {fb['message']}")
    else:
        print("  ✅ Great form detected! No issues found.")

    if goods:
        print("-" * 60)
        print("  Aspects Performed Correctly:")
        for fb in goods:
            print(f"  ✅ [{fb['aspect']}] {fb['message']}")

    print("=" * 60)
    print(f"  Video Duration:      {report['video_info']['duration']} seconds")
    print(f"  Frames Analyzed:     {report['analysis_meta']['frames_analyzed']}")
    print(f"  Processing Time:     {report['analysis_meta']['processing_time_seconds']} seconds")
    print("=" * 60)


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    main()
