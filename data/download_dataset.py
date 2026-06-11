"""
Fitify ML — Dataset Downloader
Downloads the Gym Workout/Exercises Video dataset from Kaggle.
"""

import os
import sys
import shutil


def download_dataset(target_dir: str = "data") -> str:
    """
    Download the gym workout/exercises video dataset from Kaggle.

    Args:
        target_dir: Directory to save the dataset

    Returns:
        Path to the downloaded dataset
    """
    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    os.makedirs(target_dir, exist_ok=True)

    print("=" * 60)
    print("  Fitify ML — Dataset Downloader")
    print("=" * 60)
    print()
    print("📦 Downloading: philosopher0808/gym-workoutexercises-video")
    print("⚠️  Dataset size: ~10 GB. This may take a while...")
    print()

    try:
        path = kagglehub.dataset_download("philosopher0808/gym-workoutexercises-video")
        print(f"✅ Dataset downloaded to: {path}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("💡 Manual download instructions:")
        print("   1. Go to: https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video")
        print("   2. Click 'Download' and extract to the 'data/' folder")
        print("   3. Ensure this structure exists:")
        print("      data/")
        print("        verified_data/")
        print("          data_btc_10s/")
        print("          data_crawl_10s/")
        print("        test/")
        sys.exit(1)

    # Copy verified_data and test to our data directory
    for subdir in ["verified_data", "test"]:
        src = os.path.join(path, subdir)
        dst = os.path.join(target_dir, subdir)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"📂 Copying {subdir} to {dst}...")
            shutil.copytree(src, dst)

    print()
    print("✅ Dataset ready!")
    print(f"   Location: {os.path.abspath(target_dir)}")

    return target_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Fitify ML dataset")
    parser.add_argument(
        "--target-dir", type=str, default="data",
        help="Directory to save the dataset (default: data)"
    )
    args = parser.parse_args()

    download_dataset(args.target_dir)
