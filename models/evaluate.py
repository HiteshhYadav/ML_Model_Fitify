"""
Fitify ML — Model Evaluation
Evaluates the trained exercise classifier on the test set.
"""

import os
import sys
import json

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate(
    checkpoint_dir: str = "checkpoints/best_model",
    data_dir: str = "data",
    batch_size: int = 4,
    num_frames: int = 16,
):
    """
    Evaluate the trained model on the validation/test set.
    """
    print("=" * 60)
    print("  Fitify ML — Model Evaluation")
    print("=" * 60)
    print()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Load model and processor
    print("📥 Loading model...")
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

    processor = VideoMAEImageProcessor.from_pretrained(checkpoint_dir)
    model = VideoMAEForVideoClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()

    # Load label map
    label_map_path = os.path.join(checkpoint_dir, "label_map.json")
    if not os.path.exists(label_map_path):
        label_map_path = os.path.join(data_dir, "label_map.json")

    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    id_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    print(f"   Classes: {num_classes}")
    print()

    # Load validation dataset
    from models.video_dataset import load_dataset_from_splits

    _, val_dataset, _ = load_dataset_from_splits(
        data_dir=data_dir,
        num_frames=num_frames,
        image_processor=processor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Run evaluation
    print("🔍 Running evaluation...")
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    # Per-class report
    target_names = [id_to_label[i] for i in range(num_classes)]
    report = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        output_dict=True,
    )
    report_text = classification_report(
        all_labels, all_preds,
        target_names=target_names,
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print results
    print()
    print("=" * 60)
    print("  Evaluation Results")
    print("=" * 60)
    print(f"  Accuracy:       {accuracy:.4f}")
    print(f"  F1 (macro):     {f1_macro:.4f}")
    print(f"  F1 (weighted):  {f1_weighted:.4f}")
    print()
    print("  Per-class Report:")
    print(report_text)

    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_report": report,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(all_labels),
        "num_classes": num_classes,
    }

    results_path = os.path.join("results", "evaluation_report.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  📊 Report saved to: {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Fitify ML model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/best_model")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=16)
    args = parser.parse_args()

    evaluate(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
    )
