"""
Fitify ML — Model Training
Fine-tunes VideoMAE for gym exercise video classification.
"""

import os
import sys
import json
import time
from datetime import timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train(
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
    num_epochs: int = 15,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    num_frames: int = 16,
    weight_decay: float = 0.05,
    warmup_epochs: int = 2,
    patience: int = 5,
    use_fp16: bool = True,
):
    """
    Fine-tune VideoMAE for exercise classification.
    """
    print("=" * 60)
    print("  Fitify ML — Model Training")
    print("=" * 60)
    print()

    # ---- Device setup ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️  Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  No GPU found. Training on CPU (this will be slow).")
        use_fp16 = False  # FP16 needs GPU

    print()

    # ---- Load processor and model ----
    print("📥 Loading VideoMAE model and processor...")
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    # Load label map for num_classes
    label_map_path = os.path.join(data_dir, "label_map.json")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    id_to_label_path = os.path.join(data_dir, "id_to_label.json")
    with open(id_to_label_path, "r") as f:
        id_to_label = json.load(f)

    num_classes = len(label_map)
    print(f"   Model: MCG-NJU/videomae-base")
    print(f"   Classes: {num_classes}")
    print()

    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base",
        num_labels=num_classes,
        label2id=label_map,
        id2label=id_to_label,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # ---- Load datasets ----
    print("📦 Loading datasets...")
    from models.video_dataset import load_dataset_from_splits

    train_dataset, val_dataset, _ = load_dataset_from_splits(
        data_dir=data_dir,
        num_frames=num_frames,
        image_processor=processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print()

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=learning_rate * 0.01,
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    # ---- Training state ----
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    training_history = []

    print(f"🏋️ Training config:")
    print(f"   Epochs:        {num_epochs}")
    print(f"   Batch size:    {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Weight decay:  {weight_decay}")
    print(f"   FP16:          {use_fp16}")
    print(f"   Patience:      {patience}")
    print()
    print("-" * 60)

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            leave=False,
        )

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_fp16:
                with torch.amp.autocast("cuda"):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{train_correct/train_total:.3f}"
            })

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                leave=False,
            ):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                if use_fp16:
                    with torch.amp.autocast("cuda"):
                        outputs = model(pixel_values=pixel_values, labels=labels)
                else:
                    outputs = model(pixel_values=pixel_values, labels=labels)

                val_loss += outputs.loss.item() * labels.size(0)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update scheduler (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # Log
        print(
            f"  Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {timedelta(seconds=int(epoch_time))}"
        )

        training_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "time_seconds": epoch_time,
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            best_model_dir = os.path.join(checkpoint_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            processor.save_pretrained(best_model_dir)

            # Save label maps alongside model
            with open(os.path.join(best_model_dir, "label_map.json"), "w") as f:
                json.dump(label_map, f, indent=2)
            with open(os.path.join(best_model_dir, "id_to_label.json"), "w") as f:
                json.dump(id_to_label, f, indent=2)

            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"  Total Time:        {timedelta(seconds=int(total_time))}")
    print(f"  Best Model:        {os.path.join(checkpoint_dir, 'best_model')}")
    print("=" * 60)

    # Save training history
    os.makedirs("results", exist_ok=True)
    history_path = os.path.join("results", "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"\n  📊 Training history saved to: {history_path}")

    return best_val_acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Fitify ML exercise classifier")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_frames=args.num_frames,
        patience=args.patience,
        use_fp16=not args.no_fp16,
    )
