"""
Fitify ML — Stage 4: Training
Trains the pose-landmark classifier with class-weighted loss, early stopping
on validation accuracy, then a SINGLE held-out test evaluation producing a
classification report + confusion matrix. Exports best model to ONNX.

Run:
    python scripts/train.py --model gru --epochs 80
"""

import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.dataset import load_splits, class_weights
from models.nets import build_model


def set_seed(s=42):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    acc = (y == p).mean()
    return acc, y, p


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds, val_ds, test_ds, y_train, id_to_label = load_splits(args.data_dir, args.seed)
    num_classes = len(id_to_label)
    print(f"classes: {num_classes} | train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = build_model(args.model, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model} | params: {n_params/1e6:.3f}M")

    w = class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.ckpt, exist_ok=True)
    best_val, best_epoch, no_improve = 0.0, 0, 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item(); tot += yb.size(0)
        scheduler.step()
        train_acc = correct / tot
        val_acc, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch+1, "train_acc": train_acc,
                        "val_acc": float(val_acc), "loss": loss_sum/tot})
        print(f"epoch {epoch+1:3d}/{args.epochs} | loss {loss_sum/tot:.4f} "
              f"| train {train_acc:.3f} | val {val_acc:.3f}")

        if val_acc > best_val:
            best_val, best_epoch, no_improve = val_acc, epoch+1, 0
            torch.save(model.state_dict(), os.path.join(args.ckpt, "best.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stop @ epoch {epoch+1} (best val {best_val:.3f} @ {best_epoch})")
                break

    # ---- one-time held-out test eval ----
    model.load_state_dict(torch.load(os.path.join(args.ckpt, "best.pt")))
    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print(f"\n=== TEST accuracy: {test_acc:.4f} (best val {best_val:.4f} @ epoch {best_epoch}) ===")

    labels = [id_to_label[str(i)] for i in range(num_classes)]
    os.makedirs("results", exist_ok=True)

    # classification report + confusion matrix (sklearn if available, else manual)
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        rep = classification_report(y_true, y_pred, target_names=labels, digits=3)
        cm = confusion_matrix(y_true, y_pred)
    except Exception:
        rep = "(install scikit-learn for full report)"
        cm = np.zeros((num_classes, num_classes), int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
    print("\n" + (rep if isinstance(rep, str) else ""))
    print("confusion matrix (rows=true, cols=pred):")
    print("      " + " ".join(f"{i:>4}" for i in range(num_classes)))
    for i, row in enumerate(cm):
        print(f"{i:>2} {labels[i][:14]:14} " + " ".join(f"{v:>4}" for v in row))

    with open("results/test_report.txt", "w") as fh:
        fh.write(f"test_acc {test_acc:.4f}\nbest_val {best_val:.4f} @ {best_epoch}\n\n")
        fh.write(str(rep) + "\n\nlabels: " + ", ".join(labels) + "\n")
    with open("results/history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    np.save("results/confusion_matrix.npy", cm)

    # ---- ONNX export (portable: cloud server or convert to TFLite) ----
    model.eval().cpu()
    T = args.num_frames
    dummy = torch.randn(1, T, 99)
    onnx_path = os.path.join(args.ckpt, "fitify_pose.onnx")
    try:
        torch.onnx.export(model, dummy, onnx_path,
                          input_names=["landmarks"], output_names=["logits"],
                          dynamic_axes={"landmarks": {0: "batch", 1: "frames"},
                                        "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
        size_mb = os.path.getsize(onnx_path) / 1e6
        print(f"\nexported ONNX -> {onnx_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"\n[warn] ONNX export skipped: {e}")
        print("       (pip install onnx onnxscript to enable; best.pt is still saved)")
    print(f"total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--model", default="gru", choices=["gru", "cnn"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--num-frames", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
