"""
Fitify ML — Models
Two small temporal classifiers over normalized landmark sequences (T, 99).
Both are tiny (~0.1-0.5M params) and export cleanly to ONNX / TFLite.
"""

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """Bi-GRU over the landmark sequence. Default model."""
    def __init__(self, in_dim=99, hidden=128, num_classes=12, layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, num_classes),
        )

    def forward(self, x):                    # x: (B,T,99)
        out, _ = self.gru(x)                 # (B,T,2H)
        feat = out.mean(dim=1)               # temporal mean-pool
        return self.head(feat)


class CNN1DClassifier(nn.Module):
    """1D-CNN over time. Alternative; even smaller, faster."""
    def __init__(self, in_dim=99, num_classes=12, dropout=0.3):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv1d(ci, co, 3, padding=1), nn.BatchNorm1d(co),
                nn.ReLU(), nn.MaxPool1d(2))
        self.net = nn.Sequential(block(in_dim, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x):                    # x: (B,T,99)
        x = x.transpose(1, 2)                # (B,99,T)
        return self.head(self.net(x))


def build_model(name, num_classes, in_dim=99):
    if name == "gru":
        return GRUClassifier(in_dim=in_dim, num_classes=num_classes)
    if name == "cnn":
        return CNN1DClassifier(in_dim=in_dim, num_classes=num_classes)
    raise ValueError(name)
