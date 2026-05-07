"""
CRNN for Medical Prescription OCR
Architecture: ResNet18 CNN encoder → FC projection → BiLSTM → CTC decode
"""
import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import OCR_HIDDEN_SIZE, OCR_CHARSET, OCR_BLANK_IDX

NUM_CLASSES = len(OCR_CHARSET) + 1  # +1 for CTC blank


class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, hidden_size=OCR_HIDDEN_SIZE):
        super().__init__()

        # ── CNN Encoder: ResNet18 backbone (pretrained) ──────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove avgpool and fc, keep only feature extractor layers
        self.cnn = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # ── Projection: CNN features → sequence ──────────────────────────
        # Input image: (B, 3, 32, 128) → after CNN: (B, 512, 1, 4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse height
        self.projection = nn.Sequential(
            nn.Linear(512, hidden_size * 2),
            nn.Dropout(0.3),
        )

        # ── BiLSTM Sequence Decoder ───────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.3,
        )

        # ── Output: predict character at each timestep ───────────────────
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.cnn(x)                    # (B, 512, H', W')
        features = self.adaptive_pool(features)   # (B, 512, 1, W')
        features = features.squeeze(2)            # (B, 512, W')
        features = features.permute(2, 0, 1)      # (W', B, 512) — time-first
        features = self.projection(features)       # (W', B, hidden*2)
        lstm_out, _ = self.lstm(features)          # (W', B, hidden*2)
        logits = self.fc(lstm_out)                 # (W', B, num_classes)
        return logits
