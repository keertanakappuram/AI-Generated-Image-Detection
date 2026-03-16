"""
model.py — Model Loading and Inference
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from src.schemas import PredictResponse

CLASS_NAMES = ["FAKE", "REAL"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def build_resnet():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    return model

class ImageDetector:
    def __init__(self):
        self.model     = None
        self.is_loaded = False
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, path: str):
        self.model = build_resnet().to(self.device)
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.model.eval()
        self.is_loaded = True

    def predict(self, image: Image.Image) -> PredictResponse:
        x = TRANSFORM(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return PredictResponse(
            label=CLASS_NAMES[pred_idx],
            confidence=round(float(probs[pred_idx]), 4),
            probabilities={
                CLASS_NAMES[i]: round(float(probs[i]), 4)
                for i in range(len(CLASS_NAMES))
            }
        )
