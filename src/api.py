"""
api.py — FastAPI Serving Layer for AI-Generated Image Detection
===============================================================
Exposes the trained ResNet-18 model as a REST API.

Endpoints:
    POST /predict  — upload an image, get real/fake prediction
    GET  /health   — health check
    GET  /info     — model info

Usage:
    uvicorn src.api:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.schemas import PredictResponse
from src.model import ImageDetector
import time
import logging
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI-Generated Image Detector API",
    description="Detects whether an image is real or AI-generated using fine-tuned ResNet-18",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

detector = ImageDetector()

@app.on_event("startup")
async def startup():
    logger.info("Loading image detection model...")
    detector.load("models/resnet18_cifake.pt")
    logger.info("Model loaded ✅")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector.is_loaded}

@app.get("/info")
def info():
    return {
        "model": "ResNet-18 fine-tuned on CIFAKE",
        "classes": ["FAKE", "REAL"],
        "dataset": "CIFAKE — 120K images (60K real CIFAR-10 + 60K Stable Diffusion)",
        "accuracy": "98.42%",
        "auc_roc": "0.9986",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    start = time.time()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = detector.predict(image)
        result.latency_ms = round((time.time() - start) * 1000, 2)
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
