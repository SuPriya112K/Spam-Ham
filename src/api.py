"""
api.py

FastAPI application serving the spam classifier.

Endpoints:
    GET  /health   - health check (used by deployment platforms & monitoring)
    POST /predict  - takes text, returns spam/ham prediction + confidence
    GET  /metrics  - basic operational stats (prediction counts, latency, uptime)

Run locally with:
    uvicorn api:app --reload
"""

import os
import time
import logging
import datetime
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocessing import clean_text

# ---- Logging setup ----
# Every prediction gets logged so we can analyze traffic patterns and model behavior over time.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("spam_classifier_api")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "spam_classifier_pipeline.pkl")

# Global variable to hold the loaded model (loaded once at startup, not per-request)
model_pipeline = None

# ---- Simple in-memory monitoring ----
# In a real multi-server production system, you'd use something like Prometheus
# instead of a plain Python dict (in-memory stats reset if the server restarts,
# and don't work across multiple server instances). For a single-instance
# resume project, this is a lightweight, honest way to demonstrate the concept.
metrics_store = {
    "total_predictions": 0,
    "spam_count": 0,
    "ham_count": 0,
    "total_latency_ms": 0.0,
    "start_time": datetime.datetime.utcnow()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the API starts up (before it accepts any requests) and once on shutdown.
    We use this to load the model ONE TIME, rather than reloading it on every single
    prediction request (which would be extremely slow).
    """
    global model_pipeline
    logger.info("Loading model pipeline...")
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Spam Classifier API",
    description="A machine learning API that classifies text as Spam or Ham (not spam).",
    version="1.0.0",
    lifespan=lifespan
)


# ---- Request / Response schemas ----
# Pydantic models define exactly what shape of data this API accepts and returns.
# FastAPI uses these to auto-validate incoming requests AND generate interactive docs.

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The email or message text to classify"
    )


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    spam_probability: float
    ham_probability: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    total_predictions: int
    spam_count: int
    ham_count: int
    spam_ratio: float
    average_latency_ms: float
    uptime_seconds: float


# ---- Endpoints ----

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Simple endpoint to confirm the API is running and the model is loaded.
    Deployment platforms (and later, monitoring tools) ping this regularly.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_pipeline is not None
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Classifies a piece of text as Spam or Ham.
    Returns the predicted label along with confidence scores.
    """
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    start_time = time.time()

    cleaned = clean_text(request.text)

    if cleaned.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Text became empty after cleaning (e.g. only URLs/numbers/stopwords). Provide more substantive text."
        )

    prediction = model_pipeline.predict([cleaned])[0]
    probabilities = model_pipeline.predict_proba([cleaned])[0]

    # model_pipeline.classes_ tells us which probability column corresponds to which label
    class_labels = list(model_pipeline.classes_)
    spam_prob = float(probabilities[class_labels.index("Spam")])
    ham_prob = float(probabilities[class_labels.index("Ham")])

    processing_time_ms = (time.time() - start_time) * 1000

    logger.info(
        f"Prediction made | label={prediction} | spam_prob={spam_prob:.4f} | "
        f"time_ms={processing_time_ms:.2f} | text_length={len(request.text)}"
    )

    # Update monitoring stats
    metrics_store["total_predictions"] += 1
    metrics_store["total_latency_ms"] += processing_time_ms
    if prediction == "Spam":
        metrics_store["spam_count"] += 1
    else:
        metrics_store["ham_count"] += 1

    return PredictionResponse(
        label=prediction,
        confidence=max(spam_prob, ham_prob),
        spam_probability=spam_prob,
        ham_probability=ham_prob,
        processing_time_ms=round(processing_time_ms, 2)
    )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """
    Exposes basic operational stats about this API instance.
    In a real production deployment, a tool like Prometheus/Grafana would scrape
    an endpoint like this continuously to build dashboards and alerts over time.
    """
    total = metrics_store["total_predictions"]

    if total == 0:
        avg_latency = 0.0
        spam_ratio = 0.0
    else:
        avg_latency = metrics_store["total_latency_ms"] / total
        spam_ratio = metrics_store["spam_count"] / total

    uptime = (datetime.datetime.utcnow() - metrics_store["start_time"]).total_seconds()

    return MetricsResponse(
        total_predictions=total,
        spam_count=metrics_store["spam_count"],
        ham_count=metrics_store["ham_count"],
        spam_ratio=round(spam_ratio, 4),
        average_latency_ms=round(avg_latency, 2),
        uptime_seconds=round(uptime, 1)
    )