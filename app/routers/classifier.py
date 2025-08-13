from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import RedirectResponse
import pandas as pd
from typing import Dict

from app.schemas import (
    HealthResponse, TrainSynthRequest, TrainResponse,
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse, BatchPredictItem,
    ReplyRequest, ReplyResponse,
)
from app.services import model as M
from app.services.priority import priority_score, priority_label
from app.core.config import settings

# Optional OpenAI
try:
    from openai import OpenAI
    oai_client = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
except Exception:
    oai_client = None

router = APIRouter(prefix="", tags=["Email Triage"])

@router.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@router.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse()

@router.post(
    "/train_synth",
    response_model=TrainResponse,
    status_code=status.HTTP_200_OK,
    summary="Train on synthetic samples",
    description="Generates N examples per category and trains TF-IDF + LogisticRegression.",
)
def train_synth(req: TrainSynthRequest):
    df = M.make_synthetic(n_per_cat=req.n_per_cat)
    acc, classes, samples = M.train_df(df)
    return TrainResponse(samples=samples, classes=classes, accuracy=acc)

@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Train on uploaded CSV",
    description="Upload a CSV with columns: subject, body, category.",
)
def train(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(400, f"Failed to read CSV: {e}")
    required = {"subject", "body", "category"}
    if not required.issubset(df.columns):
        raise HTTPException(400, "CSV must have columns: subject, body, category")
    df = df.assign(text=df["subject"] + " " + df["body"])
    acc, classes, samples = M.train_df(df)
    return TrainResponse(samples=samples, classes=classes, accuracy=acc)

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict category + priority",
    description="Returns predicted category, class probabilities, and a simple priority score/label.",
)
def predict(req: PredictRequest):
    try:
        preds, probas, classes = M.predict_proba([f"{req.subject} {req.body}"])
    except RuntimeError:
        raise HTTPException(400, "Model not trained yet. Call /train_synth or /train.")
    text = f"{req.subject} {req.body}"
    ps = priority_score(text); pl = priority_label(ps)
    probs: Dict[str, float] = {c: float(p) for c, p in zip(classes, probas[0])}
    return PredictResponse(category=preds[0], probabilities=probs, priority_score=ps, priority_label=pl)

@router.post(
    "/batch_predict",
    response_model=BatchPredictResponse,
    summary="Batch predict",
    description="Predict for a list of emails in one request.",
)
def batch_predict(req: BatchPredictRequest):
    texts = [f"{it.subject} {it.body}" for it in req.items]
    try:
        preds, probas, classes = M.predict_proba(texts)
    except RuntimeError:
        raise HTTPException(400, "Model not trained yet. Call /train_synth or /train.")
    out = []
    for text, pred, proba in zip(texts, preds, probas):
        ps = priority_score(text); pl = priority_label(ps)
        probs = {c: float(p) for c, p in zip(classes, proba)}
        out.append(BatchPredictItem(category=pred, probabilities=probs, priority_score=ps, priority_label=pl))
    return BatchPredictResponse(results=out)

@router.post(
    "/reply",
    response_model=ReplyResponse,
    summary="Draft a professional reply (OpenAI)",
    description="Uses OpenAI Chat Completions if OPENAI_API_KEY is configured; otherwise returns a placeholder.",
)
def reply(req: ReplyRequest):
    if not settings.OPENAI_API_KEY or oai_client is None:
        return ReplyResponse(draft="[OpenAI API key not configured — cannot generate reply.]")
    messages = [
        {"role": "system", "content": "You write concise, professional business email replies."},
        {"role": "user", "content": (
            f"Subject: {req.subject}\n"
            f"Body: {req.body}\n\n"
            f"Guidance: {req.guidance}\n\n"
            "Write the reply email body only."
        )},
    ]
    try:
        resp = oai_client.chat.completions.create(
            model=settings.OPENAI_MODEL, messages=messages, temperature=0.4
        )
        draft = resp.choices[0].message.content.strip()
        return ReplyResponse(draft=draft)
    except Exception as e:
        raise HTTPException(502, f"OpenAI error: {e}")
