from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import joblib
import time

CATEGORIES = ["support","sales","hr","other"]
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"

_MODEL = None
_MODEL_MTIME = 0.0

def _normalize(text: str) -> str:
    return (text or "").lower()

def _rule_probs(text: str):
    rules = {
        "support": ["error","bug","issue","crash","fail","cannot","can't","login","outage","urgent","downtime","refund","invoice","charge"],
        "sales": ["pricing","quote","license","seat","trial","demo","contract","upgrade","purchase"],
        "hr": ["hiring","interview","candidate","recruiting","offer","benefits","payroll","vacation","leave"],
    }
    t = _normalize(text)
    scores = {k: 0.0 for k in CATEGORIES}
    for cat, kws in rules.items():
        for kw in kws:
            if kw in t:
                scores[cat] += 1.0
    scores["other"] = max(0.25, 2.0 - sum(scores.values()))
    total = sum(scores.values()) or 1.0
    probs = {k: float(v/total) for k,v in scores.items()}
    cat = max(probs, key=probs.get)
    return cat, probs

def _maybe_reload():
    global _MODEL, _MODEL_MTIME
    try:
        m = MODEL_PATH.stat().st_mtime
    except FileNotFoundError:
        _MODEL = None
        _MODEL_MTIME = 0.0
        return
    if not _MODEL or m != _MODEL_MTIME:
        _MODEL = joblib.load(MODEL_PATH)
        _MODEL_MTIME = m

def predict(subject: str, body: str):
    _maybe_reload()
    text = f"{subject or ''} {body or ''}".strip()
    if _MODEL:
        vec, clf, classes = _MODEL["vec"], _MODEL["clf"], _MODEL["classes"]
        proba = clf.predict_proba(vec.transform([text]))[0]
        probs = {c: float(p) for c,p in zip(classes, proba)}
        cat = max(probs, key=probs.get)
        # ensure all categories present in output
        for c in CATEGORIES:
            probs.setdefault(c, 0.0)
        return cat, probs
    return _rule_probs(text)
