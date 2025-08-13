import os
import random
import joblib
import pandas as pd
from threading import RLock
from typing import List, Dict, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_DIR = "model"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
XTEST_PATH = os.path.join(MODEL_DIR, "X_test_tfidf.joblib")
YTEST_PATH = os.path.join(MODEL_DIR, "y_test.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)
_lock = RLock()

CATEGORIES = ["support", "sales", "hr", "finance", "general"]
SAMPLE_EMAILS: Dict[str, List[tuple]] = {
    "support": [
        ("Login issue", "I can’t access my dashboard despite entering the right credentials."),
        ("App keeps crashing", "The mobile app crashes every time I open the notifications tab."),
        ("Email not received", "I didn’t get the confirmation email for my registration."),
        ("Feature not working", "The export to PDF button doesn’t do anything."),
        ("Error 504", "I keep getting a 504 error when trying to reach my profile."),
    ],
    "sales": [
        ("Bulk order inquiry", "Can we get a quote for 200 units of your product?"),
        ("Interested in your service", "We’d like a demo of your analytics suite."),
        ("Pricing model", "What are your enterprise licensing options?"),
        ("Long-term contract", "Is there a discount for multi-year agreements?"),
        ("Reseller program", "Do you have a partner program for resellers?"),
    ],
    "hr": [
        ("Leave application", "I'd like to apply for leave from Sept 4–10."),
        ("Remote work policy", "Can you confirm if we’re remote on Fridays?"),
        ("Job opening", "Is the Data Analyst role still accepting applications?"),
        ("Interview follow-up", "Any update on my second-round interview?"),
        ("PTO balance", "Could you confirm how many PTO days I have left?"),
    ],
    "finance": [
        ("Missing invoice", "We didn’t receive our May invoice."),
        ("Payment delay", "Our finance team will process payment next week."),
        ("Expense approval", "Please approve the attached travel expenses."),
        ("Tax report request", "Can you provide Q2’s tax breakdown?"),
        ("Incorrect billing", "We were charged for 10 users instead of 5."),
    ],
    "general": [
        ("Office supplies", "We’re low on printer ink and sticky notes."),
        ("Meeting room booking", "Can I reserve Room A for Thursday 3PM?"),
        ("Team lunch", "Will we still have lunch catered Friday?"),
        ("Building access", "My badge stopped working this morning."),
        ("All-hands prep", "What’s the agenda for next week’s company-wide?"),
    ],
}

vectorizer: Optional[TfidfVectorizer] = None
model: Optional[LogisticRegression] = None
classes_: Optional[List[str]] = None

def _combine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["subject"] + " " + df["body"]
    return df

def make_synthetic(n_per_cat: int = 12) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_cat):
            subject, body = random.choice(SAMPLE_EMAILS[cat])
            rows.append({"subject": subject, "body": body, "category": cat})
    return _combine(pd.DataFrame(rows))

def load_if_available() -> bool:
    global vectorizer, model, classes_
    with _lock:
        if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
            vectorizer = joblib.load(VECTORIZER_PATH)
            model = joblib.load(MODEL_PATH)
            classes_ = list(model.classes_)
            return True
    return False

def train_df(df: pd.DataFrame) -> Tuple[float, List[str], int]:
    global vectorizer, model, classes_
    with _lock:
        X = df["text"]; y = df["category"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = TfidfVectorizer(stop_words="english")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        acc = float(model.score(X_test_tfidf, y_test))
        classes_ = list(model.classes_)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X_test_tfidf, XTEST_PATH)
        joblib.dump(y_test.to_numpy(), YTEST_PATH)
        return acc, classes_, len(df)

def predict_proba(texts: List[str]) -> Tuple[List[str], List[List[float]], List[str]]:
    with _lock:
        if vectorizer is None or model is None:
            if not load_if_available():
                raise RuntimeError("Model not trained")
        X = vectorizer.transform(texts)
        preds = model.predict(X).tolist()
        probas = model.predict_proba(X).tolist()
        return preds, probas, list(model.classes_)

def accuracy_snapshot() -> Optional[float]:
    with _lock:
        if model is None or not (os.path.exists(XTEST_PATH) and os.path.exists(YTEST_PATH)):
            return None
        X_test_tfidf = joblib.load(XTEST_PATH)
        y_test = joblib.load(YTEST_PATH)
        return float(model.score(X_test_tfidf, y_test))
