import os
from collections import Counter
from sqlalchemy import create_engine, text

url = os.getenv("DATABASE_URL")
e = create_engine(url, future=True)

with e.connect() as c:
    res = c.execute(text("""
        SELECT e.subject, e.body, COALESCE(f.correct_category, p.category) AS label
        FROM emails e
        LEFT JOIN predictions p ON p.email_id = e.id
        LEFT JOIN feedback f ON f.email_id = e.id
        WHERE COALESCE(f.correct_category, p.category) IS NOT NULL
    """))
    rows = res.fetchall()

texts = [f"{(s or '').strip()} {(b or '').strip()}".strip() for s,b,_ in rows]
labels = [lab for *_, lab in rows]

if len(texts) < 2 or len(set(labels)) < 2:
    print(f"not enough labeled data to train (docs={len(texts)}, classes={len(set(labels))}); skipping")
    raise SystemExit(0)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

n_docs = len(texts)
min_df = 1 if n_docs < 20 else 2
vec = TfidfVectorizer(ngram_range=(1,2), min_df=min_df, max_features=5000)
X = vec.fit_transform(texts)
clf = LogisticRegression(max_iter=300, class_weight="balanced")
clf.fit(X, labels)

from pathlib import Path
import joblib
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump({"vec": vec, "clf": clf, "classes": clf.classes_.tolist()}, MODEL_DIR / "model.pkl")
print("trained on", n_docs, "examples with classes", Counter(labels))
