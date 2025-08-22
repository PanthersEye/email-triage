#!/usr/bin/env bash
set -euo pipefail

# Must run from repo root (where .git lives)
[ -d .git ] || { echo "❌ Run from your repo root (where .git is)."; exit 1; }

# Ensure backend skeleton exists
mkdir -p app/{routers,services,core}
touch app/__init__.py app/routers/__init__.py app/services/__init__.py app/core/__init__.py

# ---------- Backend files ----------
cat > requirements.txt << 'EOF'
fastapi
uvicorn[standard]
pandas
scikit-learn
joblib
pydantic-settings
openai>=1.0.0
python-multipart
SQLAlchemy>=2.0
alembic
psycopg2-binary
EOF

cat > app/core/config.py << 'EOF'
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Email Triage & Reply API"
    APP_VERSION: str = "1.2.0"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    DATABASE_URL: str = "sqlite:///./email_triage.db"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    class Config:
        env_file = ".env"

settings = Settings()
EOF

cat > app/db.py << 'EOF'
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
EOF

cat > app/models.py << 'EOF'
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from app.db import Base

class EmailRecord(Base):
    __tablename__ = "emails"
    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String, nullable=False)
    body = Column(String, nullable=False)
    predicted_category = Column(String, nullable=True)
    probabilities = Column(JSON, nullable=True)
    priority_score = Column(Float, nullable=True)
    priority_label = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
EOF

cat > app/services/priority.py << 'EOF'
import re
URGENCY_KEYWORDS = ["urgent","asap","immediately","important","help","emergency","issue","crash","broken","can't access","need","problem"]
def priority_score(text: str) -> float:
    t = text.lower(); hits = 0
    for kw in URGENCY_KEYWORDS:
        if kw == "can't access":
            if "can't access" in t or "cant access" in t: hits += 1
        elif re.search(rf"\b{re.escape(kw)}\b", t): hits += 1
    return min(hits / len(URGENCY_KEYWORDS), 1.0)
def priority_label(score: float) -> str:
    return "High" if score > 0.30 else "Medium" if score > 0.10 else "Low"
EOF

cat > app/services/model.py << 'EOF'
import os, random, joblib, pandas as pd
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

CATEGORIES = ["support","sales","hr","finance","general"]
SAMPLE_EMAILS: Dict[str, List[tuple]] = {
    "support":[("Login issue","I can’t access my dashboard despite entering the right credentials."),
               ("App keeps crashing","The mobile app crashes every time I open the notifications tab."),
               ("Email not received","I didn’t get the confirmation email for my registration."),
               ("Feature not working","The export to PDF button doesn’t do anything."),
               ("Error 504","I keep getting a 504 error when trying to reach my profile.")],
    "sales":[("Bulk order inquiry","Can we get a quote for 200 units of your product?"),
             ("Interested in your service","We’d like a demo of your analytics suite."),
             ("Pricing model","What are your enterprise licensing options?"),
             ("Long-term contract","Is there a discount for multi-year agreements?"),
             ("Reseller program","Do you have a partner program for resellers?")],
    "hr":[("Leave application","I'd like to apply for leave from Sept 4–10."),
          ("Remote work policy","Can you confirm if we’re remote on Fridays?"),
          ("Job opening","Is the Data Analyst role still accepting applications?"),
          ("Interview follow-up","Any update on my second-round interview?"),
          ("PTO balance","Could you confirm how many PTO days I have left?")],
    "finance":[("Missing invoice","We didn’t receive our May invoice."),
               ("Payment delay","Our finance team will process payment next week."),
               ("Expense approval","Please approve the attached travel expenses."),
               ("Tax report request","Can you provide Q2’s tax breakdown?"),
               ("Incorrect billing","We were charged for 10 users instead of 5.")],
    "general":[("Office supplies","We’re low on printer ink and sticky notes."),
               ("Meeting room booking","Can I reserve Room A for Thursday 3PM?"),
               ("Team lunch","Will we still have lunch catered Friday?"),
               ("Building access","My badge stopped working this morning."),
               ("All-hands prep","What’s the agenda for next week’s company-wide?")],
}

vectorizer: Optional[TfidfVectorizer] = None
model: Optional[LogisticRegression] = None

def _combine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df["text"] = df["subject"] + " " + df["body"]; return df

def make_synthetic(n_per_cat: int = 12) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        for _ in range(n_per_cat):
            subject, body = random.choice(SAMPLE_EMAILS[cat])
            rows.append({"subject": subject, "body": body, "category": cat})
    return _combine(pd.DataFrame(rows))

def train_df(df: pd.DataFrame) -> Tuple[float, List[str], int]:
    X, y = df["text"], df["category"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vec = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    acc = float(clf.score(X_test_tfidf, y_test))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vec, VECTORIZER_PATH); joblib.dump(clf, MODEL_PATH)
    return acc, list(clf.classes_), len(df)

def predict_one(text: str) -> Tuple[str, Dict[str,float]]:
    if not (os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH)):
        raise RuntimeError("Model not trained")
    vec = joblib.load(VECTORIZER_PATH)
    clf = joblib.load(MODEL_PATH)
    X = vec.transform([text])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    return pred, {c: float(p) for c, p in zip(clf.classes_, proba)}
EOF

cat > app/schemas.py << 'EOF'
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class HealthResponse(BaseModel):
    status: str = "ok"

class TrainSynthRequest(BaseModel):
    n_per_cat: int = Field(12, ge=1, le=200)

class TrainResponse(BaseModel):
    samples: int
    classes: List[str]
    accuracy: float

class PredictRequest(BaseModel):
    subject: str
    body: str

class PredictResponse(BaseModel):
    category: str
    probabilities: Dict[str, float]
    priority_score: float
    priority_label: str

class EmailCreate(BaseModel):
    subject: str
    body: str

class EmailOut(BaseModel):
    id: int
    subject: str
    body: str
    predicted_category: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None
    priority_score: Optional[float] = None
    priority_label: Optional[str] = None
    created_at: str
    class Config:
        from_attributes = True
EOF

cat > app/routers/classifier.py << 'EOF'
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, List
from sqlalchemy.orm import Session

from app.schemas import (
    HealthResponse, TrainSynthRequest, TrainResponse,
    PredictRequest, PredictResponse,
    EmailCreate, EmailOut
)
from app.services import model as M
from app.services.priority import priority_score, priority_label
from app.db import get_db
from app.models import EmailRecord

router = APIRouter(prefix="", tags=["Email Triage"])

@router.get("/health", response_model=HealthResponse)
def health(): return HealthResponse()

@router.post("/train_synth", response_model=TrainResponse, status_code=status.HTTP_200_OK)
def train_synth(req: TrainSynthRequest):
    df = M.make_synthetic(n_per_cat=req.n_per_cat)
    acc, classes, samples = M.train_df(df)
    return TrainResponse(samples=samples, classes=classes, accuracy=acc)

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        pred, probs = M.predict_one(f"{req.subject} {req.body}")
    except RuntimeError:
        raise HTTPException(400, "Model not trained yet. Call /train_synth first.")
    text = f"{req.subject} {req.body}"
    ps = priority_score(text); pl = priority_label(ps)
    return PredictResponse(category=pred, probabilities=probs, priority_score=ps, priority_label=pl)

# DB-backed create + list
@router.post("/emails", response_model=EmailOut)
def create_email(item: EmailCreate, db: Session = Depends(get_db)):
    try:
        pred, probs = M.predict_one(f"{item.subject} {item.body}")
    except RuntimeError:
        raise HTTPException(400, "Model not trained yet. Call /train_synth first.")
    text = f"{item.subject} {item.body}"
    ps = priority_score(text); pl = priority_label(ps)
    rec = EmailRecord(
        subject=item.subject, body=item.body,
        predicted_category=pred, probabilities=probs,
        priority_score=ps, priority_label=pl
    )
    db.add(rec); db.commit(); db.refresh(rec)
    return rec

@router.get("/emails", response_model=List[EmailOut])
def list_emails(db: Session = Depends(get_db), limit: int = 50, offset: int = 0):
    return db.query(EmailRecord).order_by(EmailRecord.created_at.desc()).offset(offset).limit(limit).all()
EOF

cat > app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db import Base, engine
import app.models  # ensure models imported
from app.routers import classifier

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Classify emails, estimate priority, and persist results.",
)

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(classifier.router)
EOF

# ---------- Frontend (Vite React + Tailwind) ----------
if [ ! -d "email-triage-ui" ]; then
  npm create vite@latest email-triage-ui -- --template react-ts >/dev/null 2>&1
fi
cd email-triage-ui
npm install >/dev/null 2>&1
npm install -D tailwindcss postcss autoprefixer >/dev/null 2>&1
npx tailwindcss init -p >/dev/null 2>&1

cat > tailwind.config.js << 'EOF'
export default { content: ["./index.html","./src/**/*.{ts,tsx}"], theme: { extend: {} }, plugins: [] }
EOF

cat > src/index.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF

cat > .env << 'EOF'
VITE_API_BASE=http://localhost:8000
EOF

cat > src/api.ts << 'EOF'
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
export type EmailOut = { id:number; subject:string; body:string; predicted_category?:string; probabilities?:Record<string,number>; priority_score?:number; priority_label?:string; created_at:string; };
export async function trainSynth(n=12){const r=await fetch(`${API_BASE}/train_synth`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({n_per_cat:n})});if(!r.ok)throw new Error(await r.text());return r.json();}
export async function createEmail(subject:string, body:string):Promise<EmailOut>{const r=await fetch(`${API_BASE}/emails`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({subject,body})});if(!r.ok)throw new Error(await r.text());return r.json();}
export async function listEmails():Promise<EmailOut[]>{const r=await fetch(`${API_BASE}/emails`);if(!r.ok)throw new Error(await r.text());return r.json();}
EOF

cat > src/App.tsx << 'EOF'
import { useEffect, useState } from "react";
import { trainSynth, createEmail, listEmails, EmailOut } from "./api";

function ProbBar({ probs }: { probs?: Record<string, number> }) {
  if (!probs) return null;
  const entries = Object.entries(probs).sort((a,b)=>b[1]-a[1]);
  return (
    <div className="space-y-1">
      {entries.map(([k,v]) => (
        <div key={k}>
          <div className="flex justify-between text-sm">
            <span className="font-medium">{k}</span>
            <span>{(v*100).toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded"><div className="h-2 rounded bg-gray-700" style={{width:`${v*100}%`}}/></div>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [subject, setSubject] = useState(""); const [body, setBody] = useState("");
  const [loading, setLoading] = useState(false); const [emails, setEmails] = useState<EmailOut[]>([]);
  const [trainMsg, setTrainMsg] = useState<string>("");

  async function refresh(){ setEmails(await listEmails()); }
  useEffect(()=>{ refresh(); }, []);

  async function handleTrain(){ setLoading(true); try{ const r=await trainSynth(12); setTrainMsg(`Trained: acc ${(r.accuracy*100).toFixed(1)}% on ${r.samples} samples`);}catch(e:any){ setTrainMsg(e.message||"Training failed"); } finally{ setLoading(false); } }
  async function handleSubmit(e:React.FormEvent){ e.preventDefault(); if(!subject||!body) return; setLoading(true);
    try{ await createEmail(subject,body); setSubject(""); setBody(""); await refresh(); } finally{ setLoading(false); } }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="sticky top-0 bg-white border-b p-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold">Email Triage Dashboard</h1>
          <button onClick={handleTrain} disabled={loading} className="px-3 py-2 rounded-lg border hover:bg-gray-100 disabled:opacity-50">
            {loading ? "Training..." : "Train (Synthetic)"}
          </button>
        </div>
        {trainMsg && <p className="max-w-5xl mx-auto text-sm pt-2">{trainMsg}</p>}
      </header>

      <main className="max-w-5xl mx-auto p-4 space-y-6">
        <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow p-4 space-y-3">
          <h2 className="font-semibold">New Email</h2>
          <input value={subject} onChange={e=>setSubject(e.target.value)} placeholder="Subject" className="w-full border rounded-lg p-2"/>
          <textarea value={body} onChange={e=>setBody(e.target.value)} placeholder="Body" className="w-full border rounded-lg p-2 h-28"/>
          <div className="flex justify-end"><button className="px-4 py-2 rounded-xl bg-black text-white disabled:opacity-50" disabled={loading||!subject||!body}>{loading?"Submitting...":"Submit & Predict"}</button></div>
        </form>

        <section className="bg-white rounded-2xl shadow p-4">
          <h2 className="font-semibold mb-3">Predicted Emails</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="text-left border-b"><th className="p-2">Created</th><th className="p-2">Subject</th><th className="p-2">Category</th><th className="p-2">Priority</th><th className="p-2">Probabilities</th></tr></thead>
              <tbody>
                {emails.map(e=>(
                  <tr key={e.id} className="border-b align-top">
                    <td className="p-2 whitespace-nowrap">{new Date(e.created_at).toLocaleString()}</td>
                    <td className="p-2"><div className="font-medium">{e.subject}</div><div className="text-gray-500 line-clamp-2">{e.body}</div></td>
                    <td className="p-2 font-semibold">{e.predicted_category ?? "-"}</td>
                    <td className="p-2"><span className="rounded-full px-2 py-0.5 text-xs border">{e.priority_label ?? "-"}</span> <span className="text-xs text-gray-500">{(e.priority_score ?? 0).toFixed(2)}</span></td>
                    <td className="p-2 w-80"><ProbBar probs={e.probabilities}/></td>
                  </tr>
                ))}
                {emails.length===0 && (<tr><td className="p-2 text-gray-500" colSpan={5}>No emails yet.</td></tr>)}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}
EOF

cd ..
echo "✅ Upgrade complete: backend + email-triage-ui created"

