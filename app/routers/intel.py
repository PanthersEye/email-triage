from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import Email, Prediction, Reply, Feedback
from app.services.classifier import predict as clf_predict
from app.services.priority import score as pr_score
from app.services.replies import draft as make_draft

router = APIRouter()

@router.post("/predict")
def predict(payload: dict, db: Session = Depends(get_db)):
    subject = payload.get("subject","")
    body = payload.get("body","")
    email = Email(subject=subject, body=body)
    db.add(email); db.flush()
    cat, probs = clf_predict(subject, body)
    pr_s, pr_label = pr_score(subject, body, cat)
    pred = Prediction(email_id=email.id, category=cat, probabilities=probs, priority_score=pr_s, priority_label=pr_label)
    db.add(pred); db.commit()
    return {"email_id": email.id, "category": cat, "probabilities": probs, "priority_score": pr_s, "priority_label": pr_label}

@router.post("/reply")
def reply(payload: dict, db: Session = Depends(get_db)):
    subject = payload.get("subject","")
    body = payload.get("body","")
    email = Email(subject=subject, body=body)
    db.add(email); db.flush()
    cat, _ = clf_predict(subject, body)
    _, pr_label = pr_score(subject, body, cat)
    text = make_draft(subject, body, cat, pr_label)
    r = Reply(email_id=email.id, draft=text)
    db.add(r); db.commit()
    return {"email_id": email.id, "draft": text, "category": cat, "priority_label": pr_label}

@router.post("/feedback")
def feedback(payload: dict, db: Session = Depends(get_db)):
    fb = Feedback(
        email_id=payload.get("email_id"),
        correct_category=payload.get("correct_category"),
        correct_priority_label=payload.get("correct_priority_label"),
        draft_helpful=payload.get("draft_helpful"),
        better_draft=payload.get("better_draft"),
    )
    db.add(fb); db.commit()
    return {"ok": True}

@router.get("/metrics")
def metrics(db: Session = Depends(get_db)):
    total = db.query(Email).count()
    labeled = db.query(Feedback).count()
    return {"emails": total, "feedback": labeled}
