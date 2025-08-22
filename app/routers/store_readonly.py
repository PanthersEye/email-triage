from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db import get_db
from app.models import Email, Prediction

router = APIRouter()

@router.get("/emails")
def list_emails(db: Session = Depends(get_db)):
    return db.query(Email).order_by(Email.created_at.desc()).limit(200).all()

@router.get("/emails/{email_id}")
def get_email(email_id: int, db: Session = Depends(get_db)):
    return db.get(Email, email_id)

@router.get("/emails/{email_id}/predictions")
def get_preds(email_id: int, db: Session = Depends(get_db)):
    return (
        db.query(Prediction)
        .filter(Prediction.email_id == email_id)
        .order_by(Prediction.id.desc())
        .all()
    )
