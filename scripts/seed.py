from app.db import SessionLocal
from app.models import Email, Prediction
db = SessionLocal()
e = Email(subject="Onboarding question", body="How do I reset my password?")
db.add(e); db.flush()
p = Prediction(email_id=e.id, category="support",
               probabilities={"support":0.9,"sales":0.05,"hr":0.0,"other":0.05},
               priority_score=0.82, priority_label="High")
db.add(p); db.commit(); db.close()
print("seeded email id:", e.id)
