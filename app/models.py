from __future__ import annotations
from datetime import datetime
from typing import Dict
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base

class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    subject: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="email", cascade="all, delete-orphan"
    )
    replies: Mapped[list["Reply"]] = relationship(
        "Reply", back_populates="email", cascade="all, delete-orphan"
    )

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(50))
    probabilities: Mapped[Dict[str, float]] = mapped_column(JSON)
    priority_score: Mapped[float] = mapped_column(Float)
    priority_label: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    email: Mapped["Email"] = relationship("Email", back_populates="predictions")

class Reply(Base):
    __tablename__ = "replies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), index=True)
    draft: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    email: Mapped["Email"] = relationship("Email", back_populates="replies")
class Feedback(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id", ondelete="CASCADE"), index=True)
    correct_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    correct_priority_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    draft_helpful: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    better_draft: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    email: Mapped["Email"] = relationship("Email")
