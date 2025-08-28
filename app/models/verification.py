from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base

class Verification(Base):
    __tablename__ = "verifications"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("verification_runs.id"), nullable=False)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    evidence_id = Column(Integer, ForeignKey("evidence.id"), nullable=True)
    
    claim_text = Column(Text, nullable=False)
    evidence_summary = Column(Text, nullable=True)
    evidence_source = Column(String(255), nullable=True)
    verdict = Column(String(20), nullable=False)  # ENTAILMENT, CONTRADICTION, NEUTRAL
    confidence = Column(Float, nullable=False)
    scores = Column(JSON, nullable=False)  # {"ENTAILMENT": 0.8, "CONTRADICTION": 0.1, "NEUTRAL": 0.1}
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    run = relationship("VerificationRun", back_populates="verifications")
    claim = relationship("Claim", back_populates="verifications")
