from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base

class Evidence(Base):
    __tablename__ = "evidence"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("verification_runs.id"), nullable=False)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    
    title = Column(String(500), nullable=False)
    link = Column(String(2048), nullable=False)
    snippet = Column(Text, nullable=False)
    display_link = Column(String(255), nullable=False)
    retrieved_at = Column(DateTime, default=datetime.utcnow)
    
    run = relationship("VerificationRun", back_populates="evidence_items")
    claim = relationship("Claim", back_populates="evidence_items")
