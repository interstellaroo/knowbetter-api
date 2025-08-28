from sqlalchemy import Column, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base

class Claim(Base):
    __tablename__ = "claims"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("verification_runs.id"), nullable=False)
    decomposition_id = Column(Integer, ForeignKey("decompositions.id"), nullable=True)
    
    claim_text = Column(Text, nullable=False)
    claim_index = Column(Integer, nullable=False)  # Index within decomposition
    
    run = relationship("VerificationRun", back_populates="claims")
    evidence_items = relationship("Evidence", back_populates="claim", cascade="all, delete-orphan")
    verifications = relationship("Verification", back_populates="claim", cascade="all, delete-orphan")
