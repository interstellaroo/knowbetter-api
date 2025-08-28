from sqlalchemy import Column, Integer, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base

class Decomposition(Base):
    __tablename__ = "decompositions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("verification_runs.id"), nullable=False)
    
    original_claim = Column(Text, nullable=False)
    decomposed_claims = Column(JSON, nullable=False)  # List of claims
    
    run = relationship("VerificationRun", back_populates="decompositions")
