from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base

class Selection(Base):
    __tablename__ = "selections"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("verification_runs.id"), nullable=False)
    
    original_sentence = Column(Text, nullable=False)
    verification_label = Column(String(50), nullable=False)  # verifiable, not_verifiable
    rewritten_sentence = Column(Text, nullable=True)
    
    run = relationship("VerificationRun", back_populates="selections")
