from sqlalchemy import Column, Integer, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base

class Disambiguation(Base):
    __tablename__ = "disambiguations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("verification_runs.id"), nullable=False)
    
    original_sentence = Column(Text, nullable=False)
    disambiguated_sentence = Column(Text, nullable=False)
    reason = Column(Text, nullable=True)
    
    run = relationship("VerificationRun", back_populates="disambiguations")
