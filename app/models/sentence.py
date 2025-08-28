from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from app.db import Base

class Sentence(Base):
    __tablename__ = "sentences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("verification_runs.id"), nullable=False)
    
    index = Column(Integer, nullable=False)
    sentence = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    paragraph = Column(Text, nullable=True)
    
    run = relationship("VerificationRun", back_populates="sentences")
