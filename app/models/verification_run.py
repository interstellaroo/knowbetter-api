from sqlalchemy import Column, Integer, String, DateTime, Float, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base

class VerificationRun(Base):
    __tablename__ = "verification_runs"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Basic article metadata (NO TEXT CONTENT)
    url = Column(String(2048), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    domain = Column(String(255), nullable=False, index=True)
    authors = Column(JSON)  # List of author names
    publish_date = Column(DateTime, nullable=True)
    
    # Run metadata
    processing_start = Column(DateTime, nullable=False, index=True)
    processing_end = Column(DateTime, nullable=False)
    total_processing_time = Column(Float, nullable=False)  # seconds
    status = Column(String(20), nullable=False, default="completed")  # completed, failed, processing
    overall_credibility_score = Column(Float, nullable=False, default=0.0)
    
    # Error info (if failed)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships to pipeline stages
    sentences = relationship("Sentence", back_populates="run", cascade="all, delete-orphan")
    selections = relationship("Selection", back_populates="run", cascade="all, delete-orphan")
    disambiguations = relationship("Disambiguation", back_populates="run", cascade="all, delete-orphan")
    decompositions = relationship("Decomposition", back_populates="run", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="run", cascade="all, delete-orphan")
    evidence_items = relationship("Evidence", back_populates="run", cascade="all, delete-orphan")
    verifications = relationship("Verification", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<VerificationRun(id={self.id}, url='{self.url}', score={self.overall_credibility_score})>"
