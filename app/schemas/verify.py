from pydantic import BaseModel
from typing import Dict, List
from datetime import datetime

class NLIResult(BaseModel):
    claim: str
    evidence_summary: str
    evidence_source: str
    verdict: str  # "ENTAILMENT", "CONTRADICTION", "NEUTRAL"
    confidence: float
    scores: Dict[str, float]
    timestamp: datetime

class VerificationBatch(BaseModel):
    results: List[NLIResult]
    total_processed: int
    processing_time: float
    timestamp: datetime