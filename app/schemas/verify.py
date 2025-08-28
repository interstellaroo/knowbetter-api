from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from app.schemas.article import SelectionResult, DisambiguationResult, DecompositionResult
from app.schemas.retrive import EvidenceRetrievalResult

class NLIResult(BaseModel):
    claim: str
    evidence_summary: str
    evidence_source: str
    verdict: str  #
    confidence: float
    scores: Dict[str, float]
    timestamp: datetime

class VerificationBatch(BaseModel):
    results: List[NLIResult]
    total_processed: int
    processing_time: float
    timestamp: datetime

class ProcessingStatistics(BaseModel):
    total_sentences: int
    verifiable_sentences: int
    disambiguated_sentences: int
    total_claims: int
    claims_with_evidence: int
    verified_claims: int
    processing_time_seconds: float

class PipelineStepResults(BaseModel):
    selection_results: List[SelectionResult]
    disambiguation_results: List[DisambiguationResult]
    decomposition_results: List[DecompositionResult]
    evidence_results: Optional[EvidenceRetrievalResult]
    verification_results: Optional[VerificationBatch]

class ArticleSummary(BaseModel):
    url: str
    title: str
    domain: str
    authors: List[str]
    publish_date: Optional[datetime]

class VerdictSummary(BaseModel):
    entailed_claims: int
    contradicted_claims: int
    neutral_claims: int
    unsupported_claims: int
    overall_credibility_score: float

class FinalValidationResult(BaseModel):
    article: ArticleSummary
    statistics: ProcessingStatistics
    pipeline_results: PipelineStepResults
    verdict_summary: VerdictSummary
    processing_start: datetime
    processing_end: datetime
    total_processing_time: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }