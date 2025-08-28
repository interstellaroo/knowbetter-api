from pydantic import BaseModel
from typing import List
from datetime import datetime


class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    displayLink: str

class EvidenceResult(BaseModel):
    claim: str
    evidence: List[SearchResult]
    timestamp: datetime

class EvidenceRetrievalResult(BaseModel):
    evidence: List[EvidenceResult]
    excluded_domain: str
    timestamp: datetime