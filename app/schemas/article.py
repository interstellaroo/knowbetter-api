from pydantic import BaseModel
from typing import List
from datetime import datetime


class ArticleData(BaseModel):
    url: str
    title: str
    authors: List[str]
    text: str
    publish_date: datetime
    summary: str
    paragraphs: List[str]


class SentenceData(BaseModel):
    index: int
    sentence: str
    context: str
    paragraph: str


class SplittingData(BaseModel):
    sentences: List[SentenceData]
    count: int


class SelectionResult(BaseModel):
    original_sentence: str
    verification_label: str
    rewritten_sentence: str

class DisambiguationResult(BaseModel):
    original_sentence: str
    disambiguated_sentence: str
    reason: str

class ArticleExtractionData(BaseModel):
    article: ArticleData
    data: SplittingData
