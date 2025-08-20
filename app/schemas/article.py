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


class SplittingData(BaseModel):
    sentences: List[SentenceData]
    count: int

class ArticleProcessingData(BaseModel):
    article: ArticleData
    data: SplittingData