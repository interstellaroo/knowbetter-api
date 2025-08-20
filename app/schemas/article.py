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