from fastapi import APIRouter
from app.schemas.inputs import UrlInputSchema
from app.services.article import process_article
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/article',
    tags=["articles"]
)

@router.post("/process")
async def article_process(input: UrlInputSchema):
    logger.info(f"Article processing for: {str(input.url)}")
    return await process_article(str(input.url))
