from fastapi import APIRouter, Request, HTTPException
from newspaper.exceptions import ArticleBinaryDataException
from app.schemas.inputs import UrlInputSchema
from app.services.article import run_article_processing
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address


logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(
    prefix='/article',
    tags=["articles"]
)

@router.post("/process")
@limiter.limit("2/minute")
async def article_process(request: Request, input: UrlInputSchema):
    logger.info(f"Article processing for: {str(input.url)}")
    try:
        return await run_article_processing(str(input.url))
    except ArticleBinaryDataException as e:
        raise HTTPException(status_code=422, detail="Article processing failed due to binary data issue.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
