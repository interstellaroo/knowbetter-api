from fastapi import APIRouter, Request, HTTPException, Depends
from newspaper.exceptions import ArticleBinaryDataException
from app.schemas.inputs import UrlInputSchema
from app.services.article import run_article_processing
from app.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
import asyncio


logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(
    prefix='/article',
    tags=["articles"]
)

@router.post("/process")
@limiter.limit("3/minute")
async def article_process(request: Request, input: UrlInputSchema, db: AsyncSession = Depends(get_db)):
    logger.info(f"Article processing for: {str(input.url)}")

    if await request.is_disconnected():
        logger.info(f"Client disconnected before processing started for: {input.url}")
        raise HTTPException(status_code=499, detail="Client disconnected")
    
    try:
        task = asyncio.create_task(run_article_processing(str(input.url), db))
        
        while not task.done():
            if await request.is_disconnected():
                logger.info(f"Client disconnected during processing for: {input.url}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Successfully cancelled processing for: {input.url}")
                raise HTTPException(status_code=499, detail="Client disconnected")
            
            await asyncio.sleep(0.1)

        return await task
        
    except asyncio.CancelledError:
        logger.info(f"Processing cancelled for: {input.url}")
        raise HTTPException(status_code=499, detail="Request cancelled")
    except ArticleBinaryDataException as e:
        raise HTTPException(status_code=422, detail="Article processing failed due to binary data issue.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
