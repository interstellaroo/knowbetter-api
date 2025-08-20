from fastapi import APIRouter
from app.schemas.url_input import UrlInputSchema

router = APIRouter(
    prefix='/article',
    tags=["articles"]
)

@router.post("/process")
async def article_process(url: UrlInputSchema):
    print(url)
    return {"lol": "lol"}
