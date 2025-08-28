from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import article
import logging
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)

app = FastAPI(
    title="Credibility Evaluation",
    description="Application for article and text fact-checking/credibility evaluation",
    version="v1.0"
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = article.limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(article.router)

@app.get("/")
def check():
    return {"message": "Application is up"}