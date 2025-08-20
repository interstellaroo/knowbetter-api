from newspaper import Article
from app.schemas.article import ArticleData
from pydantic import HttpUrl
import logging

logger = logging.getLogger(__name__) 

async def extract_article(url: str) -> ArticleData:
    """Extracts the article from the provided URL using newspaper4k library.

    Args:
        url (str): URL address for the article.

    Raises:
        Exception: Throws a generic exception if anythin goes wrong.

    Returns:
        ArticleData: Data extracted from the article, such as title, authors, text, publish 
        date and summary generated using newspaperk4k NLP functionality
    """
    try:
        logger.debug(f"Starting article extraction for: {url}")
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        
        return ArticleData(
            url=url,
            title=article.title,
            authors=article.authors,
            text=article.text,
            publish_date=article.publish_date,
            summary=article.summary
        )
    except Exception as e:
        logger.error(f"There was an error durign article extraction for: {url}\n Error:{e}")
        raise Exception("There was an error with article extraction. Try again...")
    
    
async def process_article(url: str):
    article = await extract_article(url)
    return article