from newspaper import Article
from app.schemas.article import ArticleData
from bs4 import BeautifulSoup
from typing import List
import spacy
import logging

nlp = spacy.load("en_core_web_sm")
logger = logging.getLogger(__name__) 

def extract_article(url: str) -> ArticleData:
    """Extracts the article from the provided URL using newspaper4k library.

    Args:
        url (str): The URL of the article to extract

    Raises:
        Exception: If there is an error during article extraction generic error is thrown

    Returns:
        ArticleData: The extracted article data
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        paragraphs = get_article_paragraphs(article.html)
        
        result = ArticleData(
            url=url,
            title=article.title,
            authors=article.authors,
            text=article.text,
            publish_date=article.publish_date,
            summary=article.summary,
            paragraphs=paragraphs
        )
        logger.info("Article extraction completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error in extract_article: {e}")
        raise Exception(f"Article extraction failed: {str(e)}")

def get_article_paragraphs(html: str) -> List[str]:
    """Extracts paragraphs from the article HTML.

    Args:
        html (str): The HTML content of the article

    Raises:
        Exception: If there is an error during paragraph extraction

    Returns:
        List[str]: List of the extracted paragraphs
    """
    try:
        soup = BeautifulSoup(html, 'html.parser')
        paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
        return paragraphs
    except Exception as e:
        logger.error(f"Error extracting paragraphs: {e}")
        raise Exception(f"Failed to extract paragraphs: {str(e)}")

def split_into_sentences(article_text: str) -> List[str]:
    """Splitting entire article text into individual sentences with spacy

    Args:
        article_text (str): The full text of the article

    Raises:
        Exception: If there is an error during sentence splitting

    Returns:
        List[str]: List of the individual sentences
    """
    try:
        doc = nlp(article_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    except Exception as e:
        logger.error(f"Error splitting sentences: {e}")
        raise Exception(f"Failed to split sentences: {str(e)}")

def get_context(index: int, sentences: List[str]) -> str:
    """Creates the context for each sentence. The context consists of i - 1 and i + 1 sentences.

    Args:
        index (int): The index of the current sentence
        sentences (List[str]): The list of all sentences

    Returns:
        context (str): The context for the current sentence
    """
    previous = sentences[index - 1] if index > 0 else ""
    current = sentences[index]
    next = sentences[index + 1] if index < len(sentences) - 1 else ""
    context = f"{previous} {current} {next}".strip()
    return context

def match_paragraph(paragraphs: List[str], sentence: str) -> str:
    """Matches the sentence to the article paragraph. Later the paragraph is provided as an extended context during sentence processing.

    Args:
        paragraphs (List[str]): _description_
        sentence (str): _description_

    Returns:
        str: _description_
    """
    sentence = sentence.strip().replace("\n", "")
    for paragraph in paragraphs:
        paragraph = paragraph.strip().replace("\n", "")
        if sentence and sentence in paragraph:
            return paragraph
    return "Not found"

async def create_processing_data(article_data: ArticleData):
    try:
        logger.info("Creating processing data...")
        sentences = split_into_sentences(article_data.text)
        
        processed_sentences = []
        for i, sentence in enumerate(sentences):
            context = get_context(i, sentences)
            paragraph = match_paragraph(article_data.paragraphs, sentence)
            processed_sentences.append({
                "index": i,
                "sentence": sentence,
                "context": context,
                "paragraph": paragraph
            })
        
        logger.info("Processing data created successfully")
        return processed_sentences
    except Exception as e:
        logger.error(f"Error creating processing data: {e}")
        raise

async def process_article(url: str):
    try:
        article = extract_article(url)
        sentences_data = await create_processing_data(article)
        
        return {
            "article": article,
            "sentences": sentences_data
        }
    except Exception as e:
        logger.error(f"Error in process_article: {e}")
        raise