import spacy
from newspaper import Article
from typing import List, Dict, Any
from bs4 import BeautifulSoup
import trafilatura
from app.schemas.article import ArticleData, SentenceData, SplittingData, ArticleExtractionData
from newspaper.exceptions import ArticleBinaryDataException
from app.services.llm import select_sentences, disambiguate_sentences, decompose_sentences
from app.services.retrieve import process_decomposition_evidence
from app.services.verify import process_evidence_verification 
import re
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)
nlp = spacy.load("en_core_web_sm")

def _normalize_whitespace(text: str) -> str:
    """Helper function for normalizing whitespace in the text.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    if not text:
        return ""
    
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()

def get_paragraphs(url: str, html: str) -> List[str]:
    """Extracting paragraphs from the article HTML content using trafilatura. Can fallback to downloading the article straight from the
    provided URL address. 

    Args:
        url (str): The URL of the article.
        html (str): The HTML content of the article.

    Returns:
        List[str]: A list of extracted paragraphs.
    Raises:
        ValueError: If no content is downloaded from the URL.
        ValueError: If no TEI content is extracted from the URL.
        Exception: If an error occurs while processing the article.

    Returns:
        List[str]: A list of extracted paragraphs.
    """
    try:
        downloaded = html if html else trafilatura.fetch_url(url)
        if not downloaded:
            logger.warning("Failed to download article content from %s", url)
            raise ValueError("No content downloaded from the URL")

        tei = trafilatura.extract(
            downloaded,
            output_format="xml",
            include_comments=False,
            include_images=False,
            include_tables=False,
            with_metadata=False,
        )
        
        if not tei:
            logger.warning("No TEI content extracted from %s", url)
            raise ValueError("No TEI content extracted from the URL")
        
        soup = BeautifulSoup(tei, "xml")
        paragraphs = [_normalize_whitespace(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
        
        return paragraphs

    except Exception as e:
        logger.exception("Error getting paragraphs from %s: %s", url, e)
        raise Exception(f"Failed to get paragraphs: {e}")

def split_sentences(paragraphs: List[str]) -> List[Dict[str, Any]]:
    """Splits paragraphs into sentences and extracts contextual information.

    Args:
        paragraphs (List[str]): A list of paragraphs to split.

    Returns:
        List[Dict[str, Any]]: A list of sentence records with contextual information.
    """
    records: List[Dict[str, Any]] = []
    index = 0
    
    for p_index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        
        doc = nlp(paragraph)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        
        for sent in sents:
            records.append({
                "index": index,
                "sentence": sent,
                "paragraph": paragraph,
                "context": None,
            })
            index += 1
    
    for i in range(len(records)):
        prev_s = records[i - 1]["sentence"] if i > 0 else ""
        cur_s = records[i]["sentence"]
        next_s = records[i + 1]["sentence"] if i < len(records) - 1 else ""
        records[i]["context"] = _normalize_whitespace(f"{prev_s} {cur_s} {next_s}")

    return records
        

def extract_article(url: str) -> ArticleData:
    """Extracts article metadata and content from the given URL.

    Args:
        url (str): The URL of the article.

    Raises:
        Exception: If an error occurs while extracting the article.

    Returns:
        ArticleData: The extracted article data.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        
        paragraphs = get_paragraphs(url, article.html)
        domain = get_domain(url)
        
        result = ArticleData(
            url=url,
            title=article.title,
            authors=article.authors,
            text=article.text,
            publish_date=article.publish_date,
            summary=article.summary,
            paragraphs=paragraphs,
            domain=domain
        )
        logger.info("Successfully extracted article from %s", url)
        return result
    
    except ArticleBinaryDataException as e:
        raise
    except Exception as e:
        raise Exception(f"Article extraction failed: {e}")

def get_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        if not domain:
            raise ValueError("No domain found in the URL")
        if domain.startswith("www."):
            domain = domain[4:]
            
        return domain
    except Exception as e:
        logger.error("Failed to extract domain from URL %s: %s", url, e)
        raise ValueError(f"Domain extraction error: {url}")

async def create_processing_data(article_data: ArticleData) -> SplittingData:
    """Creates processing data for the article by splitting its paragraphs into sentences.

    Args:
        article_data (ArticleData): The article data to process.

    Raises:
        Exception: If an error occurs while creating processing data.

    Returns:
        SplittingData: The created processing data.
    """
    try:
        records = split_sentences(article_data.paragraphs)

        sentences = [
            SentenceData(
                index=rec["index"],
                sentence=rec["sentence"],
                context=rec["context"],
                paragraph=rec["paragraph"],
            )
            for rec in records
        ]

        result = SplittingData(
            sentences=sentences,
            count=len(sentences)
        )

        logger.info("Successfully created processing data for article: %s", article_data.url)
        return result

    except Exception as e:
        logger.exception("Error creating processing data for article: %s", e)
        raise Exception(f"Failed to create processing data: {e}")


async def process_article(url: str) -> ArticleExtractionData:
    """Processes the article at the given URL and extracts relevant data.

    Args:
        url (str): The URL of the article to process.

    Raises:
        ArticleBinaryDataException: If the article contains binary data.
        Exception: If an error occurs while processing the article.

    Returns:
        ArticleExtractionData: The extracted article data.
    """
    try:
        article = extract_article(url)
        splitting_data = await create_processing_data(article)

        return ArticleExtractionData(
            article=article,
            data=splitting_data
        )
    except ArticleBinaryDataException:
        logger.warning("Binary data detected in article from %s", url)
        raise 
    except Exception as e:
        logger.exception("Error processing article from %s: %s", url, e)
        raise Exception(f"Article processing failed")

async def run_article_processing(url: str):
    try:
        article_data = await process_article(url)
        logger.info("Extracted %d sentences from article", article_data.data.count)
        
        selection_results = await select_sentences(article_data.data)
        verifiable_count = sum(1 for r in selection_results if r.verification_label != "not_verifiable")
        logger.info(f"Selection complete: {verifiable_count}/{len(selection_results)} sentences marked as verifiable")
        
        disambiguation_results = await disambiguate_sentences(selection_results, article_data.data)
        logger.info(f"Disambiguation complete: {len(disambiguation_results)} sentences disambiguated")
        
        decomposition_results = await decompose_sentences(disambiguation_results)
        total_claims = sum(len(result.decomposed_claims) for result in decomposition_results)
        logger.info(f"Decomposition complete: {len(decomposition_results)} sentences produced {total_claims} total claims")
        
        if decomposition_results and total_claims > 0:
            evidence_results = await process_decomposition_evidence(decomposition_results, article_data.article.domain)
            logger.info(f"Evidence retrieval complete: {len(evidence_results.evidence)} evidence results")
            
            # Verify claims against evidence
            if evidence_results.evidence:
                verification_results = await process_evidence_verification([evidence_results])
                logger.info(f"Verification complete: {verification_results.total_processed} claims verified in {verification_results.processing_time:.2f}s")
                return verification_results
            else:
                logger.warning("No evidence found for verification")
                return evidence_results
        else:
            logger.warning("No claims found for evidence retrieval")
            evidence_results = []

        return evidence_results
    except ArticleBinaryDataException:
        logger.warning("Binary data detected during article processing for %s", url)
        raise  # Re-raise to preserve exception type for route handler
    except Exception as e:
        logger.error(f"Error processing article: {str(e)}")
        raise Exception(f"Failed to process article: {str(e)}")

async def get_article_summary(url: str):
    try:
        article = Article(url)
        article.download()
        article.nlp()
        return article.summary()
    except Exception as e:
        logger.error(f"Error getting article summary: {str(e)}")
        return None