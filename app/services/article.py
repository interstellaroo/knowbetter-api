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
from app.schemas.verify import FinalValidationResult, ProcessingStatistics, PipelineStepResults, ArticleSummary, VerdictSummary 
import re
from urllib.parse import urlparse
import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import VerificationRun, Sentence, Selection, Disambiguation, Decomposition, Claim, Evidence, Verification

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
        raise Exception(f"Article extraction failed. The website might be protected.")

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
        raise Exception(f"Article processing failed.")

async def run_article_processing(url: str, db: Optional[AsyncSession] = None) -> FinalValidationResult:
    processing_start = datetime.now()
    
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
        
        evidence_results = None
        verification_results = None
        claims_with_evidence = 0
        verified_claims = 0
        
        if decomposition_results and total_claims > 0:
            evidence_results = await process_decomposition_evidence(decomposition_results, article_data.article.domain)
            claims_with_evidence = len([e for e in evidence_results.evidence if e.evidence])
            logger.info(f"Evidence retrieval complete: {claims_with_evidence}/{len(evidence_results.evidence)} claims have evidence")
            
            # Verify claims against evidence
            if evidence_results.evidence:
                verification_results = await process_evidence_verification([evidence_results])
                verified_claims = verification_results.total_processed
                logger.info(f"Verification complete: {verified_claims} claims verified in {verification_results.processing_time:.2f}s")
            else:
                logger.warning("No evidence found for verification")
        else:
            logger.warning("No claims found for evidence retrieval")

        processing_end = datetime.now()
        total_processing_time = (processing_end - processing_start).total_seconds()
        
        final_result = _build_final_validation_result(
            article_data=article_data,
            selection_results=selection_results,
            disambiguation_results=disambiguation_results,
            decomposition_results=decomposition_results,
            evidence_results=evidence_results,
            verification_results=verification_results,
            processing_start=processing_start,
            processing_end=processing_end,
            total_processing_time=total_processing_time,
            verifiable_count=verifiable_count,
            total_claims=total_claims,
            claims_with_evidence=claims_with_evidence,
            verified_claims=verified_claims
        )
        if db:
            await _save_to_database(db, final_result)
            
        return final_result
        
    except ArticleBinaryDataException:
        logger.warning("Binary data detected during article processing for %s", url)
        raise
    except Exception as e:
        logger.error(f"Error processing article: {str(e)}")
        raise

def _build_final_validation_result(
    article_data: ArticleExtractionData,
    selection_results: List,
    disambiguation_results: List,
    decomposition_results: List,
    evidence_results: Optional[Any],
    verification_results: Optional[Any],
    processing_start: datetime,
    processing_end: datetime,
    total_processing_time: float,
    verifiable_count: int,
    total_claims: int,
    claims_with_evidence: int,
    verified_claims: int
) -> FinalValidationResult:
    """Build the final validation result from all pipeline components."""
    
    # Create article summary
    article_summary = ArticleSummary(
        url=article_data.article.url,
        title=article_data.article.title,
        domain=article_data.article.domain,
        authors=article_data.article.authors,
        publish_date=article_data.article.publish_date
    )
    
    # Create processing statistics
    statistics = ProcessingStatistics(
        total_sentences=article_data.data.count,
        verifiable_sentences=verifiable_count,
        disambiguated_sentences=len(disambiguation_results),
        total_claims=total_claims,
        claims_with_evidence=claims_with_evidence,
        verified_claims=verified_claims,
        processing_time_seconds=total_processing_time
    )
    
    # Create pipeline results
    pipeline_results = PipelineStepResults(
        selection_results=selection_results,
        disambiguation_results=disambiguation_results,
        decomposition_results=decomposition_results,
        evidence_results=evidence_results,
        verification_results=verification_results
    )
    
    # Calculate verdict summary
    verdict_summary = _calculate_verdict_summary(verification_results, total_claims, claims_with_evidence)
    
    return FinalValidationResult(
        article=article_summary,
        statistics=statistics,
        pipeline_results=pipeline_results,
        verdict_summary=verdict_summary,
        processing_start=processing_start,
        processing_end=processing_end,
        total_processing_time=total_processing_time
    )

def _calculate_verdict_summary(verification_results: Optional[Any], total_claims: int, claims_with_evidence: int) -> VerdictSummary:
    """Calculate the verdict summary from verification results."""
    if not verification_results or not verification_results.results:
        return VerdictSummary(
            entailed_claims=0,
            contradicted_claims=0,
            neutral_claims=0,
            unsupported_claims=total_claims,
            overall_credibility_score=0.0
        )
    
    # Count verdicts
    entailed = sum(1 for r in verification_results.results if r.verdict == "ENTAILMENT")
    contradicted = sum(1 for r in verification_results.results if r.verdict == "CONTRADICTION")
    neutral = sum(1 for r in verification_results.results if r.verdict == "NEUTRAL")
    unsupported = total_claims - claims_with_evidence
    
    # Calculate overall credibility score (0-1 scale)
    if total_claims == 0:
        credibility_score = 0.0
    else:
        # Weight: ENTAILMENT = +1, NEUTRAL = 0, CONTRADICTION = -1, UNSUPPORTED = -0.5
        weighted_score = (entailed * 1.0) + (neutral * 0.0) + (contradicted * -1.0) + (unsupported * -0.5)
        max_possible_score = total_claims * 1.0
        min_possible_score = total_claims * -1.0
        
        # Normalize to 0-1 scale
        if max_possible_score == min_possible_score:
            credibility_score = 0.5
        else:
            credibility_score = (weighted_score - min_possible_score) / (max_possible_score - min_possible_score)
    
    return VerdictSummary(
        entailed_claims=entailed,
        contradicted_claims=contradicted,
        neutral_claims=neutral,
        unsupported_claims=unsupported,
        overall_credibility_score=round(credibility_score, 3)
    )

async def _save_to_database(db: AsyncSession, result: FinalValidationResult):
    """Save verification run to database with normalized structure."""
    try:
        # 1. Create main verification run
        verification_run = VerificationRun(
            url=result.article.url,
            title=result.article.title,
            domain=result.article.domain,
            authors=result.article.authors,
            publish_date=result.article.publish_date,
            processing_start=result.processing_start,
            processing_end=result.processing_end,
            total_processing_time=result.total_processing_time,
            status="completed",
            overall_credibility_score=result.verdict_summary.overall_credibility_score
        )
        
        db.add(verification_run)
        await db.flush()  # Get ID without committing
        run_id = verification_run.id
        
        # 2. Save sentences
        if result.pipeline_results.selection_results:
            for i, sentence_data in enumerate(result.pipeline_results.selection_results):
                sentence = Sentence(
                    run_id=run_id,
                    index=i,
                    sentence=sentence_data.original_sentence,
                    context=None,
                    paragraph=None
                )
                db.add(sentence)
        
        # 3. Save selections
        if result.pipeline_results.selection_results:
            for selection_data in result.pipeline_results.selection_results:
                selection = Selection(
                    run_id=run_id,
                    original_sentence=selection_data.original_sentence,
                    verification_label=selection_data.verification_label,
                    rewritten_sentence=selection_data.rewritten_sentence
                )
                db.add(selection)
        
        # 4. Save disambiguations
        if result.pipeline_results.disambiguation_results:
            for disambig_data in result.pipeline_results.disambiguation_results:
                disambiguation = Disambiguation(
                    run_id=run_id,
                    original_sentence=disambig_data.original_sentence,
                    disambiguated_sentence=disambig_data.disambiguated_sentence,
                    reason=disambig_data.reason
                )
                db.add(disambiguation)
        
        # 5. Save decompositions and claims
        claim_id_map = {}
        
        if result.pipeline_results.decomposition_results:
            for decomp_data in result.pipeline_results.decomposition_results:
                decomposition = Decomposition(
                    run_id=run_id,
                    original_claim=decomp_data.original_claim,
                    decomposed_claims=decomp_data.decomposed_claims
                )
                db.add(decomposition)
                await db.flush()
                
                # Save individual claims
                for i, claim_text in enumerate(decomp_data.decomposed_claims):
                    claim = Claim(
                        run_id=run_id,
                        decomposition_id=decomposition.id,
                        claim_text=claim_text,
                        claim_index=i
                    )
                    db.add(claim)
                    await db.flush()
                    claim_id_map[claim_text] = claim.id
        
        # 6. Save evidence
        if result.pipeline_results.evidence_results and result.pipeline_results.evidence_results.evidence:
            for evidence_result in result.pipeline_results.evidence_results.evidence:
                claim_text = evidence_result.claim
                claim_id = claim_id_map.get(claim_text)
                
                if claim_id:
                    for evidence_item in evidence_result.evidence:
                        evidence = Evidence(
                            run_id=run_id,
                            claim_id=claim_id,
                            title=evidence_item.title,
                            link=evidence_item.link,
                            snippet=evidence_item.snippet,
                            display_link=evidence_item.displayLink,
                            retrieved_at=evidence_result.timestamp
                        )
                        db.add(evidence)
        
        # 7. Save verifications
        if result.pipeline_results.verification_results and result.pipeline_results.verification_results.results:
            for verification_result in result.pipeline_results.verification_results.results:
                claim_text = verification_result.claim
                claim_id = claim_id_map.get(claim_text)
                
                if claim_id:
                    verification = Verification(
                        run_id=run_id,
                        claim_id=claim_id,
                        claim_text=verification_result.claim,
                        evidence_summary=verification_result.evidence_summary,
                        evidence_source=verification_result.evidence_source,
                        verdict=verification_result.verdict,
                        confidence=verification_result.confidence,
                        scores=verification_result.scores,
                        timestamp=verification_result.timestamp
                    )
                    db.add(verification)
        
        # Commit all changes
        await db.commit()
        logger.info(f"Saved verification run {verification_run.id} to database")
        
    except Exception as e:
        logger.error(f"Failed to save to database: {e}")
        await db.rollback()
        raise

async def get_article_summary(url: str):
    try:
        article = Article(url)
        article.download()
        article.nlp()
        return article.summary()
    except Exception as e:
        logger.error(f"Error getting article summary: {str(e)}")
        return None