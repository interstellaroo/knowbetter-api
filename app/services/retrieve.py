from app.core.config import settings
from app.schemas.retrive import EvidenceRetrievalResult, EvidenceResult, SearchResult
from app.schemas.article import DecompositionResult
import logging
import httpx
from datetime import datetime
from typing import List
import logging

logger = logging.getLogger(__name__)

API_KEY = settings.google_search_api_key
ENGINE_ID = settings.google_search_engine_id


async def retrieve_evidence(domain: str, query: str) -> EvidenceResult:
    base_url = "https://www.googleapis.com/customsearch/v1"
    search_query = f"{query} -site:{domain}"
    params = {"key": API_KEY, "cx": ENGINE_ID, "q": search_query, "num": 3}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get("items", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        link=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        displayLink=item.get("displayLink", ""),
                    )
                )

            logger.info(
                f"Retrieved {len(results)} evidence sources for query: {query[:10]}..."
            )
            return EvidenceResult(
                claim=query, evidence=results, timestamp=datetime.now()
            )
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} for query '{query}': {e}")
        return EvidenceResult(claim=query, evidence=[], timestamp=datetime.now())
    except httpx.RequestError as e:
        logger.error(f"Network error for query '{query}': {e}")
        return EvidenceResult(claim=query, evidence=[], timestamp=datetime.now())
    except Exception as e:
        logger.error(f'Error retrieving evidence for query "{query}": {e}')
        return EvidenceResult(claim=query, evidence=[], timestamp=datetime.now())


async def process_decomposition_evidence(
    decomposition_results: List[DecompositionResult], domain: str
) -> EvidenceRetrievalResult:
    """Process decomposition results and retrieve evidence for each claim.

    Args:
        decomposition_results (List[DecompositionResult]): Results from decompose_sentences
        domain (str): Domain to exclude from search results
    Returns:
        EvidenceRetrievalResult: Contains a list of EvidenceResult objects for all claims
    """
    all_evidence_results = []
    try:
        for decomposition_result in decomposition_results:
            decomposed_claims = decomposition_result.decomposed_claims
            for claim in decomposed_claims:
                evidence_result = await retrieve_evidence(domain, claim)
                all_evidence_results.append(evidence_result)
        logger.info(f"Retrieved evidence for {len(all_evidence_results)} total claims")
        return EvidenceRetrievalResult(
            evidence=all_evidence_results,
            excluded_domain=domain,
            timestamp=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Error processing decomposition evidence: {e}")
        if all_evidence_results:
            logger.warning(
                f"Returning partial results: {len(all_evidence_results)} evidence results"
            )
            return EvidenceRetrievalResult(
                evidence=all_evidence_results,
                excluded_domain=domain,
                timestamp=datetime.now(),
            )
        else:
            return EvidenceRetrievalResult(
                evidence=[], excluded_domain=domain, timestamp=datetime.now()
            )
