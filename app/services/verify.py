import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime
import httpx
from newspaper import Article
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.schemas.retrive import EvidenceRetrievalResult, SearchResult
from app.schemas.verify import NLIResult, VerificationBatch

logger = logging.getLogger(__name__)

class NLIModel:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name  # Fixed: store model name properly
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        logger.info(f"NLI model initialized: {model_name} on {self.device}")

    async def load(self):
        if self._loaded:
            return

        try:
            logger.info(f"Loading NLI model: {self.model_name}")  # Fixed: f-string
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self._loaded = True
            logger.info("NLI model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {str(e)}")
            raise e

    def predict_entailment(
        self, premise: str, hypothesis: str
    ):  # Fixed: method belongs to class
        """
        Predict NLI relationship between premise (evidence) and hypothesis (claim).
        """
        if not self._loaded:
            raise RuntimeError("NLI model is not loaded")

        try:
            # Tokenize premise and hypothesis
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)[0]  # Get first (and only) example

            # BART-MNLI labels: [contradiction, neutral, entailment]
            labels = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]

            # Get scores for each label
            scores = {labels[i]: float(probs[i]) for i in range(len(labels))}

            # Get predicted label and confidence
            predicted_idx = torch.argmax(probs).item()
            verdict = labels[predicted_idx]
            confidence = float(probs[predicted_idx])

            return {"verdict": verdict, "confidence": confidence, "scores": scores}

        except Exception as e:
            logger.error(f"NLI prediction failed: {e}")
            return {
                "verdict": "NEUTRAL",
                "confidence": 0.0,
                "scores": {"CONTRADICTION": 0.0, "NEUTRAL": 1.0, "ENTAILMENT": 0.0},
            }


# Global NLI model instance
nli_model = NLIModel("facebook/bart-large-mnli")


async def extract_evidence_summary(evidence: SearchResult) -> str:
    """Extract summary from evidence URL using newspaper4k."""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(evidence.link)
            response.raise_for_status()

            article = Article(evidence.link)
            article.set_html(response.text)
            article.parse()

            # Get article summary - prefer parsed summary, fallback to text excerpt
            if article.summary and len(article.summary.strip()) > 20:
                summary = article.summary
            elif article.text:
                # Take first 3 sentences as summary
                sentences = article.text.split(". ")[:3]
                summary = ". ".join(sentences)
                if not summary.endswith("."):
                    summary += "."
            else:
                # Fallback to search snippet
                summary = evidence.snippet

            # Limit length for NLI model
            if len(summary) > 800:
                summary = summary[:800] + "..."

            logger.debug(
                f"Extracted summary from {evidence.displayLink}: {len(summary)} chars"
            )
            return summary.strip()

    except Exception as e:
        logger.warning(f"Failed to extract summary from {evidence.link}: {e}")
        # Fallback to search snippet
        return evidence.snippet


async def verify_claim_against_evidence(
    claim: str, evidence: SearchResult
) -> NLIResult:
    """Verify a single claim against a single piece of evidence using NLI."""
    try:
        # Ensure model is loaded
        if not nli_model._loaded:
            await nli_model.load()

        # Extract evidence summary
        evidence_summary = await extract_evidence_summary(evidence)

        if not evidence_summary or len(evidence_summary.strip()) < 10:
            return NLIResult(
                claim=claim,
                evidence_summary="No content available",
                evidence_source=evidence.displayLink,
                verdict="NEUTRAL",
                confidence=0.0,
                scores={"ENTAILMENT": 0.0, "CONTRADICTION": 0.0, "NEUTRAL": 1.0},
                timestamp=datetime.now(),
            )

        # Run NLI: premise=evidence_summary, hypothesis=claim
        prediction = nli_model.predict_entailment(evidence_summary, claim)

        result = NLIResult(
            claim=claim,
            evidence_summary=evidence_summary,
            evidence_source=evidence.displayLink,
            verdict=prediction["verdict"],
            confidence=prediction["confidence"],
            scores=prediction["scores"],
            timestamp=datetime.now(),
        )

        logger.info(
            f"NLI: {result.verdict} ({result.confidence:.3f}) - {claim[:50]}... vs {evidence.displayLink}"
        )
        return result

    except Exception as e:
        logger.error(f"Error verifying claim against evidence: {e}")
        return NLIResult(
            claim=claim,
            evidence_summary="Error processing evidence",
            evidence_source=evidence.displayLink if evidence else "",
            verdict="NEUTRAL",
            confidence=0.0,
            scores={"ENTAILMENT": 0.0, "CONTRADICTION": 0.0, "NEUTRAL": 1.0},
            timestamp=datetime.now(),
        )


async def process_evidence_verification(
    evidence_results: List[EvidenceRetrievalResult],
) -> VerificationBatch:
    """
    Main verification function - process evidence retrieval results and verify claims.

    Args:
        evidence_results: List of evidence retrieval results from your pipeline

    Returns:
        VerificationBatch: Complete verification results
    """
    start_time = datetime.now()

    try:
        # Ensure model is loaded once at the start
        if not nli_model._loaded:
            await nli_model.load()

        verification_tasks = []

        # Iterate through each EvidenceRetrievalResult and its contained EvidenceResult objects
        for retrieval_result in evidence_results:
            if hasattr(retrieval_result, "evidence") and retrieval_result.evidence:
                for evidence_result in retrieval_result.evidence:
                    if hasattr(evidence_result, "claim") and evidence_result.claim:
                        claim = evidence_result.claim
                        if evidence_result.evidence:
                            for evidence in evidence_result.evidence[
                                :3
                            ]:  # Top 3 evidence per claim
                                task = verify_claim_against_evidence(claim, evidence)
                                verification_tasks.append(task)
                        else:
                            # No evidence found for this claim
                            no_evidence_result = NLIResult(
                                claim=claim,
                                evidence_summary="",
                                evidence_source="No evidence found",
                                verdict="NEUTRAL",
                                confidence=0.0,
                                scores={
                                    "ENTAILMENT": 0.0,
                                    "CONTRADICTION": 0.0,
                                    "NEUTRAL": 1.0,
                                },
                                timestamp=datetime.now(),
                            )
                            verification_tasks.append(
                                asyncio.create_task(
                                    asyncio.sleep(0, result=no_evidence_result)
                                )
                            )
                    else:
                        logger.warning("Evidence result missing claim - skipping")
            else:
                logger.warning(
                    "EvidenceRetrievalResult missing evidence list - skipping"
                )

        # Execute all verification tasks
        valid_results = []
        if verification_tasks:
            results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            # Filter out exceptions and get valid results
            for result in results:
                if isinstance(result, NLIResult):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Verification task failed: {result}")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        logger.info(
            f"Verification completed: {len(valid_results)} claim-evidence pairs in {processing_time:.2f}s"
        )

        return VerificationBatch(
            results=valid_results,
            total_processed=len(valid_results),
            processing_time=processing_time,
            timestamp=end_time,
        )

    except Exception as e:
        logger.error(f"Error in evidence verification: {e}")
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        return VerificationBatch(
            results=[],
            total_processed=0,
            processing_time=processing_time,
            timestamp=end_time,
        )


# Note: Summary and aggregation functions have been moved to article.py 
# as part of the FinalValidationResult schema integration
