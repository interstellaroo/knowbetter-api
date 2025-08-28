import ollama
import asyncio
from typing import List
from app.core.prompts import SelectionPrompts, DisambiguationPrompts, DecompositionPrompts
from app.schemas.article import SplittingData, SelectionResult, DisambiguationResult, DecompositionResult

client = ollama.AsyncClient()


async def _select_sentence(
    sentence: str, context: str, paragraph: str
) -> SelectionResult:
    try:
        prompt = SelectionPrompts.get_prompt(sentence, context, paragraph)

        response = await client.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": SelectionPrompts.GUIDELINES},
                {"role": "user", "content": prompt},
            ],
            format=SelectionResult.model_json_schema(),
            stream=False,
            options={"temperature": 0.0},
        )

        result = SelectionResult.model_validate_json(response.message.content)
        return result
    except Exception as e:
        return SelectionResult(
            original_sentence=sentence,
            verification_label="not_verifiable",
            rewritten_sentence=sentence,
        )

async def _disambiguate_sentence(selection_result: SelectionResult, sentence_data) -> DisambiguationResult:
    try:
        prompt = DisambiguationPrompts.get_prompt(
            sentence=selection_result.rewritten_sentence,
            context=sentence_data.context,
            paragraph=sentence_data.paragraph
        )
        
        response = await client.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": DisambiguationPrompts.GUIDELINES},
                {"role": "user", "content": prompt},
            ],
            format=DisambiguationResult.model_json_schema(),
            stream=False,
            options={"temperature": 0.0},
        )
        
        result = DisambiguationResult.model_validate_json(response.message.content)
        return result
        
    except Exception as e:
        return DisambiguationResult(
            original_sentence=selection_result.original_sentence,
            disambiguated_sentence=selection_result.rewritten_sentence,
            reason=str(e)
        )

async def _decompose_sentence(disambiguation_result: DisambiguationResult) -> DecompositionResult:
    try:
        prompt = DecompositionPrompts.get_prompt(claim=disambiguation_result.disambiguated_sentence)
        
        response = await client.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": DecompositionPrompts.GUIDELINES},
                {"role": "user", "content": prompt},
            ],
            format=DecompositionResult.model_json_schema(),
            stream=False,
            options={"temperature": 0.0},
        )
        
        result = DecompositionResult.model_validate_json(response.message.content)
        return result
    except Exception as e:
        return DecompositionResult(
            original_claim=disambiguation_result.disambiguated_sentence,
            decomposed_claims=[],
        )

async def decompose_sentences(disambiguation_results: List[DisambiguationResult]) -> List[DecompositionResult]:
    """
    Process disambiguated sentences and decompose them into individual verifiable claims.

    Args:
        disambiguation_results (List[DisambiguationResult]): The disambiguated sentences to decompose

    Returns:
        List[DecompositionResult]: List of decomposed claims for each sentence
    """
    try:
        tasks = []
        for disambiguation_result in disambiguation_results:
            task = _decompose_sentence(disambiguation_result)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        raise e

async def disambiguate_sentences(selection_results: List[SelectionResult], data: SplittingData) -> List[DisambiguationResult]:
    try:
        tasks = []
        for i, selection_result in enumerate(selection_results):
            if selection_result.verification_label != "not_verifiable":
                sentence_data = data.sentences[i]
                task = _disambiguate_sentence(selection_result, sentence_data)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        return 

async def select_sentences(splitting_data: SplittingData) -> List[SelectionResult]:
    """
    Process all sentences from SplittingData and return a list of SelectionResult objects.

    Args:
        splitting_data (SplittingData): The splitting data containing sentences to process

    Returns:
        List[SelectionResult]: List of processed sentence results
    """
    tasks = []

    for sentence_data in splitting_data.sentences:
        task = _select_sentence(
            sentence=sentence_data.sentence,
            context=sentence_data.context,
            paragraph=sentence_data.paragraph,
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            sentence_data = splitting_data.sentences[i]
            fallback_result = SelectionResult(
                original_sentence=sentence_data.sentence,
                verification_label="not_verifiable",
                rewritten_sentence=sentence_data.sentence,
            )
            processed_results.append(fallback_result)
        else:
            processed_results.append(result)

    return processed_results