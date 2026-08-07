"""
app/services/llm_service.py

Facade Application Service orchestrating internal LLM requests,
prompt building, token limits, and output parsing.
"""

import json
import logging
import re
import time
from typing import List

from app.core.config import settings
from app.core.metrics import LLM_LATENCY, LLM_REQUESTS, LLM_TOKENS
from app.managers.llm_manager import LLMManager
from app.prompts.fact_prompt import FACT_SYSTEM_PROMPT, FACT_USER_PROMPT_TEMPLATE
from app.prompts.summary_prompt import (
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT_TEMPLATE,
)
from app.schemas.llm import (
    ExtractedFact,
    FactExtractRequest,
    FactExtractResponse,
    SummarizeRequest,
    SummarizeResponse,
)

logger = logging.getLogger("memory_service.services.llm_service")


class LLMService:
    """
    Facade Application Service orchestrating internal LLM requests,
    prompt building, token limits, and output parsing.
    """

    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        # Pattern matches category:importance:statement
        # E.g. "preferences:0.9:Likes coding" or "habits:0.8:Wakes up early"
        self._fact_pattern = re.compile(
            r"^\s*([\w\-]+)\s*:\s*(0?\.\d+|1\.0|1)\s*:\s*(.+)$"
        )

    async def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        """
        Builds the summary prompt, calls LLMManager, and returns the updated summary.
        """
        # Formulate new messages representation
        new_messages_list = [
            {"role": msg.role, "content": msg.content}
            for msg in request.new_messages
        ]
        new_messages_json = json.dumps(new_messages_list, indent=2)

        # Build prompt
        user_content = SUMMARY_USER_PROMPT_TEMPLATE.format(
            previous_summary=request.previous_summary or "No previous summary exists.",
            new_messages_json=new_messages_json,
        )

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        logger.info("Executing conversation summarization through LLMManager...")
        start_time = time.monotonic()
        try:
            summary_raw = await self.llm_manager.generate_with_retry(messages)
            latency = time.monotonic() - start_time
            
            # Record metrics
            LLM_LATENCY.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="summarize").observe(latency)
            LLM_REQUESTS.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="summarize", status="success").inc()
            
            prompt_tokens_est = sum(len(msg["content"]) for msg in messages) // 4
            completion_tokens_est = len(summary_raw) // 4
            LLM_TOKENS.labels(type="prompt").inc(prompt_tokens_est)
            LLM_TOKENS.labels(type="completion").inc(completion_tokens_est)
            
            logger.info(f"Summarization succeeded. Latency: {latency:.3f}s. Est prompt tokens: {prompt_tokens_est}, completion tokens: {completion_tokens_est}")
        except Exception as e:
            latency = time.monotonic() - start_time
            LLM_REQUESTS.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="summarize", status="failure").inc()
            logger.error(f"Summarization failed. Latency: {latency:.3f}s. Error: {e}")
            raise e

        summary_clean = summary_raw.strip()
        return SummarizeResponse(summary=summary_clean)

    async def extract_facts(self, request: FactExtractRequest) -> FactExtractResponse:
        """
        Builds the fact extraction prompt, calls LLMManager, parses lines, and returns facts.
        """
        user_content = FACT_USER_PROMPT_TEMPLATE.format(summary=request.summary)

        messages = [
            {"role": "system", "content": FACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        logger.info("Executing fact extraction through LLMManager...")
        start_time = time.monotonic()
        try:
            raw_output = await self.llm_manager.generate_with_retry(messages)
            latency = time.monotonic() - start_time
            
            # Record metrics
            LLM_LATENCY.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="extract_facts").observe(latency)
            LLM_REQUESTS.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="extract_facts", status="success").inc()
            
            prompt_tokens_est = sum(len(msg["content"]) for msg in messages) // 4
            completion_tokens_est = len(raw_output) // 4
            LLM_TOKENS.labels(type="prompt").inc(prompt_tokens_est)
            LLM_TOKENS.labels(type="completion").inc(completion_tokens_est)
            
            logger.info(f"Fact extraction succeeded. Latency: {latency:.3f}s. Est prompt tokens: {prompt_tokens_est}, completion tokens: {completion_tokens_est}")
        except Exception as e:
            latency = time.monotonic() - start_time
            LLM_REQUESTS.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="extract_facts", status="failure").inc()
            logger.error(f"Fact extraction failed. Latency: {latency:.3f}s. Error: {e}")
            raise e

        # Parse facts
        parsed_facts = self._parse_facts(raw_output)

        facts_dto = [
            ExtractedFact(
                category=f["category"],
                importance=f["importance"],
                statement=f["statement"],
            )
            for f in parsed_facts
        ]

        return FactExtractResponse(facts=facts_dto)

    def _parse_facts(self, text: str) -> List[dict]:
        """
        Splits LLM completion output, sanitizes markdown lists,
        and matches the fact pattern.
        """
        facts = []
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            # Strip common markdown list symbols
            line = re.sub(r"^[\-\*\+]\s+", "", line)
            line = re.sub(r"^\d+\.\s+", "", line)
            line = line.strip()

            match = self._fact_pattern.match(line)
            if match:
                category = match.group(1).lower().strip()
                try:
                    importance = float(match.group(2))
                except ValueError:
                    continue
                statement = match.group(3).strip()
                facts.append(
                    {
                        "category": category,
                        "importance": importance,
                        "statement": statement,
                    }
                )
        return facts
