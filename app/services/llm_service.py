"""
app/services/llm_service.py

Facade Application Service orchestrating internal LLM requests,
prompt building, token limits, and output parsing.
"""

import json
import logging
import re
from typing import List

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
        summary_raw = await self.llm_manager.generate_with_retry(messages)
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
        raw_output = await self.llm_manager.generate_with_retry(messages)

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
