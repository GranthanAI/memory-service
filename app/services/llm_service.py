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
from app.prompts.anchor_prompt import ANCHOR_SYSTEM_PROMPT, ANCHOR_USER_PROMPT_TEMPLATE
from app.prompts.fact_judge_prompt import FACT_JUDGE_SYSTEM_PROMPT, FACT_JUDGE_USER_PROMPT_TEMPLATE
from app.prompts.summary_prompt import (
    SUMMARY_CONSOLIDATE_SYSTEM_PROMPT,
    SUMMARY_CONSOLIDATE_USER_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT_TEMPLATE,
)
from app.schemas.llm import (
    ConsolidateSummaryRequest,
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

    async def extract_anchors(self, messages: List[dict], open_items: List[str]) -> tuple:
        """(anchors, done_texts) for the judgement-based anchor kinds. Raises on an unusable answer (callers continue with
        the deterministic anchors only)."""
        from app.services.anchors import parse_llm_anchors
        listing = "\n".join(f"{i}: {m.get('role')}: {str(m.get('content') or '')[:1500]}" for i, m in enumerate(messages))
        prompt = [
            {"role": "system", "content": ANCHOR_SYSTEM_PROMPT},
            {"role": "user", "content": ANCHOR_USER_PROMPT_TEMPLATE.format(
                open_items="\n".join(f"- {t}" for t in open_items) or "(none)", messages=listing)},
        ]
        return parse_llm_anchors(await self.llm_manager.generate_with_retry(prompt), messages)

    async def judge_fact_relation(self, new_statement: str, existing: List[str]) -> dict:
        """Explicit LLM judgement of how a new statement relates to existing memories: {"relation": same | correction |
        different, "index": int | None}. Raises on an unusable answer so the caller can fall back to the non-destructive path."""
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(existing))
        messages = [
            {"role": "system", "content": FACT_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": FACT_JUDGE_USER_PROMPT_TEMPLATE.format(new_statement=new_statement, existing=listing)},
        ]
        raw = await self.llm_manager.generate_with_retry(messages)
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            raise ValueError("fact judge returned no JSON")
        data = json.loads(match.group(0))
        relation = str(data.get("relation") or "").strip().lower()
        if relation not in ("same", "correction", "different"):
            raise ValueError("fact judge returned an unknown relation")
        index = data.get("index")
        if relation != "different":
            if not isinstance(index, int) or not 0 <= index < len(existing):
                raise ValueError("fact judge returned an invalid index")
        return {"relation": relation, "index": index if relation != "different" else None}

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
            feedback_block=(
                "\nREVIEWER FEEDBACK on your previous attempt (fix these, change nothing else):\n" + request.feedback + "\n"
                if request.feedback else ""
            ),
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

    async def consolidate_summary(self, request: ConsolidateSummaryRequest) -> SummarizeResponse:
        """Rewrites an existing summary shorter, through the same provider path as summarize(). Raises on failure: the
        caller (SummaryService) owns the fallback, so the token limit can never be bypassed."""
        target = request.target_tokens
        messages = [
            {"role": "system", "content": SUMMARY_CONSOLIDATE_SYSTEM_PROMPT.format(
                target_tokens=target, target_words=max(1, int(target * 0.7)))},
            {"role": "user", "content": SUMMARY_CONSOLIDATE_USER_PROMPT_TEMPLATE.format(
                summary=request.summary, target_tokens=target)},
        ]
        start_time = time.monotonic()
        try:
            raw = await self.llm_manager.generate_with_retry(messages, max_tokens=int(target * 1.4) + 64)
            LLM_LATENCY.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="consolidate_summary").observe(
                time.monotonic() - start_time)
            LLM_REQUESTS.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="consolidate_summary", status="success").inc()
        except Exception:
            LLM_REQUESTS.labels(provider=settings.LLM_PROVIDER, model=settings.LLM_MODEL, action="consolidate_summary", status="failure").inc()
            raise
        return SummarizeResponse(summary=(raw or "").strip())

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
