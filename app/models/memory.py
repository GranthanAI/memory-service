"""
app/models/memory.py

Domain models and enums representing memory states.
"""

from enum import Enum


class MemoryState(str, Enum):
    """
    State machine states tracking the pipeline execution phase of a conversation's memory.
    """
    ACTIVE = "ACTIVE"
    SUMMARY_PENDING = "SUMMARY_PENDING"
    SUMMARIZING = "SUMMARIZING"
    FACT_PENDING = "FACT_PENDING"
    EXTRACTING_FACTS = "EXTRACTING_FACTS"
    EMBEDDING_PENDING = "EMBEDDING_PENDING"
    READY = "READY"
    FAILED = "FAILED"
