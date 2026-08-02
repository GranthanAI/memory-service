"""
app/models/summary.py

Conversation summary domain representation.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class SummaryRecord:
    """
    Represents an AI-generated conversation summary at a specific version point.
    """
    conversation_id: str
    summary_text: str
    summary_version: int
    model_name: str
    model_version: str
    generated_at: datetime
