import pytest
from app.core.config import settings

@pytest.fixture(scope="session", autouse=True)
def force_mock_llm_provider():
    """Forces the LLM Provider to mock during test execution to prevent real API calls and boot-time errors."""
    settings.LLM_PROVIDER = "mock"


@pytest.fixture(autouse=True)
def _no_shared_groq_budget(monkeypatch):
    """Unit tests never touch the shared dev Redis: the Groq budget is an in-process one with no ceiling unless a test sets its own."""
    from app.clients import groq_budget as gb

    monkeypatch.setattr(gb, "_BUDGET", gb.GroqBudget(None, tpm_limit=0))
