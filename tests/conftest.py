import pytest
from app.core.config import settings

@pytest.fixture(scope="session", autouse=True)
def force_mock_llm_provider():
    """Forces the LLM Provider to mock during test execution to prevent real API calls and boot-time errors."""
    settings.LLM_PROVIDER = "mock"
