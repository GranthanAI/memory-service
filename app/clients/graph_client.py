"""
app/clients/graph_client.py

Graph Service HTTP Client to fetch conversation ancestor lineage/summaries.
"""

import logging
from typing import Any, Dict, List
import httpx

from app.core.config import settings

logger = logging.getLogger("memory_service.clients.graph_client")


class GraphClient:
    """
    HTTP Client interacting with the external Graph Service.
    Retrieves historical context lineage (parent/ancestor conversation summaries).
    """

    def __init__(self, base_url: str = settings.GRAPH_SERVICE_URL):
        self.base_url = base_url
        self.timeout_seconds = settings.GRAPH_SERVICE_TIMEOUT_MS / 1000.0

    async def get_ancestors(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Sends a GET request to the Graph Service to fetch ancestor summaries.
        """
        url = f"{self.base_url}/conversations/{conversation_id}/ancestors"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                # Expecting a list of ancestor summaries: [{"conversation_id": "...", "summary": "..."}]
                return response.json()
        except Exception as e:
            logger.error(f"Graph Service HTTP request failed for {conversation_id} at {url}: {e}")
            raise e
