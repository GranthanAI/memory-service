import logging
from pymilvus import connections, utility

logger = logging.getLogger("memory_service.db.milvus")

def connect_milvus(host: str, port: int, alias: str = "default") -> None:
    """
    Connects to the Milvus standalone or cluster service using pymilvus.
    """
    logger.info(f"Connecting to Milvus at {host}:{port} using alias '{alias}'")
    connections.connect(
        alias=alias,
        host=host,
        port=port
    )

def disconnect_milvus(alias: str = "default") -> None:
    """
    Disconnects the Milvus connection.
    """
    logger.info(f"Disconnecting Milvus alias '{alias}'...")
    try:
        connections.disconnect(alias)
    except Exception as e:
        logger.error(f"Error disconnecting Milvus alias '{alias}': {str(e)}")

def check_milvus_ready() -> bool:
    """
    Checks if Milvus is ready and active by executing a simple utility command.
    """
    try:
        # Check list_collections as a lightweight ping operation
        utility.list_collections()
        return True
    except Exception as e:
        logger.error(f"Milvus connection check failed: {str(e)}")
        return False
