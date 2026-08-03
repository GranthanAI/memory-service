"""
app/lifespan.py

FastAPI Lifespan management for startup/shutdown actions,
integrating database pools initialization, migrations execution, and DI container setups.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.container import Container
from app.db.session import initialize_db_sessions, close_db_sessions

logger = logging.getLogger("memory_service.lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    Bootstraps all database connection pools, executes schema migrations,
    and initializes the dependency injection container.
    """
    logger.info("=== Starting application bootstrap lifespans ===")
    
    # 1. Initialize databases (Cassandra, Redis, Milvus) and run migrations
    await initialize_db_sessions()
    
    # 2. Build and bootstrap the DI Container
    container = Container()
    await container.init_resources()
    
    # Store container in app state for access in endpoints & worker threads
    app.state.container = container
    
    logger.info("=== Application bootstrap completed successfully ===")
    
    yield
    
    logger.info("=== Stopping application and tearing down resources ===")
    
    # 3. Shutdown connection pools and container clients
    await container.shutdown_resources()
    await close_db_sessions()
    
    logger.info("=== Application shutdown complete ===")
