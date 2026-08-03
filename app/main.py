"""
app/main.py

FastAPI Application Entry Point.
"""

from fastapi import FastAPI
from app.lifespan import lifespan

app = FastAPI(
    title="GraphGPT Memory Service",
    description="Derived Cognitive AI Memory Engine",
    version="4.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():
    """Simple HTTP health check endpoint."""
    return {"status": "healthy"}
