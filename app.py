"""ASGI entrypoint for uvicorn: `uvicorn app:app --host 0.0.0.0 --port 5001`."""

from src.api import app

__all__ = ["app"]
