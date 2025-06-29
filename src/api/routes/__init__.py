"""API routes for parakeetv2API."""

from src.api.routes.models import router as models_router
from src.api.routes.transcription import router as transcription_router

__all__ = ["models_router", "transcription_router"]