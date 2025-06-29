"""Main entry point for parakeetv2API."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from src.api import models_router, transcription_router
from src.config import settings
from src.core.exceptions import ParakeetAPIException
from src.core.model_manager import model_manager

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if settings.log_format == "text"
    else '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting parakeetv2API server")
    
    # Set CUDA device if specified
    if settings.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
        logger.info(f"Set CUDA_VISIBLE_DEVICES={settings.cuda_visible_devices}")
    
    # Load ASR model
    try:
        logger.info("Loading ASR model...")
        model_manager.load_model()
        logger.info("ASR model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {str(e)}")
        raise
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down parakeetv2API server")
    
    # Cleanup resources
    try:
        model_manager.unload_model()
    except Exception as e:
        logger.error(f"Error during model cleanup: {str(e)}")
    
    logger.info("Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="parakeetv2API",
    description="OpenAI-compatible API server for NVIDIA's parakeet-tdt-0.6b-v2 ASR model",
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ParakeetAPIException)
async def parakeet_exception_handler(request: Request, exc: ParakeetAPIException):
    """Handle custom parakeet API exceptions."""
    logger.warning(f"ParakeetAPIException: {exc.message}")
    return ORJSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": "api_error",
                "details": exc.details,
            }
        }
    )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    # If detail is already in the correct format, return as-is
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return ORJSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # Otherwise, wrap in standard error format
    return ORJSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "http_error",
                "code": str(exc.status_code),
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return ORJSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An unexpected error occurred",
                "type": "server_error", 
                "code": "internal_error",
            }
        }
    )


# Include API routes
app.include_router(transcription_router, prefix=settings.api_prefix)
app.include_router(models_router, prefix=settings.api_prefix)


@app.get("/", tags=["Health"])
async def root() -> Dict[str, Any]:
    """Root endpoint for health check."""
    return {
        "status": "healthy",
        "service": "parakeetv2API",
        "version": "0.1.0",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
    }


def main():
    """Run the application."""
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()