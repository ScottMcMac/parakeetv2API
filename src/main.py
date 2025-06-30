"""Main entry point for parakeetv2API."""

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
from src.api.middleware import RequestTracingMiddleware, PerformanceMonitoringMiddleware
from src.config import settings
from src.core.exceptions import ParakeetAPIException
from src.core.logging import setup_logging, get_logger
from src.core.model_manager import model_manager
from src.core.monitoring import system_monitor

# Setup structured logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info(
        "starting_server",
        host=settings.host,
        port=settings.port,
        environment=settings.environment,
    )
    
    # Set CUDA device if specified
    if settings.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = settings.cuda_visible_devices
        logger.info("cuda_device_set", device=settings.cuda_visible_devices)
    
    # Load ASR model
    try:
        logger.info("loading_model", model_name=settings.model_name)
        model_manager.load_model()
        logger.info("model_loaded_successfully")
    except Exception as e:
        logger.error("model_load_failed", error=str(e), exc_info=True)
        raise
    
    # Start system monitoring
    if settings.environment == "production":
        system_monitor.start_monitoring(interval=60)
    
    logger.info("startup_complete")
    
    yield
    
    # Shutdown
    logger.info("shutting_down_server")
    
    # Stop system monitoring
    system_monitor.stop_monitoring()
    
    # Cleanup resources
    try:
        model_manager.unload_model()
        logger.info("model_unloaded")
    except Exception as e:
        logger.error("model_cleanup_error", error=str(e), exc_info=True)
    
    logger.info("shutdown_complete")


# API documentation
API_DESCRIPTION = """
# parakeetv2API

An OpenAI-compatible API server for NVIDIA's parakeet-tdt-0.6b-v2 automatic speech recognition (ASR) model.

## Overview

This API provides audio transcription capabilities using NVIDIA's parakeet-tdt-0.6b-v2 model, while maintaining compatibility with the OpenAI Audio API specification.

### Key Features

- **OpenAI API Compatible**: Drop-in replacement for OpenAI's transcription endpoints
- **High Performance**: GPU-accelerated inference with NVIDIA's optimized ASR model
- **Multiple Format Support**: Accepts various audio formats (WAV, MP3, FLAC, M4A, OGG, etc.)
- **Automatic Conversion**: Handles format conversion automatically using FFmpeg
- **Production Ready**: Includes health checks, monitoring, and structured logging

### Supported Audio Formats

- **WAV** (.wav) - Recommended format
- **MP3** (.mp3)
- **FLAC** (.flac)
- **M4A** (.m4a)
- **OGG** (.ogg)
- **MP4** (.mp4, .mpeg, .mpga)
- **WebM** (.webm)

### Model Information

All model aliases (`whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `parakeet-tdt-0.6b-v2`) use the same backend: NVIDIA's parakeet-tdt-0.6b-v2 model.

The model expects 16kHz mono audio. Files in other formats are automatically converted.

### Authentication

The API supports optional authentication via Bearer tokens in the Authorization header. If authentication is not configured, the API will accept all requests.

### Rate Limits

Default rate limits:
- 60 requests per minute per client
- Burst allowance of 10 requests

### Example Usage

```bash
# Transcribe an audio file
curl -X POST http://localhost:8011/v1/audio/transcriptions \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: multipart/form-data" \\
  -F file="@audio.mp3" \\
  -F model="whisper-1"

# List available models
curl http://localhost:8011/v1/models \\
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Response Format

Transcription responses follow the OpenAI format:

```json
{
  "text": "The transcribed text appears here.",
  "usage": {
    "type": "tokens",
    "input_tokens": 1,
    "input_token_details": {
      "text_tokens": 0,
      "audio_tokens": 1
    },
    "output_tokens": 1,
    "total_tokens": 2
  }
}
```
"""

# Create FastAPI application
app = FastAPI(
    title="parakeetv2API",
    description=API_DESCRIPTION,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Audio",
            "description": "Audio transcription endpoints",
        },
        {
            "name": "Models",
            "description": "Model information endpoints",
        },
        {
            "name": "Health",
            "description": "Health check and monitoring endpoints",
        },
    ],
    servers=[
        {
            "url": "http://localhost:8011",
            "description": "Local development server",
        },
        {
            "url": "https://api.example.com",
            "description": "Production server (update with your domain)",
        },
    ],
)

# Add middleware in reverse order (last added is executed first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add performance monitoring middleware
app.add_middleware(PerformanceMonitoringMiddleware, threshold_ms=3000)

# Add request tracing middleware (executed first)
app.add_middleware(RequestTracingMiddleware)


# Exception handlers
@app.exception_handler(ParakeetAPIException)
async def parakeet_exception_handler(request: Request, exc: ParakeetAPIException):
    """Handle custom parakeet API exceptions."""
    logger.warning(
        "api_exception",
        status_code=exc.status_code,
        message=exc.message,
        details=exc.details,
        request_id=getattr(request.state, "request_id", None),
    )
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
    logger.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error_message=str(exc),
        request_id=getattr(request.state, "request_id", None),
        exc_info=True,
    )
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


@app.get(
    "/",
    tags=["Health"],
    summary="Root health check",
    description="Basic health check endpoint to verify the service is running.",
    response_description="Service status information",
)
async def root() -> Dict[str, Any]:
    """Root endpoint for basic health check.
    
    Returns basic service information without detailed metrics.
    This endpoint is not rate-limited and can be used for simple availability checks.
    
    Returns:
        Dict containing service status, name, and version
    """
    return {
        "status": "healthy",
        "service": "parakeetv2API",
        "version": "0.1.0",
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Detailed health check",
    description="Comprehensive health check with system metrics and model status.",
    response_description="Detailed health information including system metrics",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "model_loaded": True,
                        "warnings": [],
                        "metrics": {
                            "process_cpu_percent": 15.2,
                            "process_memory_mb": 1024.5,
                            "system_memory_used_mb": 8192.0,
                            "system_memory_available_mb": 7168.0,
                            "system_memory_percent": 53.3,
                            "gpu_memory_used_mb": 2048.0,
                            "gpu_memory_total_mb": 11264.0,
                            "gpu_memory_percent": 18.2,
                            "gpu_utilization_percent": 25,
                            "gpu_temperature_c": 45
                        }
                    }
                }
            }
        },
        200: {
            "description": "Service is degraded (with warnings)",
            "content": {
                "application/json": {
                    "example": {
                        "status": "degraded",
                        "model_loaded": True,
                        "warnings": ["High GPU temperature"],
                        "metrics": {
                            "gpu_temperature_c": 85
                        }
                    }
                }
            }
        }
    }
)
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint with system metrics.
    
    This endpoint provides detailed health information including:
    - Model loading status
    - System resource usage (CPU, memory)
    - GPU metrics (if available)
    - Performance warnings
    
    The status can be:
    - "healthy": All systems operating normally
    - "degraded": Operating with warnings (high resource usage, temperature, etc.)
    
    Returns:
        Dict containing detailed health status, metrics, and any warnings
    """
    health_data = system_monitor.check_health()
    
    return {
        "status": health_data["status"],
        "model_loaded": model_manager.is_loaded,
        "warnings": health_data.get("warnings", []),
        "metrics": health_data.get("metrics", {}),
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