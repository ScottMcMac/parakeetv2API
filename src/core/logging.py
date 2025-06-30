"""Logging configuration for parakeetv2API."""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.contextvars import merge_contextvars

from src.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level,
    )
    
    # Configure structlog
    processors = [
        merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.environment == "development":
        # Use pretty console output in development
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # Use JSON output in production
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Logger for HTTP request/response tracking."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def log_request(
        self,
        method: str,
        path: str,
        request_id: str,
        client_host: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log incoming request."""
        self.logger.info(
            "request_started",
            method=method,
            path=path,
            request_id=request_id,
            client_host=client_host,
            **kwargs
        )
    
    def log_response(
        self,
        method: str,
        path: str,
        request_id: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ) -> None:
        """Log outgoing response."""
        log_method = self.logger.info if status_code < 400 else self.logger.warning
        
        log_method(
            "request_completed",
            method=method,
            path=path,
            request_id=request_id,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            **kwargs
        )
    
    def log_error(
        self,
        method: str,
        path: str,
        request_id: str,
        error: Exception,
        **kwargs
    ) -> None:
        """Log request error."""
        self.logger.error(
            "request_failed",
            method=method,
            path=path,
            request_id=request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )


class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def log_model_inference(
        self,
        request_id: str,
        duration_ms: float,
        audio_duration_s: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log model inference performance."""
        metrics = {
            "inference_duration_ms": round(duration_ms, 2),
            "request_id": request_id,
        }
        
        if audio_duration_s is not None:
            metrics["audio_duration_s"] = round(audio_duration_s, 2)
            metrics["rtf"] = round(duration_ms / 1000 / audio_duration_s, 3)  # Real-time factor
        
        self.logger.info("model_inference_completed", **metrics, **kwargs)
    
    def log_audio_processing(
        self,
        request_id: str,
        operation: str,
        duration_ms: float,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log audio processing performance."""
        self.logger.info(
            "audio_processing_completed",
            request_id=request_id,
            operation=operation,
            duration_ms=round(duration_ms, 2),
            input_format=input_format,
            output_format=output_format,
            **kwargs
        )
    
    def log_memory_usage(
        self,
        used_mb: float,
        available_mb: float,
        gpu_used_mb: Optional[float] = None,
        gpu_total_mb: Optional[float] = None,
        **kwargs
    ) -> None:
        """Log memory usage metrics."""
        metrics = {
            "memory_used_mb": round(used_mb, 2),
            "memory_available_mb": round(available_mb, 2),
            "memory_usage_percent": round(used_mb / (used_mb + available_mb) * 100, 2),
        }
        
        if gpu_used_mb is not None and gpu_total_mb is not None:
            metrics.update({
                "gpu_memory_used_mb": round(gpu_used_mb, 2),
                "gpu_memory_total_mb": round(gpu_total_mb, 2),
                "gpu_memory_usage_percent": round(gpu_used_mb / gpu_total_mb * 100, 2),
            })
        
        self.logger.info("memory_usage", **metrics, **kwargs)


# Global logger instances
request_logger = RequestLogger()
performance_logger = PerformanceLogger()