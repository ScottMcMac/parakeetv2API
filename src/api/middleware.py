"""Middleware for request tracking and monitoring."""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from structlog.contextvars import bind_contextvars, clear_contextvars

from src.core.logging import request_logger, get_logger


logger = get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracing and logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and add tracing."""
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Bind request ID to context for structured logging
        clear_contextvars()
        bind_contextvars(request_id=request_id)
        
        # Store request ID in request state
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        request_logger.log_request(
            method=request.method,
            path=request.url.path,
            request_id=request_id,
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            request_logger.log_response(
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log error
            request_logger.log_error(
                method=request.method,
                path=request.url.path,
                request_id=request_id,
                error=e,
                duration_ms=duration_ms,
            )
            
            # Re-raise the exception
            raise
        finally:
            # Clear context
            clear_contextvars()


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring."""
    
    def __init__(self, app, threshold_ms: float = 5000):
        super().__init__(app)
        self.threshold_ms = threshold_ms
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log slow requests
        if duration_ms > self.threshold_ms:
            logger.warning(
                "slow_request_detected",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                threshold_ms=self.threshold_ms,
                request_id=getattr(request.state, "request_id", None),
            )
        
        # Add performance headers
        response.headers["X-Response-Time-ms"] = str(round(duration_ms, 2))
        
        return response