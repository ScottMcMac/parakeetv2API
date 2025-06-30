"""Enhanced error handling with graceful degradation."""

import traceback
from typing import Dict, Optional, Any
from functools import wraps
import asyncio

from fastapi import HTTPException, Request, status
from fastapi.responses import ORJSONResponse

from src.core.exceptions import (
    ParakeetAPIException,
    AudioValidationError,
    AudioProcessingError,
    ModelError,
    ModelNotLoadedError,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


class ErrorHandler:
    """Centralized error handling with graceful degradation."""
    
    # Error code mapping
    ERROR_CODES = {
        AudioValidationError: "audio_validation_error",
        AudioProcessingError: "audio_processing_error",
        ModelError: "model_error",
        ModelNotLoadedError: "model_not_loaded",
    }
    
    # Client-friendly error messages
    USER_MESSAGES = {
        "audio_validation_error": "The provided audio file is invalid or unsupported.",
        "audio_processing_error": "Failed to process the audio file. Please try again.",
        "model_error": "An error occurred during transcription. Please try again.",
        "model_not_loaded": "The transcription service is temporarily unavailable.",
        "rate_limit_exceeded": "Too many requests. Please slow down.",
        "internal_error": "An unexpected error occurred. Please try again later.",
    }
    
    @classmethod
    def format_error_response(
        cls,
        error: Exception,
        request_id: Optional[str] = None,
        include_details: bool = True,
    ) -> Dict[str, Any]:
        """Format error into standardized response.
        
        Args:
            error: The exception that occurred
            request_id: Request ID for tracking
            include_details: Whether to include detailed error info
            
        Returns:
            Formatted error response
        """
        # Determine error code and type
        error_code = cls.ERROR_CODES.get(type(error), "internal_error")
        error_type = "invalid_request_error" if isinstance(error, (AudioValidationError,)) else "api_error"
        
        # Get user-friendly message
        if isinstance(error, ParakeetAPIException):
            user_message = error.message
        else:
            user_message = cls.USER_MESSAGES.get(error_code, cls.USER_MESSAGES["internal_error"])
        
        # Build response
        response = {
            "error": {
                "message": user_message,
                "type": error_type,
                "code": error_code,
            }
        }
        
        # Add request ID if available
        if request_id:
            response["error"]["request_id"] = request_id
        
        # Add details in development mode
        if include_details and hasattr(error, "details"):
            response["error"]["details"] = error.details
        
        return response
    
    @classmethod
    def get_status_code(cls, error: Exception) -> int:
        """Get appropriate HTTP status code for error.
        
        Args:
            error: The exception
            
        Returns:
            HTTP status code
        """
        if isinstance(error, AudioValidationError):
            return status.HTTP_400_BAD_REQUEST
        elif isinstance(error, AudioProcessingError):
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        elif isinstance(error, ModelNotLoadedError):
            return status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(error, ModelError):
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        elif isinstance(error, HTTPException):
            return error.status_code
        else:
            return status.HTTP_500_INTERNAL_SERVER_ERROR
    
    @classmethod
    async def handle_request_error(
        cls,
        request: Request,
        error: Exception,
    ) -> ORJSONResponse:
        """Handle error during request processing.
        
        Args:
            request: FastAPI request object
            error: The exception that occurred
            
        Returns:
            JSON error response
        """
        request_id = getattr(request.state, "request_id", None)
        
        # Log the error
        if isinstance(error, ParakeetAPIException):
            logger.warning(
                "request_error",
                error_type=type(error).__name__,
                error_message=str(error),
                request_id=request_id,
                path=request.url.path,
            )
        else:
            logger.error(
                "unexpected_error",
                error_type=type(error).__name__,
                error_message=str(error),
                request_id=request_id,
                path=request.url.path,
                exc_info=True,
            )
        
        # Format response
        status_code = cls.get_status_code(error)
        response_data = cls.format_error_response(
            error,
            request_id=request_id,
            include_details=request.app.debug,
        )
        
        return ORJSONResponse(
            status_code=status_code,
            content=response_data,
        )


def with_error_handling(func):
    """Decorator for consistent error handling in async functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ParakeetAPIException:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Log and convert other exceptions
            logger.error(
                "function_error",
                function=func.__name__,
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            # Convert to appropriate exception type
            if "ffmpeg" in str(e).lower():
                raise AudioProcessingError(f"Audio processing failed: {str(e)}")
            elif "cuda" in str(e).lower() or "gpu" in str(e).lower():
                raise ModelError(f"GPU error: {str(e)}")
            else:
                raise ModelError(f"Operation failed: {str(e)}")
    
    return wrapper


class CircuitBreaker:
    """Circuit breaker for handling repeated failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    async def call(self, func, *args, **kwargs):
        """Call function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise ModelNotLoadedError("Service temporarily unavailable due to repeated failures")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        import time
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None
    
    def _on_failure(self):
        """Handle failed call."""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                recovery_timeout=self.recovery_timeout,
            )


# Global error handler instance
error_handler = ErrorHandler()