"""Custom exceptions for parakeetv2API."""

from typing import Any, Dict, Optional


class ParakeetAPIException(Exception):
    """Base exception for all parakeetv2API exceptions."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            status_code: HTTP status code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class AudioValidationError(ParakeetAPIException):
    """Raised when audio file validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize audio validation error."""
        super().__init__(message, status_code=400, details=details)


class AudioProcessingError(ParakeetAPIException):
    """Raised when audio processing fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize audio processing error."""
        super().__init__(message, status_code=500, details=details)


class ModelError(ParakeetAPIException):
    """Raised when model operations fail."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize model error."""
        super().__init__(message, status_code=500, details=details)


class ModelNotLoadedError(ModelError):
    """Raised when trying to use model before it's loaded."""
    
    def __init__(self):
        """Initialize model not loaded error."""
        super().__init__(
            "Model is not loaded. Please wait for server initialization to complete.",
            details={"error_code": "model_not_loaded"},
        )


class UnsupportedParameterError(ParakeetAPIException):
    """Raised when an unsupported parameter is provided."""
    
    def __init__(self, parameter: str, value: Any, reason: str):
        """Initialize unsupported parameter error."""
        super().__init__(
            f"Unsupported value for parameter '{parameter}': {value}. {reason}",
            status_code=400,
            details={
                "parameter": parameter,
                "value": value,
                "reason": reason,
            },
        )