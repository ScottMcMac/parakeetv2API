"""Model service for handling model information and management."""

import logging
from typing import Optional

from src.models import ModelInfo, ModelListResponse, get_model_info, get_model_list

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling model information and queries."""

    def __init__(self):
        """Initialize the model service."""
        self.supported_models = {
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "parakeet-tdt-0.6b-v2", 
            "whisper-1"
        }

    def list_models(self, request_id: Optional[str] = None) -> ModelListResponse:
        """
        Get list of available models.

        Args:
            request_id: Optional request ID for logging

        Returns:
            List of available models
        """
        logger.info(f"Listing models - request_id: {request_id}")
        
        model_list = get_model_list()
        
        logger.debug(f"Returned {len(model_list.data)} models")
        return model_list

    def get_model_info(
        self, 
        model_id: str, 
        request_id: Optional[str] = None
    ) -> Optional[ModelInfo]:
        """
        Get information about a specific model.

        Args:
            model_id: The ID of the model to retrieve
            request_id: Optional request ID for logging

        Returns:
            Model information if found, None otherwise
        """
        logger.info(f"Getting model info - model_id: {model_id}, request_id: {request_id}")
        
        model_info = get_model_info(model_id)
        
        if model_info:
            logger.debug(f"Model found: {model_id}")
        else:
            logger.warning(f"Model not found: {model_id}")
        
        return model_info

    def is_model_supported(self, model_id: str) -> bool:
        """
        Check if a model is supported.

        Args:
            model_id: Model ID to check

        Returns:
            True if model is supported
        """
        return model_id in self.supported_models

    def get_supported_models(self) -> list[str]:
        """
        Get list of supported model IDs.

        Returns:
            List of supported model IDs
        """
        return list(self.supported_models)

    def get_backend_model_name(self, model_id: str) -> str:
        """
        Get the backend model name for a given model ID.
        
        All models use the same backend (parakeet-tdt-0.6b-v2).

        Args:
            model_id: Model ID from request

        Returns:
            Backend model name
        """
        # All models use the same backend
        backend_model = "nvidia/parakeet-tdt-0.6b-v2"
        
        if model_id not in self.supported_models:
            logger.info(
                f"Using non-standard model name '{model_id}', "
                f"will use {backend_model} backend"
            )
        
        return backend_model

    def validate_model_id(self, model_id: str) -> bool:
        """
        Validate if a model ID is acceptable.
        
        We accept any model ID for compatibility but log if it's non-standard.

        Args:
            model_id: Model ID to validate

        Returns:
            Always True (for compatibility)
        """
        if model_id not in self.supported_models:
            logger.info(f"Non-standard model ID: {model_id}")
        
        return True


# Global model service instance
model_service = ModelService()