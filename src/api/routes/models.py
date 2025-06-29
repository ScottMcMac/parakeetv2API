"""Model information API routes."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_request_id, verify_api_key
from src.models import ModelInfo, ModelListResponse
from src.services import model_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["Models"])


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List models",
    description="Lists the currently available models.",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "object": "list",
                        "data": [
                            {
                                "id": "gpt-4o-transcribe",
                                "object": "model",
                                "created": 1744718400,
                                "owned_by": "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
                            },
                            {
                                "id": "whisper-1",
                                "object": "model", 
                                "created": 1744718400,
                                "owned_by": "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def list_models(
    api_key: Optional[str] = Depends(verify_api_key),
    request_id: Optional[str] = Depends(get_request_id),
) -> ModelListResponse:
    """
    List available models.
    
    Returns a list of models that are available for transcription.
    All models use the same parakeet-tdt-0.6b-v2 backend.
    """
    return model_service.list_models(request_id=request_id)


@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    summary="Retrieve model",
    description="Retrieves a model instance, providing basic information about the model.",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "id": "parakeet-tdt-0.6b-v2",
                        "object": "model",
                        "created": 1744718400,
                        "owned_by": "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
                    }
                }
            }
        },
        404: {
            "description": "Model not found",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "message": "Model 'unknown-model' not found",
                            "type": "invalid_request_error",
                            "param": "model_id",
                            "code": "model_not_found"
                        }
                    }
                }
            }
        }
    }
)
async def get_model(
    model_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
    request_id: Optional[str] = Depends(get_request_id),
) -> ModelInfo:
    """
    Get information about a specific model.
    
    Args:
        model_id: The ID of the model to retrieve
        
    Returns:
        Model information if found
        
    Raises:
        HTTPException: If model is not found
    """
    model_info = model_service.get_model_info(model_id, request_id=request_id)
    
    if model_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": f"Model '{model_id}' not found",
                    "type": "invalid_request_error",
                    "param": "model_id",
                    "code": "model_not_found"
                }
            }
        )
    
    return model_info