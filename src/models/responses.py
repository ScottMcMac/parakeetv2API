"""Response models for parakeetv2API."""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TokenUsageDetails(BaseModel):
    """Token usage details for input."""
    
    text_tokens: int = Field(
        default=0,
        description="Number of text tokens (always 0 for audio).",
    )
    audio_tokens: int = Field(
        default=1,
        description="Number of audio tokens (always 1).",
    )


class TokenUsage(BaseModel):
    """Token usage information."""
    
    type: Literal["tokens"] = Field(
        default="tokens",
        description="Type of usage measurement.",
    )
    input_tokens: int = Field(
        default=1,
        description="Number of input tokens (always 1).",
    )
    input_token_details: TokenUsageDetails = Field(
        default_factory=lambda: TokenUsageDetails(),
        description="Breakdown of input token usage.",
    )
    output_tokens: int = Field(
        default=1,
        description="Number of output tokens (always 1).",
    )
    total_tokens: int = Field(
        default=2,
        description="Total number of tokens (always 2).",
    )


class TranscriptionResponse(BaseModel):
    """Response model for audio transcription."""
    
    text: str = Field(
        ...,
        description="The transcribed text.",
    )
    usage: TokenUsage = Field(
        default_factory=lambda: TokenUsage(),
        description="Token usage information (fixed values for compatibility).",
    )


class ModelInfo(BaseModel):
    """Information about a single model."""
    
    id: str = Field(
        ...,
        description="The model identifier.",
    )
    object: Literal["model"] = Field(
        default="model",
        description="Object type.",
    )
    created: int = Field(
        default=1744718400,  # 2025-01-13 timestamp
        description="Unix timestamp when the model was created.",
    )
    owned_by: str = Field(
        default="parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license",
        description="The organization that owns the model.",
    )


class ModelListResponse(BaseModel):
    """Response model for listing available models."""
    
    object: Literal["list"] = Field(
        default="list",
        description="Object type.",
    )
    data: List[ModelInfo] = Field(
        ...,
        description="List of available models.",
    )


class ErrorDetail(BaseModel):
    """Error detail information."""
    
    message: str = Field(
        ...,
        description="Error message.",
    )
    type: str = Field(
        ...,
        description="Error type.",
    )
    param: Optional[str] = Field(
        default=None,
        description="Parameter that caused the error.",
    )
    code: Optional[str] = Field(
        default=None,
        description="Error code.",
    )


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: ErrorDetail = Field(
        ...,
        description="Error details.",
    )


# Pre-defined model information
AVAILABLE_MODELS = [
    ModelInfo(
        id="gpt-4o-transcribe",
        object="model",
        created=1744718400,
        owned_by="parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
    ),
    ModelInfo(
        id="gpt-4o-mini-transcribe",
        object="model",
        created=1744718400,
        owned_by="parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
    ),
    ModelInfo(
        id="parakeet-tdt-0.6b-v2",
        object="model",
        created=1744718400,
        owned_by="parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
    ),
    ModelInfo(
        id="whisper-1",
        object="model",
        created=1744718400,
        owned_by="parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"
    ),
]


def get_model_list() -> ModelListResponse:
    """Get the list of available models."""
    return ModelListResponse(
        object="list",
        data=AVAILABLE_MODELS
    )


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """
    Get information about a specific model.
    
    Args:
        model_id: The model ID to look up
        
    Returns:
        ModelInfo if found, None otherwise
    """
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model
    return None