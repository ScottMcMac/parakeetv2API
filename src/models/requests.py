"""Request models for parakeetv2API."""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

from src.core.exceptions import UnsupportedParameterError


class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    
    model: str = Field(
        default="whisper-1",
        description="ID of the model to use. All models use the same backend.",
    )
    language: Optional[str] = Field(
        default=None,
        description="The language of the input audio. Only 'en' is supported.",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="An optional text to guide the model's style. (Ignored)",
    )
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Field(
        default="json",
        description="The format of the transcript output. Only 'json' is supported.",
    )
    temperature: Optional[float] = Field(
        default=None,
        description="The sampling temperature. (Ignored)",
        ge=0.0,
        le=1.0,
    )
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Field(
        default=None,
        description="The timestamp granularities to populate. (Not supported)",
    )
    
    # OpenAI API compatibility fields (ignored)
    chunking_strategy: Optional[Union[str, dict]] = Field(
        default=None,
        description="The chunking strategy. (Ignored)",
    )
    include: Optional[List[str]] = Field(
        default=None,
        description="Additional information to include. (Ignored)",
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response. (Not supported for some models)",
    )
    
    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model selection."""
        # All models are supported (they all use the same backend)
        supported_models = {
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe", 
            "parakeet-tdt-0.6b-v2",
            "whisper-1"
        }
        
        # We accept any model name for compatibility
        # Log if it's not one of the expected ones
        if v not in supported_models:
            # We still accept it, just log
            import logging
            logging.getLogger(__name__).info(
                f"Using non-standard model name '{v}', will use parakeet-tdt-0.6b-v2 backend"
            )
        
        return v
    
    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate language selection."""
        if v is not None and v != "en":
            raise UnsupportedParameterError(
                parameter="language",
                value=v,
                reason="Only English ('en') is supported by this model."
            )
        return v
    
    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate response format."""
        if v is not None and v != "json":
            raise UnsupportedParameterError(
                parameter="response_format",
                value=v,
                reason="Only 'json' format is supported."
            )
        return v
    
    @field_validator("timestamp_granularities")
    @classmethod
    def validate_timestamp_granularities(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate timestamp granularities."""
        if v is not None:
            raise UnsupportedParameterError(
                parameter="timestamp_granularities",
                value=v,
                reason="Timestamp granularities are not supported by this implementation."
            )
        return v
    
    @field_validator("stream")
    @classmethod
    def validate_stream(cls, v: Optional[bool], values) -> Optional[bool]:
        """Validate stream parameter based on model."""
        if v is True:
            model = values.data.get("model", "whisper-1")
            # Only error for OpenAI-style models when streaming is requested
            if model in ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"]:
                raise UnsupportedParameterError(
                    parameter="stream",
                    value=v,
                    reason=f"Streaming is not supported for model '{model}'."
                )
        return v


class ModelListRequest(BaseModel):
    """Request model for listing models (no parameters needed)."""
    pass


class ModelInfoRequest(BaseModel):
    """Request model for getting model info."""
    
    model_id: str = Field(
        ...,
        description="The ID of the model to get info for.",
    )