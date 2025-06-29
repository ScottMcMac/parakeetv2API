"""Unit tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from src.core.exceptions import UnsupportedParameterError
from src.models import (
    AVAILABLE_MODELS,
    ModelInfo,
    ModelListResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    TokenUsage,
    TokenUsageDetails,
    get_model_info,
    get_model_list,
)


class TestTokenUsageDetails:
    """Test token usage details model."""

    def test_default_values(self):
        """Test default values."""
        details = TokenUsageDetails()
        assert details.text_tokens == 0
        assert details.audio_tokens == 1

    def test_custom_values(self):
        """Test custom values."""
        details = TokenUsageDetails(text_tokens=5, audio_tokens=3)
        assert details.text_tokens == 5
        assert details.audio_tokens == 3


class TestTokenUsage:
    """Test token usage model."""

    def test_default_values(self):
        """Test default values."""
        usage = TokenUsage()
        assert usage.type == "tokens"
        assert usage.input_tokens == 1
        assert usage.output_tokens == 1
        assert usage.total_tokens == 2
        assert usage.input_token_details.text_tokens == 0
        assert usage.input_token_details.audio_tokens == 1

    def test_custom_values(self):
        """Test custom values."""
        details = TokenUsageDetails(text_tokens=2, audio_tokens=3)
        usage = TokenUsage(
            input_tokens=5,
            output_tokens=10,
            total_tokens=15,
            input_token_details=details
        )
        assert usage.input_tokens == 5
        assert usage.output_tokens == 10
        assert usage.total_tokens == 15
        assert usage.input_token_details.text_tokens == 2
        assert usage.input_token_details.audio_tokens == 3


class TestTranscriptionRequest:
    """Test transcription request model."""

    def test_default_values(self):
        """Test default values."""
        request = TranscriptionRequest()
        assert request.model == "whisper-1"
        assert request.language is None
        assert request.response_format == "json"
        assert request.stream is False

    def test_valid_request(self):
        """Test valid request creation."""
        request = TranscriptionRequest(
            model="gpt-4o-transcribe",
            language="en",
            response_format="json"
        )
        assert request.model == "gpt-4o-transcribe"
        assert request.language == "en"
        assert request.response_format == "json"

    def test_model_validation(self):
        """Test model validation."""
        # Valid models should work
        valid_models = [
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe", 
            "parakeet-tdt-0.6b-v2",
            "whisper-1"
        ]
        
        for model in valid_models:
            request = TranscriptionRequest(model=model)
            assert request.model == model

        # Non-standard models should be accepted but logged
        request = TranscriptionRequest(model="custom-model")
        assert request.model == "custom-model"

    def test_language_validation(self):
        """Test language validation."""
        # English should be accepted
        request = TranscriptionRequest(language="en")
        assert request.language == "en"

        # Other languages should raise error
        with pytest.raises(UnsupportedParameterError) as exc_info:
            TranscriptionRequest(language="fr")
        
        assert "Only English" in str(exc_info.value)

    def test_response_format_validation(self):
        """Test response format validation."""
        # JSON should be accepted
        request = TranscriptionRequest(response_format="json")
        assert request.response_format == "json"

        # Other formats should raise error
        with pytest.raises(UnsupportedParameterError) as exc_info:
            TranscriptionRequest(response_format="text")
        
        assert "Only 'json' format" in str(exc_info.value)

    def test_timestamp_granularities_validation(self):
        """Test timestamp granularities validation."""
        # Should raise error if provided
        with pytest.raises(UnsupportedParameterError) as exc_info:
            TranscriptionRequest(timestamp_granularities=["word"])
        
        assert "not supported" in str(exc_info.value)

    def test_stream_validation(self):
        """Test stream validation."""
        # Should be fine for whisper-1 and parakeet models
        request = TranscriptionRequest(model="whisper-1", stream=True)
        assert request.stream is True

        request = TranscriptionRequest(model="parakeet-tdt-0.6b-v2", stream=True)
        assert request.stream is True

        # Should raise error for OpenAI models
        with pytest.raises(UnsupportedParameterError) as exc_info:
            TranscriptionRequest(model="gpt-4o-transcribe", stream=True)
        
        assert "not supported for model" in str(exc_info.value)

    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperature values
        request = TranscriptionRequest(temperature=0.0)
        assert request.temperature == 0.0

        request = TranscriptionRequest(temperature=1.0)
        assert request.temperature == 1.0

        request = TranscriptionRequest(temperature=0.5)
        assert request.temperature == 0.5

        # Invalid temperature values
        with pytest.raises(ValidationError):
            TranscriptionRequest(temperature=-0.1)

        with pytest.raises(ValidationError):
            TranscriptionRequest(temperature=1.1)


class TestTranscriptionResponse:
    """Test transcription response model."""

    def test_basic_response(self):
        """Test basic response creation."""
        response = TranscriptionResponse(text="Hello world")
        assert response.text == "Hello world"
        assert response.usage.type == "tokens"
        assert response.usage.input_tokens == 1
        assert response.usage.output_tokens == 1
        assert response.usage.total_tokens == 2

    def test_empty_text(self):
        """Test response with empty text."""
        response = TranscriptionResponse(text="")
        assert response.text == ""
        assert response.usage.total_tokens == 2  # Still fixed usage


class TestModelInfo:
    """Test model info model."""

    def test_model_info_creation(self):
        """Test model info creation."""
        model = ModelInfo(id="test-model")
        assert model.id == "test-model"
        assert model.object == "model"
        assert model.created == 1744718400
        assert model.owned_by == "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"

    def test_custom_values(self):
        """Test custom values."""
        model = ModelInfo(
            id="custom-model",
            created=1234567890,
            owned_by="custom-owner"
        )
        assert model.id == "custom-model"
        assert model.created == 1234567890
        assert model.owned_by == "custom-owner"


class TestModelListResponse:
    """Test model list response."""

    def test_model_list_creation(self):
        """Test model list creation."""
        models = [
            ModelInfo(id="model1"),
            ModelInfo(id="model2")
        ]
        response = ModelListResponse(data=models)
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "model1"
        assert response.data[1].id == "model2"


class TestModelUtilities:
    """Test model utility functions."""

    def test_available_models(self):
        """Test available models list."""
        assert len(AVAILABLE_MODELS) == 4
        model_ids = [model.id for model in AVAILABLE_MODELS]
        expected_ids = [
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "parakeet-tdt-0.6b-v2",
            "whisper-1"
        ]
        assert set(model_ids) == set(expected_ids)

    def test_get_model_list(self):
        """Test get_model_list function."""
        model_list = get_model_list()
        assert isinstance(model_list, ModelListResponse)
        assert model_list.object == "list"
        assert len(model_list.data) == 4

    def test_get_model_info_existing(self):
        """Test get_model_info for existing models."""
        for model in AVAILABLE_MODELS:
            info = get_model_info(model.id)
            assert info is not None
            assert info.id == model.id
            assert info.object == "model"

    def test_get_model_info_nonexistent(self):
        """Test get_model_info for non-existent model."""
        info = get_model_info("non-existent-model")
        assert info is None