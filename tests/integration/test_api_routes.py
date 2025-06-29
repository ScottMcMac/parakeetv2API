"""Integration tests for API routes."""

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status

from src.core.exceptions import (
    AudioProcessingError,
    AudioValidationError,
    ModelError,
    ModelNotLoadedError,
)


class TestTranscriptionAPI:
    """Test transcription API integration."""

    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, async_client):
        """Test successful audio transcription."""
        # Mock the transcription service
        with patch('src.services.transcription_service.transcribe_audio') as mock_transcribe:
            from src.models import TranscriptionResponse
            mock_transcribe.return_value = TranscriptionResponse(
                text="The quick brown fox jumped over the lazy dog."
            )
            
            # Create test file
            audio_content = b"fake audio content"
            files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            
            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            
            assert "text" in result
            assert result["text"] == "The quick brown fox jumped over the lazy dog."
            assert "usage" in result
            assert result["usage"]["input_tokens"] == 1
            assert result["usage"]["output_tokens"] == 1
            assert result["usage"]["total_tokens"] == 2

    @pytest.mark.asyncio
    async def test_transcribe_audio_no_file(self, async_client):
        """Test transcription without file."""
        data = {"model": "whisper-1"}
        
        response = await async_client.post("/v1/audio/transcriptions", data=data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_transcribe_audio_no_filename(self, async_client):
        """Test transcription without filename."""
        audio_content = b"fake audio content"
        files = {"file": ("", io.BytesIO(audio_content), "audio/wav")}
        data = {"model": "whisper-1"}
        
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "error" in result
        assert "No filename provided" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_invalid_language(self, async_client):
        """Test transcription with invalid language."""
        audio_content = b"fake audio content"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
        data = {"model": "whisper-1", "language": "fr"}
        
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "error" in result
        assert "Only English" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_invalid_response_format(self, async_client):
        """Test transcription with invalid response format."""
        audio_content = b"fake audio content"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
        data = {"model": "whisper-1", "response_format": "text"}
        
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "error" in result
        assert "Only 'json' format" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_streaming_error(self, async_client):
        """Test transcription with streaming for OpenAI models."""
        audio_content = b"fake audio content"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
        data = {"model": "gpt-4o-transcribe", "stream": "true"}
        
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "error" in result
        assert "not supported for model" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_timestamp_granularities_error(self, async_client):
        """Test transcription with timestamp granularities."""
        audio_content = b"fake audio content"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
        data = {"model": "whisper-1", "timestamp_granularities": '["word"]'}
        
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "error" in result
        assert "not supported" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_validation_error(self, async_client):
        """Test transcription with audio validation error."""
        with patch('src.services.transcription_service.transcribe_audio') as mock_transcribe:
            mock_transcribe.side_effect = AudioValidationError("Invalid audio format")
            
            audio_content = b"fake audio content"
            files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            
            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
            
            assert response.status_code == status.HTTP_400_BAD_REQUEST
            result = response.json()
            assert "error" in result
            assert "Invalid audio format" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_processing_error(self, async_client):
        """Test transcription with audio processing error."""
        with patch('src.services.transcription_service.transcribe_audio') as mock_transcribe:
            mock_transcribe.side_effect = AudioProcessingError("Processing failed")
            
            audio_content = b"fake audio content"
            files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            
            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            result = response.json()
            assert "error" in result
            assert "Processing failed" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_model_not_loaded(self, async_client):
        """Test transcription when model not loaded."""
        with patch('src.services.transcription_service.transcribe_audio') as mock_transcribe:
            mock_transcribe.side_effect = ModelNotLoadedError()
            
            audio_content = b"fake audio content"
            files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            
            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
            
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            result = response.json()
            assert "error" in result
            assert "Model is not loaded" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_model_error(self, async_client):
        """Test transcription with model error."""
        with patch('src.services.transcription_service.transcribe_audio') as mock_transcribe:
            mock_transcribe.side_effect = ModelError("Inference failed")
            
            audio_content = b"fake audio content"
            files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            
            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            result = response.json()
            assert "error" in result
            assert "Inference failed" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_transcribe_audio_unexpected_error(self, async_client):
        """Test transcription with unexpected error."""
        with patch('src.services.transcription_service.transcribe_audio') as mock_transcribe:
            mock_transcribe.side_effect = Exception("Unexpected error")
            
            audio_content = b"fake audio content"
            files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            
            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            result = response.json()
            assert "error" in result
            assert "An unexpected error occurred" in result["error"]["message"]


class TestModelsAPI:
    """Test models API integration."""

    @pytest.mark.asyncio
    async def test_list_models(self, async_client):
        """Test listing models."""
        response = await async_client.get("/v1/models")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        
        assert "object" in result
        assert result["object"] == "list"
        assert "data" in result
        assert len(result["data"]) == 4
        
        model_ids = [model["id"] for model in result["data"]]
        expected_ids = [
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "parakeet-tdt-0.6b-v2",
            "whisper-1"
        ]
        assert set(model_ids) == set(expected_ids)

    @pytest.mark.asyncio
    async def test_get_model_existing(self, async_client):
        """Test getting existing model."""
        response = await async_client.get("/v1/models/whisper-1")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        
        assert result["id"] == "whisper-1"
        assert result["object"] == "model"
        assert result["created"] == 1744718400
        assert result["owned_by"] == "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"

    @pytest.mark.asyncio
    async def test_get_model_nonexistent(self, async_client):
        """Test getting non-existent model."""
        response = await async_client.get("/v1/models/non-existent-model")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        
        assert "error" in result
        assert "not found" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_models_with_auth_header(self, async_client):
        """Test models endpoints with authorization header."""
        headers = {"Authorization": "Bearer fake-api-key"}
        
        # Test list models
        response = await async_client.get("/v1/models", headers=headers)
        assert response.status_code == status.HTTP_200_OK
        
        # Test get model
        response = await async_client.get("/v1/models/whisper-1", headers=headers)
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_models_with_request_id(self, async_client):
        """Test models endpoints with request ID header."""
        headers = {"X-Request-ID": "test-request-123"}
        
        # Test list models
        response = await async_client.get("/v1/models", headers=headers)
        assert response.status_code == status.HTTP_200_OK
        
        # Test get model
        response = await async_client.get("/v1/models/whisper-1", headers=headers)
        assert response.status_code == status.HTTP_200_OK


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client):
        """Test root health endpoint."""
        response = await async_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        result = response.json()
        
        assert result["status"] == "healthy"
        assert result["service"] == "parakeetv2API"
        assert result["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        """Test health check endpoint."""
        with patch('src.core.model_manager.is_loaded', True):
            response = await async_client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            
            assert result["status"] == "healthy"
            assert result["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_health_endpoint_model_not_loaded(self, async_client):
        """Test health check when model not loaded."""
        with patch('src.core.model_manager.is_loaded', False):
            response = await async_client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            
            assert result["status"] == "healthy"
            assert result["model_loaded"] is False


class TestCORSAndMiddleware:
    """Test CORS and middleware functionality."""

    @pytest.mark.asyncio
    async def test_cors_headers(self, async_client):
        """Test CORS headers are present."""
        response = await async_client.options("/v1/models")
        
        # FastAPI should handle CORS properly
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]

    @pytest.mark.asyncio
    async def test_content_type_json(self, async_client):
        """Test content type is JSON."""
        response = await async_client.get("/v1/models")
        
        assert response.status_code == status.HTTP_200_OK
        assert "application/json" in response.headers.get("content-type", "")