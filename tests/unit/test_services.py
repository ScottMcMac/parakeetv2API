"""Unit tests for services."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from src.core.exceptions import (
    AudioProcessingError,
    AudioValidationError,
    ModelError,
    ModelNotLoadedError,
)
from src.models import TranscriptionRequest, TranscriptionResponse
from src.services import audio_service, model_service, transcription_service


class TestModelService:
    """Test model service."""

    def test_list_models(self):
        """Test listing models."""
        result = model_service.list_models(request_id="test-123")
        
        assert result.object == "list"
        assert len(result.data) == 4
        model_ids = [model.id for model in result.data]
        expected_ids = [
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
            "parakeet-tdt-0.6b-v2",
            "whisper-1"
        ]
        assert set(model_ids) == set(expected_ids)

    def test_get_model_info_existing(self):
        """Test getting info for existing model."""
        model_info = model_service.get_model_info("whisper-1", request_id="test-123")
        
        assert model_info is not None
        assert model_info.id == "whisper-1"
        assert model_info.object == "model"
        assert model_info.owned_by == "parakeet-tdt-0.6b-v2-released-by-nvidia-with-cc-by-40-license"

    def test_get_model_info_nonexistent(self):
        """Test getting info for non-existent model."""
        model_info = model_service.get_model_info("non-existent", request_id="test-123")
        assert model_info is None

    def test_is_model_supported(self):
        """Test model support checking."""
        assert model_service.is_model_supported("whisper-1")
        assert model_service.is_model_supported("gpt-4o-transcribe")
        assert not model_service.is_model_supported("unknown-model")

    def test_get_supported_models(self):
        """Test getting supported models list."""
        models = model_service.get_supported_models()
        assert len(models) == 4
        assert "whisper-1" in models
        assert "gpt-4o-transcribe" in models

    def test_get_backend_model_name(self):
        """Test getting backend model name."""
        # All supported models should return same backend
        for model_id in model_service.get_supported_models():
            backend = model_service.get_backend_model_name(model_id)
            assert backend == "nvidia/parakeet-tdt-0.6b-v2"

        # Unknown model should also return same backend
        backend = model_service.get_backend_model_name("unknown-model")
        assert backend == "nvidia/parakeet-tdt-0.6b-v2"

    def test_validate_model_id(self):
        """Test model ID validation."""
        # All model IDs should be considered valid for compatibility
        assert model_service.validate_model_id("whisper-1")
        assert model_service.validate_model_id("unknown-model")


class TestAudioService:
    """Test audio service."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio content")
            temp_path = Path(tmp.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.mark.asyncio
    async def test_validate_and_save_file_valid(self):
        """Test validating and saving a valid file."""
        content = b"fake audio content"
        filename = "test.wav"
        
        with patch.object(audio_service.audio_processor, 'save_uploaded_file') as mock_save:
            mock_save.return_value = Path("/tmp/saved_file.wav")
            
            result = await audio_service.validate_and_save_file(
                content, filename, request_id="test-123"
            )
            
            assert result == Path("/tmp/saved_file.wav")
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_and_save_file_invalid_extension(self):
        """Test validating file with invalid extension."""
        content = b"fake content"
        filename = "test.txt"
        
        with pytest.raises(AudioValidationError, match="Unsupported file format"):
            await audio_service.validate_and_save_file(
                content, filename, request_id="test-123"
            )

    @pytest.mark.asyncio
    async def test_validate_and_save_file_too_large(self):
        """Test validating file that's too large."""
        content = b"x" * (30 * 1024 * 1024)  # 30MB
        filename = "test.wav"
        
        with pytest.raises(AudioValidationError, match="File too large"):
            await audio_service.validate_and_save_file(
                content, filename, request_id="test-123"
            )

    @pytest.mark.asyncio
    async def test_process_for_transcription(self, temp_file):
        """Test processing audio for transcription."""
        with patch.object(audio_service.audio_processor, 'process_audio_file') as mock_process:
            mock_process.return_value = (temp_file, False)
            
            result_path, needs_cleanup = await audio_service.process_for_transcription(
                temp_file, request_id="test-123"
            )
            
            assert result_path == temp_file
            assert needs_cleanup is False
            mock_process.assert_called_once_with(temp_file)

    @pytest.mark.asyncio
    async def test_get_audio_metadata(self, temp_file):
        """Test getting audio metadata."""
        expected_metadata = {
            "sample_rate": 16000,
            "channels": 1,
            "codec_name": "pcm_s16le",
            "duration": 1.0,
            "bit_rate": 256000
        }
        
        with patch.object(audio_service.audio_processor, 'validate_audio_file') as mock_validate:
            mock_validate.return_value = expected_metadata
            
            metadata = await audio_service.get_audio_metadata(
                temp_file, request_id="test-123"
            )
            
            assert metadata == expected_metadata
            mock_validate.assert_called_once_with(temp_file)

    @pytest.mark.asyncio
    async def test_cleanup_file(self, temp_file):
        """Test file cleanup."""
        with patch.object(audio_service.audio_processor, 'cleanup_temp_file') as mock_cleanup:
            await audio_service.cleanup_file(temp_file, request_id="test-123")
            mock_cleanup.assert_called_once_with(temp_file)

    def test_is_format_supported(self):
        """Test format support checking."""
        assert audio_service.is_format_supported("wav")
        assert audio_service.is_format_supported("mp3")
        assert audio_service.is_format_supported("WAV")  # Case insensitive
        assert not audio_service.is_format_supported("txt")

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = audio_service.get_supported_formats()
        expected_formats = {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}
        assert set(formats) == expected_formats

    @pytest.mark.asyncio
    async def test_validate_audio_content(self, temp_file):
        """Test validating audio content."""
        with patch.object(audio_service.audio_processor, 'validate_audio_file') as mock_validate:
            mock_validate.return_value = {"sample_rate": 16000}
            
            result = await audio_service.validate_audio_content(
                temp_file, request_id="test-123"
            )
            
            assert result is True
            mock_validate.assert_called_once_with(temp_file)


class TestTranscriptionService:
    """Test transcription service."""

    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self):
        """Test successful audio transcription."""
        content = b"fake audio content"
        filename = "test.wav"
        request = TranscriptionRequest(model="whisper-1")
        
        with patch.object(transcription_service.audio_processor, 'save_uploaded_file') as mock_save, \
             patch.object(transcription_service.audio_processor, 'process_audio_file') as mock_process, \
             patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe, \
             patch.object(transcription_service, '_cleanup_files') as mock_cleanup:
            
            mock_save.return_value = Path("/tmp/uploaded.wav")
            mock_process.return_value = (Path("/tmp/processed.wav"), True)
            mock_transcribe.return_value = ["The quick brown fox jumped over the lazy dog."]
            mock_cleanup.return_value = None
            
            response = await transcription_service.transcribe_audio(
                content, filename, request, request_id="test-123"
            )
            
            assert isinstance(response, TranscriptionResponse)
            assert response.text == "The quick brown fox jumped over the lazy dog."
            assert response.usage.input_tokens == 1
            assert response.usage.output_tokens == 1
            assert response.usage.total_tokens == 2

    @pytest.mark.asyncio
    async def test_transcribe_audio_invalid_filename(self):
        """Test transcription with invalid filename."""
        content = b"fake content"
        filename = "test.txt"
        request = TranscriptionRequest(model="whisper-1")
        
        with pytest.raises(AudioValidationError, match="Unsupported file format"):
            await transcription_service.transcribe_audio(
                content, filename, request, request_id="test-123"
            )

    @pytest.mark.asyncio
    async def test_transcribe_audio_file_too_large(self):
        """Test transcription with file too large."""
        content = b"x" * (30 * 1024 * 1024)  # 30MB
        filename = "test.wav"
        request = TranscriptionRequest(model="whisper-1")
        
        with pytest.raises(AudioValidationError, match="File too large"):
            await transcription_service.transcribe_audio(
                content, filename, request, request_id="test-123"
            )

    @pytest.mark.asyncio
    async def test_transcribe_audio_model_not_loaded(self):
        """Test transcription when model is not loaded."""
        content = b"fake audio content"
        filename = "test.wav"
        request = TranscriptionRequest(model="whisper-1")
        
        with patch.object(transcription_service.audio_processor, 'save_uploaded_file') as mock_save, \
             patch.object(transcription_service.audio_processor, 'process_audio_file') as mock_process, \
             patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=False)), \
             patch.object(transcription_service, '_cleanup_files') as mock_cleanup:
            
            mock_save.return_value = Path("/tmp/uploaded.wav")
            mock_process.return_value = (Path("/tmp/processed.wav"), True)
            mock_cleanup.return_value = None
            
            with pytest.raises(ModelNotLoadedError):
                await transcription_service.transcribe_audio(
                    content, filename, request, request_id="test-123"
                )

    @pytest.mark.asyncio
    async def test_transcribe_audio_model_error(self):
        """Test transcription with model error."""
        content = b"fake audio content"
        filename = "test.wav"
        request = TranscriptionRequest(model="whisper-1")
        
        with patch.object(transcription_service.audio_processor, 'save_uploaded_file') as mock_save, \
             patch.object(transcription_service.audio_processor, 'process_audio_file') as mock_process, \
             patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe, \
             patch.object(transcription_service, '_cleanup_files') as mock_cleanup:
            
            mock_save.return_value = Path("/tmp/uploaded.wav")
            mock_process.return_value = (Path("/tmp/processed.wav"), True)
            mock_transcribe.side_effect = Exception("CUDA out of memory")
            mock_cleanup.return_value = None
            
            with pytest.raises(ModelError, match="Transcription failed"):
                await transcription_service.transcribe_audio(
                    content, filename, request, request_id="test-123"
                )

    def test_get_supported_models(self):
        """Test getting supported models."""
        models = transcription_service.get_supported_models()
        expected_models = [
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe", 
            "parakeet-tdt-0.6b-v2",
            "whisper-1"
        ]
        assert models == expected_models

    def test_is_model_supported(self):
        """Test model support checking."""
        assert transcription_service.is_model_supported("whisper-1")
        assert transcription_service.is_model_supported("gpt-4o-transcribe")
        assert not transcription_service.is_model_supported("unknown-model")

    @pytest.mark.asyncio
    async def test_cleanup_files(self):
        """Test file cleanup."""
        uploaded_path = Path("/tmp/uploaded.wav")
        processed_path = Path("/tmp/processed.wav")
        
        with patch.object(transcription_service.audio_processor, 'cleanup_temp_file') as mock_cleanup:
            # Test cleanup when files are different
            await transcription_service._cleanup_files(uploaded_path, processed_path, True)
            
            assert mock_cleanup.call_count == 2
            mock_cleanup.assert_any_call(uploaded_path)
            mock_cleanup.assert_any_call(processed_path)

    @pytest.mark.asyncio
    async def test_cleanup_files_same_path(self):
        """Test file cleanup when paths are the same."""
        file_path = Path("/tmp/file.wav")
        
        with patch.object(transcription_service.audio_processor, 'cleanup_temp_file') as mock_cleanup:
            # Test cleanup when files are the same
            await transcription_service._cleanup_files(file_path, file_path, False)
            
            # Should only cleanup once
            assert mock_cleanup.call_count == 1
            mock_cleanup.assert_called_with(file_path)