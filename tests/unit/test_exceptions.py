"""Unit tests for custom exceptions."""

import pytest

from src.core.exceptions import (
    AudioProcessingError,
    AudioValidationError,
    ModelError,
    ModelNotLoadedError,
    ParakeetAPIException,
    UnsupportedParameterError,
)


class TestParakeetAPIException:
    """Test base exception class."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        exc = ParakeetAPIException("Test error")
        assert exc.message == "Test error"
        assert exc.status_code == 500
        assert exc.details == {}

    def test_exception_with_status_code(self):
        """Test exception with custom status code."""
        exc = ParakeetAPIException("Test error", status_code=400)
        assert exc.message == "Test error"
        assert exc.status_code == 400
        assert exc.details == {}

    def test_exception_with_details(self):
        """Test exception with details."""
        details = {"param": "test", "value": "invalid"}
        exc = ParakeetAPIException("Test error", details=details)
        assert exc.message == "Test error"
        assert exc.status_code == 500
        assert exc.details == details

    def test_exception_inheritance(self):
        """Test exception inherits from Exception."""
        exc = ParakeetAPIException("Test error")
        assert isinstance(exc, Exception)
        assert str(exc) == "Test error"


class TestAudioValidationError:
    """Test audio validation error."""

    def test_basic_audio_validation_error(self):
        """Test basic audio validation error."""
        exc = AudioValidationError("Invalid audio file")
        assert exc.message == "Invalid audio file"
        assert exc.status_code == 400
        assert exc.details == {}

    def test_audio_validation_error_with_details(self):
        """Test audio validation error with details."""
        details = {"format": "txt", "expected": "wav"}
        exc = AudioValidationError("Invalid format", details=details)
        assert exc.message == "Invalid format"
        assert exc.status_code == 400
        assert exc.details == details

    def test_inheritance(self):
        """Test inheritance from ParakeetAPIException."""
        exc = AudioValidationError("Test error")
        assert isinstance(exc, ParakeetAPIException)
        assert isinstance(exc, Exception)


class TestAudioProcessingError:
    """Test audio processing error."""

    def test_basic_audio_processing_error(self):
        """Test basic audio processing error."""
        exc = AudioProcessingError("Processing failed")
        assert exc.message == "Processing failed"
        assert exc.status_code == 500
        assert exc.details == {}

    def test_audio_processing_error_with_details(self):
        """Test audio processing error with details."""
        details = {"ffmpeg_error": "Invalid format"}
        exc = AudioProcessingError("FFmpeg failed", details=details)
        assert exc.message == "FFmpeg failed"
        assert exc.status_code == 500
        assert exc.details == details


class TestModelError:
    """Test model error."""

    def test_basic_model_error(self):
        """Test basic model error."""
        exc = ModelError("Model inference failed")
        assert exc.message == "Model inference failed"
        assert exc.status_code == 500
        assert exc.details == {}

    def test_model_error_with_details(self):
        """Test model error with details."""
        details = {"model": "parakeet", "error": "CUDA out of memory"}
        exc = ModelError("Inference failed", details=details)
        assert exc.message == "Inference failed"
        assert exc.status_code == 500
        assert exc.details == details


class TestModelNotLoadedError:
    """Test model not loaded error."""

    def test_model_not_loaded_error(self):
        """Test model not loaded error."""
        exc = ModelNotLoadedError()
        assert "Model is not loaded" in exc.message
        assert exc.status_code == 500
        assert exc.details == {"error_code": "model_not_loaded"}


class TestUnsupportedParameterError:
    """Test unsupported parameter error."""

    def test_unsupported_parameter_error(self):
        """Test unsupported parameter error."""
        exc = UnsupportedParameterError(
            parameter="language",
            value="fr",
            reason="Only English is supported"
        )
        
        expected_message = "Unsupported value for parameter 'language': fr. Only English is supported"
        assert exc.message == expected_message
        assert exc.status_code == 400
        assert exc.details == {
            "parameter": "language",
            "value": "fr",
            "reason": "Only English is supported"
        }

    def test_unsupported_parameter_error_with_none_value(self):
        """Test unsupported parameter error with None value."""
        exc = UnsupportedParameterError(
            parameter="test_param",
            value=None,
            reason="None values not allowed"
        )
        
        expected_message = "Unsupported value for parameter 'test_param': None. None values not allowed"
        assert exc.message == expected_message
        assert exc.details["value"] is None