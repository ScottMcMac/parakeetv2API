"""Integration tests for model behavior."""

import pytest
from pathlib import Path

from src.core.model_manager import model_manager


class TestModelIntegration:
    """Test actual model behavior without mocks."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_model_returns_expected_format(self):
        """Test that the model returns the expected format.
        
        This test ensures that our assumptions about the model's return
        format are correct. It helps catch issues where the real model
        behavior differs from our mocked behavior in unit tests.
        """
        # Find a test audio file
        test_file = Path("./tests/audio_files/test_wav_16000Hz_mono.wav")
        if not test_file.exists():
            pytest.skip("Test audio file not found")
        
        # Load the model
        if not model_manager.is_loaded:
            model_manager.load_model()
        
        # Test the transcription
        result = model_manager.transcribe(test_file)
        
        # Verify the return format matches our expectations
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert len(result) > 0, "Expected non-empty result"
        assert isinstance(result[0], str), f"Expected string, got {type(result[0])}"
        
        # Verify the content is reasonable
        text = result[0]
        assert len(text) > 0, "Expected non-empty transcription"
        assert isinstance(text, str), "Transcription should be a string"
        
        # The test files should transcribe to something like "The quick brown fox..."
        # We'll be lenient with the exact text since models can vary
        assert len(text.split()) > 3, "Expected transcription with multiple words"

    @pytest.mark.slow
    def test_model_warmup_behavior(self):
        """Test that model warmup works correctly.
        
        This ensures the warmup process doesn't break when using the real model.
        """
        # Unload model if loaded
        if model_manager.is_loaded:
            model_manager.unload_model()
        
        # Load model (this triggers warmup)
        model_manager.load_model()
        
        # Verify model is loaded and ready
        assert model_manager.is_loaded
        
        # Verify we can transcribe immediately after warmup
        test_file = Path("./tests/audio_files/test_wav_16000Hz_mono.wav")
        if test_file.exists():
            result = model_manager.transcribe(test_file)
            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], str)