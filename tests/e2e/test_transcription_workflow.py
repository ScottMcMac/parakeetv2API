"""End-to-end tests for transcription workflow with real audio files."""

import io
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
from fastapi import status

from src.services import transcription_service
from src.utils.validators import compare_transcriptions


class TestE2ETranscription:
    """End-to-end transcription tests with test audio files."""

    @pytest.fixture
    def expected_transcription(self):
        """Expected transcription text for test files."""
        return "The quick brown fox jumped over the lazy dog"

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_transcribe_real_audio_files(self, async_client, test_audio_dir, expected_transcription):
        """Test transcription with real audio files from test directory."""
        if not test_audio_dir.exists():
            pytest.skip("Test audio directory not found")

        # Mock the model manager and transcription to return expected text
        # Since we don't have the actual model loaded in tests
        with patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe:
            
            mock_transcribe.return_value = [expected_transcription]
            
            # Test different audio formats
            test_files = [
                "test_wav_16000Hz_mono.wav",
                "test_mp3_16000Hz_mono.mp3", 
                "test_flac_8000Hz_mono.flac",
                "test_m4a_16000Hz_mono.m4a",
            ]
            
            for filename in test_files:
                file_path = test_audio_dir / filename
                if not file_path.exists():
                    continue
                    
                with open(file_path, "rb") as f:
                    audio_content = f.read()
                
                files = {"file": (filename, io.BytesIO(audio_content), "audio/*")}
                data = {"model": "whisper-1"}
                
                response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
                
                assert response.status_code == status.HTTP_200_OK, f"Failed for {filename}"
                result = response.json()
                
                assert "text" in result
                # Use our comparison function to handle variations
                assert compare_transcriptions(result["text"], expected_transcription, strict=False), \
                    f"Transcription mismatch for {filename}. Got: {result['text']}"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_transcribe_various_sample_rates(self, async_client, test_audio_dir, expected_transcription):
        """Test transcription with different sample rates."""
        if not test_audio_dir.exists():
            pytest.skip("Test audio directory not found")

        with patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe:
            
            mock_transcribe.return_value = [expected_transcription]
            
            # Test files with different sample rates
            sample_rate_files = [
                "test_wav_8000Hz_mono.wav",
                "test_wav_16000Hz_mono.wav",
                "test_wav_24000Hz_mono.wav",
                "test_wav_48000Hz_mono.wav",
            ]
            
            for filename in sample_rate_files:
                file_path = test_audio_dir / filename
                if not file_path.exists():
                    continue
                
                with open(file_path, "rb") as f:
                    audio_content = f.read()
                
                files = {"file": (filename, io.BytesIO(audio_content), "audio/wav")}
                data = {"model": "whisper-1"}
                
                response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
                
                assert response.status_code == status.HTTP_200_OK, f"Failed for {filename}"
                result = response.json()
                assert compare_transcriptions(result["text"], expected_transcription, strict=False)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_transcribe_stereo_vs_mono(self, async_client, test_audio_dir, expected_transcription):
        """Test transcription with stereo vs mono files."""
        if not test_audio_dir.exists():
            pytest.skip("Test audio directory not found")

        with patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe:
            
            mock_transcribe.return_value = [expected_transcription]
            
            # Test mono vs stereo
            channel_files = [
                "test_wav_16000Hz_mono.wav",
                "test_wav_16000Hz_stereo.wav",
            ]
            
            for filename in channel_files:
                file_path = test_audio_dir / filename
                if not file_path.exists():
                    continue
                
                with open(file_path, "rb") as f:
                    audio_content = f.read()
                
                files = {"file": (filename, io.BytesIO(audio_content), "audio/wav")}
                data = {"model": "whisper-1"}
                
                response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
                
                assert response.status_code == status.HTTP_200_OK, f"Failed for {filename}"
                result = response.json()
                assert compare_transcriptions(result["text"], expected_transcription, strict=False)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_transcribe_different_bit_depths(self, async_client, test_audio_dir, expected_transcription):
        """Test transcription with different bit depths."""
        if not test_audio_dir.exists():
            pytest.skip("Test audio directory not found")

        with patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe:
            
            mock_transcribe.return_value = [expected_transcription]
            
            # Test different bit depths
            bit_depth_files = [
                "test_wav_16000Hz_mono_16bit.wav",
                "test_wav_16000Hz_mono_32bit.wav",
            ]
            
            for filename in bit_depth_files:
                file_path = test_audio_dir / filename
                if not file_path.exists():
                    continue
                
                with open(file_path, "rb") as f:
                    audio_content = f.read()
                
                files = {"file": (filename, io.BytesIO(audio_content), "audio/wav")}
                data = {"model": "whisper-1"}
                
                response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
                
                assert response.status_code == status.HTTP_200_OK, f"Failed for {filename}"
                result = response.json()
                assert compare_transcriptions(result["text"], expected_transcription, strict=False)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_transcribe_all_supported_formats(self, async_client, test_audio_dir, expected_transcription):
        """Test transcription with all supported audio formats."""
        if not test_audio_dir.exists():
            pytest.skip("Test audio directory not found")

        with patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe:
            
            mock_transcribe.return_value = [expected_transcription]
            
            # Map of extensions to MIME types
            format_mimes = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".flac": "audio/flac",
                ".m4a": "audio/mp4",
                ".ogg": "audio/ogg",
                ".webm": "audio/webm",
                ".mp4": "video/mp4",
                ".mpeg": "audio/mpeg",
                ".mpga": "audio/mpeg",
            }
            
            # Find all audio files in test directory
            audio_files = []
            for ext in format_mimes.keys():
                audio_files.extend(test_audio_dir.glob(f"*{ext}"))
            
            if not audio_files:
                pytest.skip("No test audio files found")
            
            for file_path in audio_files[:5]:  # Limit to first 5 files for speed
                with open(file_path, "rb") as f:
                    audio_content = f.read()
                
                mime_type = format_mimes.get(file_path.suffix.lower(), "audio/*")
                files = {"file": (file_path.name, io.BytesIO(audio_content), mime_type)}
                data = {"model": "whisper-1"}
                
                response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
                
                assert response.status_code == status.HTTP_200_OK, f"Failed for {file_path.name}"
                result = response.json()
                assert compare_transcriptions(result["text"], expected_transcription, strict=False)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_transcribe_with_different_models(self, async_client, test_audio_dir, expected_transcription):
        """Test transcription with different model aliases."""
        if not test_audio_dir.exists():
            pytest.skip("Test audio directory not found")

        # Find a test audio file
        audio_file = None
        for pattern in ["*.wav", "*.mp3", "*.flac"]:
            files = list(test_audio_dir.glob(pattern))
            if files:
                audio_file = files[0]
                break
        
        if not audio_file:
            pytest.skip("No test audio files found")

        with patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe:
            
            mock_transcribe.return_value = [expected_transcription]
            
            # Test all supported model aliases
            models = [
                "whisper-1",
                "gpt-4o-transcribe",
                "gpt-4o-mini-transcribe",
                "parakeet-tdt-0.6b-v2"
            ]
            
            with open(audio_file, "rb") as f:
                audio_content = f.read()
            
            for model in models:
                files = {"file": (audio_file.name, io.BytesIO(audio_content), "audio/*")}
                data = {"model": model}
                
                response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
                
                assert response.status_code == status.HTTP_200_OK, f"Failed for model {model}"
                result = response.json()
                assert compare_transcriptions(result["text"], expected_transcription, strict=False)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_api_workflow(self, async_client, test_audio_dir, expected_transcription):
        """Test complete API workflow: models list, model info, transcription."""
        with patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe:
            
            mock_transcribe.return_value = [expected_transcription]
            
            # 1. List available models
            response = await async_client.get("/v1/models")
            assert response.status_code == status.HTTP_200_OK
            models_data = response.json()
            assert len(models_data["data"]) == 4
            
            # 2. Get info for a specific model
            response = await async_client.get("/v1/models/whisper-1")
            assert response.status_code == status.HTTP_200_OK
            model_info = response.json()
            assert model_info["id"] == "whisper-1"
            
            # 3. Transcribe audio (if test files available)
            if test_audio_dir.exists():
                audio_files = list(test_audio_dir.glob("*.wav"))
                if audio_files:
                    audio_file = audio_files[0]
                    with open(audio_file, "rb") as f:
                        audio_content = f.read()
                    
                    files = {"file": (audio_file.name, io.BytesIO(audio_content), "audio/wav")}
                    data = {"model": "whisper-1"}
                    
                    response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
                    assert response.status_code == status.HTTP_200_OK
                    result = response.json()
                    assert compare_transcriptions(result["text"], expected_transcription, strict=False)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_transcription_response_format(self, async_client, test_audio_dir, expected_transcription):
        """Test transcription response follows OpenAI format exactly."""
        if not test_audio_dir.exists():
            pytest.skip("Test audio directory not found")

        audio_files = list(test_audio_dir.glob("*.wav"))
        if not audio_files:
            pytest.skip("No WAV test files found")

        with patch.object(transcription_service.model_manager.__class__, 'is_loaded', new_callable=lambda: PropertyMock(return_value=True)), \
             patch.object(transcription_service.model_manager, 'transcribe') as mock_transcribe:
            
            mock_transcribe.return_value = [expected_transcription]
            
            audio_file = audio_files[0]
            with open(audio_file, "rb") as f:
                audio_content = f.read()
            
            files = {"file": (audio_file.name, io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            
            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
            
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            
            # Verify exact OpenAI response format
            assert "text" in result
            assert "usage" in result
            
            usage = result["usage"]
            assert usage["type"] == "tokens"
            assert usage["input_tokens"] == 1
            assert usage["output_tokens"] == 1
            assert usage["total_tokens"] == 2
            
            assert "input_token_details" in usage
            token_details = usage["input_token_details"]
            assert token_details["text_tokens"] == 0
            assert token_details["audio_tokens"] == 1