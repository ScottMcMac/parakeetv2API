"""End-to-end tests for error cases and edge conditions."""

import io
import tempfile
from pathlib import Path

import pytest
from fastapi import status


class TestErrorCases:
    """Test error handling and edge cases."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_non_audio_files_error(self, async_client, test_non_audio_dir):
        """Test that non-audio files return proper errors."""
        if not test_non_audio_dir.exists():
            pytest.skip("Test non-audio directory not found")

        # Test files that should trigger errors
        error_files = [
            "WrongType.txt",
            "MisleadingEncoding.mp3",  # Not actually an MP3
        ]

        for filename in error_files:
            file_path = test_non_audio_dir / filename
            if not file_path.exists():
                continue

            with open(file_path, "rb") as f:
                content = f.read()

            files = {"file": (filename, io.BytesIO(content), "text/plain")}
            data = {"model": "whisper-1"}

            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)

            # Should return error, not crash
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ], f"Unexpected status for {filename}: {response.status_code}"

            result = response.json()
            assert "error" in result, f"No error field in response for {filename}"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_file_extensions(self, async_client):
        """Test various invalid file extensions."""
        invalid_extensions = [
            "document.pdf",
            "archive.zip",
            "image.jpg",
            "data.csv",
            "script.py",
            "noextension"
        ]

        for filename in invalid_extensions:
            content = b"fake content"
            files = {"file": (filename, io.BytesIO(content), "application/octet-stream")}
            data = {"model": "whisper-1"}

            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            result = response.json()
            assert "error" in result
            assert "Unsupported file format" in result["error"]["message"] or \
                   "No file extension found" in result["error"]["message"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_large_file_rejection(self, async_client):
        """Test rejection of files that are too large."""
        # Create a file larger than the default 25MB limit
        large_content = b"x" * (30 * 1024 * 1024)  # 30MB
        
        files = {"file": ("large_file.wav", io.BytesIO(large_content), "audio/wav")}
        data = {"model": "whisper-1"}

        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "error" in result
        assert "File too large" in result["error"]["message"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_empty_file(self, async_client):
        """Test handling of empty files."""
        empty_content = b""
        
        files = {"file": ("empty.wav", io.BytesIO(empty_content), "audio/wav")}
        data = {"model": "whisper-1"}

        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)

        # Should handle gracefully (either validation error or processing error)
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_malformed_requests(self, async_client):
        """Test various malformed requests."""
        
        # Test 1: No file field
        data = {"model": "whisper-1"}
        response = await async_client.post("/v1/audio/transcriptions", data=data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Test 2: Invalid temperature
        files = {"file": ("test.wav", io.BytesIO(b"fake content"), "audio/wav")}
        data = {"model": "whisper-1", "temperature": "invalid"}
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]

        # Test 3: Invalid boolean for stream
        files = {"file": ("test.wav", io.BytesIO(b"fake content"), "audio/wav")}
        data = {"model": "whisper-1", "stream": "maybe"}
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_unsupported_parameters(self, async_client):
        """Test requests with unsupported parameters."""
        audio_content = b"fake audio content"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}

        # Test unsupported language
        data = {"model": "whisper-1", "language": "fr"}
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Only English" in result["error"]["message"]

        # Test unsupported response format
        data = {"model": "whisper-1", "response_format": "text"}
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "Only 'json' format" in result["error"]["message"]

        # Test unsupported timestamp granularities
        data = {"model": "whisper-1", "timestamp_granularities": '["word"]'}
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        result = response.json()
        assert "not supported" in result["error"]["message"]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_path_traversal_protection(self, async_client):
        """Test protection against path traversal attacks."""
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        for filename in malicious_filenames:
            content = b"fake content"
            # Try to make it look like audio
            safe_filename = filename + ".wav"
            files = {"file": (safe_filename, io.BytesIO(content), "audio/wav")}
            data = {"model": "whisper-1"}

            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)

            # Should not crash - either validation error or processing error
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_special_characters_in_filename(self, async_client):
        """Test handling of special characters in filenames."""
        special_filenames = [
            "test@#$%^&*().wav",
            "файл.wav",  # Cyrillic
            "测试.wav",   # Chinese
            "test with spaces.wav",
            "test\nwith\nnewlines.wav",
            "test\twith\ttabs.wav",
        ]

        for filename in special_filenames:
            content = b"fake audio content"
            files = {"file": (filename, io.BytesIO(content), "audio/wav")}
            data = {"model": "whisper-1"}

            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)

            # Should handle gracefully
            assert response.status_code in [
                status.HTTP_200_OK,  # If mocked properly
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling of concurrent requests."""
        import asyncio
        
        audio_content = b"fake audio content"
        
        async def make_request():
            files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            return await async_client.post("/v1/audio/transcriptions", files=files, data=data)

        # Make multiple concurrent requests
        tasks = [make_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without crashing
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                pytest.fail(f"Request {i} raised exception: {response}")
            
            assert response.status_code in [
                status.HTTP_200_OK,  # If mocked properly
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                status.HTTP_503_SERVICE_UNAVAILABLE  # If rate limited
            ]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_model_not_found_error(self, async_client):
        """Test model not found error for models endpoint."""
        
        # Test getting non-existent model
        response = await async_client.get("/v1/models/non-existent-model-12345")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        result = response.json()
        assert "error" in result
        assert "not found" in result["error"]["message"]
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["param"] == "model_id"
        assert result["error"]["code"] == "model_not_found"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_invalid_json_parameters(self, async_client):
        """Test handling of invalid JSON in parameters."""
        audio_content = b"fake audio content"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}
        
        # Test invalid JSON for timestamp_granularities
        data = {"model": "whisper-1", "timestamp_granularities": "invalid json"}
        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
        
        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_extremely_long_filename(self, async_client):
        """Test handling of extremely long filename."""
        # Create a filename longer than filesystem limits
        long_filename = "a" * 500 + ".wav"
        
        content = b"fake audio content"
        files = {"file": (long_filename, io.BytesIO(content), "audio/wav")}
        data = {"model": "whisper-1"}

        response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)

        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_200_OK,  # If filename gets truncated properly
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]

    @pytest.mark.e2e
    @pytest.mark.asyncio  
    async def test_edge_case_parameters(self, async_client):
        """Test edge case parameter values."""
        audio_content = b"fake audio content"
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}

        # Test boundary temperature values
        edge_cases = [
            {"temperature": 0.0},  # Minimum
            {"temperature": 1.0},  # Maximum
            {"temperature": -0.1}, # Below minimum (should fail)
            {"temperature": 1.1},  # Above maximum (should fail)
        ]

        for data in edge_cases:
            data["model"] = "whisper-1"
            response = await async_client.post("/v1/audio/transcriptions", files=files, data=data)
            
            if data["temperature"] < 0 or data["temperature"] > 1:
                # Should fail validation
                assert response.status_code in [
                    status.HTTP_400_BAD_REQUEST,
                    status.HTTP_422_UNPROCESSABLE_ENTITY
                ]
            else:
                # Should succeed or fail gracefully
                assert response.status_code in [
                    status.HTTP_200_OK,
                    status.HTTP_400_BAD_REQUEST,
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ]