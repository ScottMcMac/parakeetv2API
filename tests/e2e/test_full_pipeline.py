"""Comprehensive end-to-end tests for the full pipeline with real server and all test files."""

import asyncio
import io
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Dict, List

import httpx
import pytest
import requests
from fastapi import status

from src.utils.validators import compare_transcriptions


@pytest.fixture(scope="module")
def server_url():
    """Start a real server instance for testing."""
    # Create a temporary config for testing
    test_config = {
        "PORT": "8012",  # Use different port to avoid conflicts
        "HOST": "127.0.0.1",
        "LOG_LEVEL": "WARNING",  # Reduce noise during tests
        "AUDIO_MAX_SIZE_MB": "25",
        "TEMP_CLEANUP_ENABLED": "true"
    }
    
    # Start the server as a subprocess
    env = os.environ.copy()
    env.update(test_config)
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.main:app", 
         "--host", "127.0.0.1", "--port", "8012"],
        env=env,
        stdout=subprocess.DEVNULL,  # Avoid blocking on pipe buffer
        stderr=subprocess.DEVNULL   # Avoid blocking on pipe buffer
    )
    
    # Wait for server to start
    server_url = "http://127.0.0.1:8012"
    max_retries = 60  # Increase timeout for model loading
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                # Extra wait to ensure model is fully loaded
                time.sleep(5)
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.RequestException):
            pass
        
        time.sleep(1)
        retry_count += 1
    else:
        server_process.terminate()
        pytest.fail("Server failed to start within timeout")
    
    yield server_url
    
    # Cleanup: terminate the server
    server_process.terminate()
    try:
        server_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_process.kill()


class TestFullPipeline:
    """Test the complete pipeline with a real server against all test files."""

    @pytest.fixture
    def expected_transcription(self):
        """Expected transcription text for audio test files."""
        return "The quick brown fox jumped over the lazy dog"

    @pytest.mark.e2e
    @pytest.mark.slow  
    @pytest.mark.asyncio
    async def test_all_audio_files_transcription_batch_1(self, server_url, expected_transcription):
        """Test transcription of audio files (batch 1: files 1-25)."""
        await self._test_audio_files_batch(server_url, expected_transcription, 0, 25)

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_all_audio_files_transcription_batch_2(self, server_url, expected_transcription):
        """Test transcription of audio files (batch 2: files 26-50)."""
        await self._test_audio_files_batch(server_url, expected_transcription, 25, 50)

    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_all_audio_files_transcription_batch_3(self, server_url, expected_transcription):
        """Test transcription of audio files (batch 3: files 51-73)."""
        await self._test_audio_files_batch(server_url, expected_transcription, 50, 73)

    async def _test_audio_files_batch(self, server_url, expected_transcription, start_idx, end_idx):
        """Helper method to test a batch of audio files."""
        audio_dir = Path("./tests/audio_files/")
        if not audio_dir.exists():
            pytest.skip("Audio files directory not found")

        # Get all audio files
        audio_files = []
        for pattern in ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg", "*.webm", "*.mp4", "*.mpeg", "*.mpga"]:
            audio_files.extend(audio_dir.glob(pattern))

        if not audio_files:
            pytest.skip("No audio files found in test directory")

        # Get the batch
        batch_files = audio_files[start_idx:end_idx]
        if not batch_files:
            pytest.skip(f"No files in batch {start_idx}-{end_idx}")

        success_count = 0
        failure_count = 0
        failures: List[Dict] = []

        # Process files sequentially to avoid overwhelming the server
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, audio_file in enumerate(batch_files):
                print(f"Processing file {start_idx + i + 1}/{len(audio_files)}: {audio_file.name}")
                try:
                    with open(audio_file, "rb") as f:
                        audio_content = f.read()

                    # Determine MIME type based on extension
                    mime_types = {
                        ".wav": "audio/wav",
                        ".mp3": "audio/mpeg", 
                        ".flac": "audio/flac",
                        ".m4a": "audio/mp4",
                        ".ogg": "audio/ogg",
                        ".webm": "audio/webm",
                        ".mp4": "video/mp4",
                        ".mpeg": "audio/mpeg",
                        ".mpga": "audio/mpeg"
                    }
                    mime_type = mime_types.get(audio_file.suffix.lower(), "audio/*")

                    files = {"file": (audio_file.name, io.BytesIO(audio_content), mime_type)}
                    data = {"model": "whisper-1"}

                    response = await client.post(f"{server_url}/v1/audio/transcriptions", 
                                               files=files, data=data)
                    
                    # Small delay between requests to avoid overwhelming the server
                    await asyncio.sleep(0.05)

                    if response.status_code == status.HTTP_200_OK:
                        result = response.json()
                        
                        # Verify response structure with detailed error handling
                        if "text" not in result:
                            failure_count += 1
                            failures.append({
                                "file": audio_file.name,
                                "reason": f"No 'text' field in response. Response: {result}",
                                "response": result
                            })
                            continue
                            
                        if "usage" not in result:
                            failure_count += 1
                            failures.append({
                                "file": audio_file.name,
                                "reason": f"No 'usage' field in response. Response: {result}",
                                "response": result
                            })
                            continue
                        
                        # Verify transcription content (case-insensitive, flexible comparison)
                        transcribed_text = result["text"]
                        
                        if compare_transcriptions(transcribed_text, expected_transcription, strict=False):
                            success_count += 1
                        else:
                            failure_count += 1
                            failures.append({
                                "file": audio_file.name,
                                "expected": expected_transcription,
                                "actual": transcribed_text,
                                "reason": "Transcription mismatch"
                            })
                    else:
                        failure_count += 1
                        failures.append({
                            "file": audio_file.name,
                            "status_code": response.status_code,
                            "response": response.text,
                            "reason": "HTTP error"
                        })

                except Exception as e:
                    failure_count += 1
                    failures.append({
                        "file": audio_file.name,
                        "error": str(e),
                        "reason": "Exception during test"
                    })

        # Report results
        print(f"\nBatch {start_idx+1}-{end_idx} results:")
        print(f"Files processed: {len(batch_files)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failure_count}")
        
        if failures:
            print(f"Failures:")
            for failure in failures:
                print(f"  {failure['file']}: {failure['reason']}")
                if 'actual' in failure:
                    print(f"    Expected: {failure['expected']}")
                    print(f"    Actual: {failure['actual']}")

        # Expect very high success rate (allow 1-2 minor transcription variations)
        success_rate = success_count / len(batch_files) if len(batch_files) > 0 else 0
        assert success_rate >= 0.95, f"Batch {start_idx+1}-{end_idx} success rate too low: {success_rate:.1%} ({success_count}/{len(batch_files)}). Failures: {failures[:3]}"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_all_non_audio_files_error_handling(self, server_url):
        """Test that every file in tests/non_audio_files/ returns proper errors."""
        non_audio_dir = Path("./tests/non_audio_files/")
        if not non_audio_dir.exists():
            pytest.skip("Non-audio files directory not found")

        # Get all non-audio files
        non_audio_files = list(non_audio_dir.glob("*"))
        if not non_audio_files:
            pytest.skip("No non-audio files found in test directory")

        success_count = 0
        failure_count = 0
        failures: List[Dict] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for non_audio_file in non_audio_files:
                if non_audio_file.is_file():
                    try:
                        with open(non_audio_file, "rb") as f:
                            content = f.read()

                        files = {"file": (non_audio_file.name, io.BytesIO(content), "application/octet-stream")}
                        data = {"model": "whisper-1"}

                        response = await client.post(f"{server_url}/v1/audio/transcriptions", 
                                                   files=files, data=data)

                        # Should return an error, not success
                        if response.status_code in [
                            status.HTTP_400_BAD_REQUEST,
                            status.HTTP_422_UNPROCESSABLE_ENTITY,
                            status.HTTP_500_INTERNAL_SERVER_ERROR
                        ]:
                            result = response.json()
                            if "error" in result:
                                success_count += 1
                            else:
                                failure_count += 1
                                failures.append({
                                    "file": non_audio_file.name,
                                    "reason": "Error response missing 'error' field",
                                    "response": result
                                })
                        else:
                            failure_count += 1
                            failures.append({
                                "file": non_audio_file.name,
                                "status_code": response.status_code,
                                "reason": "Unexpected status code (should be error)",
                                "response": response.text
                            })

                    except Exception as e:
                        failure_count += 1
                        failures.append({
                            "file": non_audio_file.name,
                            "error": str(e),
                            "reason": "Exception during test"
                        })

        # Report results
        total_files = len(non_audio_files)
        print(f"\nNon-audio file test results:")
        print(f"Total files: {total_files}")
        print(f"Properly handled errors: {success_count}")
        print(f"Failed to handle: {failure_count}")

        if failures:
            print(f"\nFailures:")
            for failure in failures:
                print(f"  {failure['file']}: {failure['reason']}")

        # All non-audio files should return proper errors
        assert failure_count == 0, f"Some non-audio files were not handled properly: {failures}"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_server_endpoints_health(self, server_url):
        """Test that all server endpoints are working."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test health endpoint
            response = await client.get(f"{server_url}/health")
            assert response.status_code == status.HTTP_200_OK

            # Test models list endpoint
            response = await client.get(f"{server_url}/v1/models")
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert "data" in result
            assert len(result["data"]) == 4

            # Test specific model info
            response = await client.get(f"{server_url}/v1/models/whisper-1")
            assert response.status_code == status.HTTP_200_OK
            result = response.json()
            assert result["id"] == "whisper-1"

            # Test non-existent model
            response = await client.get(f"{server_url}/v1/models/non-existent")
            assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_different_model_aliases(self, server_url):
        """Test that all model aliases work with the same backend."""
        audio_dir = Path("./tests/audio_files/")
        if not audio_dir.exists():
            pytest.skip("Audio files directory not found")

        # Find a test audio file
        audio_file = None
        for pattern in ["test_wav_16000Hz_mono.wav", "*.wav", "*.mp3"]:
            files = list(audio_dir.glob(pattern))
            if files:
                audio_file = files[0]
                break

        if not audio_file:
            pytest.skip("No test audio files found")

        model_aliases = [
            "whisper-1",
            "gpt-4o-transcribe", 
            "gpt-4o-mini-transcribe",
            "parakeet-tdt-0.6b-v2"
        ]

        with open(audio_file, "rb") as f:
            audio_content = f.read()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for model in model_aliases:
                files = {"file": (audio_file.name, io.BytesIO(audio_content), "audio/wav")}
                data = {"model": model}

                response = await client.post(f"{server_url}/v1/audio/transcriptions", 
                                           files=files, data=data)

                # All models should work (they use the same backend)
                assert response.status_code == status.HTTP_200_OK, \
                    f"Model {model} failed with status {response.status_code}: {response.text}"
                
                result = response.json()
                assert "text" in result
                assert "usage" in result

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_performance_under_load(self, server_url):
        """Test server performance with multiple concurrent requests."""
        audio_dir = Path("./tests/audio_files/")
        if not audio_dir.exists():
            pytest.skip("Audio files directory not found")

        # Find a small test file
        audio_file = None
        for pattern in ["test_wav_16000Hz_mono.wav", "*.wav"]:
            files = list(audio_dir.glob(pattern))
            if files:
                audio_file = files[0]
                break

        if not audio_file:
            pytest.skip("No test audio files found")

        with open(audio_file, "rb") as f:
            audio_content = f.read()

        async def make_request(client, request_id):
            """Make a single transcription request."""
            files = {"file": (f"test_{request_id}.wav", io.BytesIO(audio_content), "audio/wav")}
            data = {"model": "whisper-1"}
            
            start_time = time.time()
            response = await client.post(f"{server_url}/v1/audio/transcriptions", 
                                       files=files, data=data)
            end_time = time.time()
            
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == status.HTTP_200_OK
            }

        # Test with 5 concurrent requests
        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = [make_request(client, i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful_requests = 0
        total_time = 0
        max_time = 0

        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Request failed with exception: {result}")
            
            if result["success"]:
                successful_requests += 1
                total_time += result["response_time"]
                max_time = max(max_time, result["response_time"])

        print(f"\nPerformance test results:")
        print(f"Successful requests: {successful_requests}/5")
        print(f"Average response time: {total_time/successful_requests:.2f}s")
        print(f"Max response time: {max_time:.2f}s")

        # Most requests should succeed
        assert successful_requests >= 4, f"Too many failed requests: {5 - successful_requests}"
        
        # Response times should be reasonable (under 30 seconds per request)
        assert max_time < 30, f"Response time too slow: {max_time:.2f}s"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_file_format_edge_cases(self, server_url):
        """Test edge cases with file formats and content."""
        test_cases = [
            # Empty file
            {"filename": "empty.wav", "content": b"", "should_fail": True},
            
            # Very small file
            {"filename": "tiny.wav", "content": b"fake", "should_fail": True},
            
            # File with wrong extension
            {"filename": "notaudio.wav", "content": b"This is not audio content", "should_fail": True},
            
            # Invalid extension  
            {"filename": "test.xyz", "content": b"content", "should_fail": True},
            
            # No extension
            {"filename": "noextension", "content": b"content", "should_fail": True},
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            for test_case in test_cases:
                files = {"file": (test_case["filename"], io.BytesIO(test_case["content"]), "application/octet-stream")}
                data = {"model": "whisper-1"}

                response = await client.post(f"{server_url}/v1/audio/transcriptions", 
                                           files=files, data=data)

                if test_case["should_fail"]:
                    # Should return an error
                    assert response.status_code in [
                        status.HTTP_400_BAD_REQUEST,
                        status.HTTP_422_UNPROCESSABLE_ENTITY, 
                        status.HTTP_500_INTERNAL_SERVER_ERROR
                    ], f"Expected error for {test_case['filename']}, got {response.status_code}"
                    
                    if response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR:
                        result = response.json()
                        assert "error" in result, f"Missing error field for {test_case['filename']}"
                else:
                    # Should succeed
                    assert response.status_code == status.HTTP_200_OK, \
                        f"Expected success for {test_case['filename']}, got {response.status_code}"