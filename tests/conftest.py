"""Pytest configuration and fixtures."""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.main import app


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_audio_file() -> Generator[Path, None, None]:
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = Path(tmp.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def test_audio_dir() -> Path:
    """Get the path to test audio files."""
    return Path(__file__).parent.parent / "tests" / "audio_files"


@pytest.fixture
def test_non_audio_dir() -> Path:
    """Get the path to test non-audio files."""
    return Path(__file__).parent.parent / "tests" / "non_audio_files"


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    # Store original environment
    original_env = dict(os.environ)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_settings(monkeypatch):
    """Fixture to mock application settings."""
    def _mock_settings(**kwargs):
        from src.config import settings
        for key, value in kwargs.items():
            monkeypatch.setattr(settings, key, value)
        return settings
    
    return _mock_settings


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)