"""Unit tests for configuration."""

import pytest
from pydantic import ValidationError

from src.config import Settings


class TestSettings:
    """Test configuration settings."""

    def test_default_settings(self):
        """Test default settings are valid."""
        settings = Settings()
        assert settings.host == "localhost"
        assert settings.port == 8011
        assert settings.api_prefix == "/v1"
        assert settings.model_name == "nvidia/parakeet-tdt-0.6b-v2"

    def test_host_validation(self):
        """Test host validation."""
        # Valid hosts
        valid_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
        for host in valid_hosts:
            settings = Settings(host=host)
            assert settings.host == host

        # Invalid host should raise error
        with pytest.raises(ValidationError):
            Settings(host="invalid.host.com")

    def test_port_validation(self):
        """Test port validation."""
        # Valid port
        settings = Settings(port=8080)
        assert settings.port == 8080

        # Invalid ports
        with pytest.raises(ValidationError):
            Settings(port=0)
        
        with pytest.raises(ValidationError):
            Settings(port=70000)

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        for level in valid_levels:
            settings = Settings(log_level=level)
            assert settings.log_level == level.upper()

        # Case insensitive
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

        # Invalid log level
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")

    def test_log_format_validation(self):
        """Test log format validation."""
        # Valid formats
        settings = Settings(log_format="json")
        assert settings.log_format == "json"
        
        settings = Settings(log_format="text")
        assert settings.log_format == "text"

        # Case insensitive
        settings = Settings(log_format="JSON")
        assert settings.log_format == "json"

        # Invalid format
        with pytest.raises(ValidationError):
            Settings(log_format="invalid")

    def test_cuda_visible_devices_property(self):
        """Test CUDA_VISIBLE_DEVICES property."""
        # Without GPU device
        settings = Settings()
        assert settings.cuda_visible_devices is None

        # With GPU device
        settings = Settings(gpu_device=1)
        assert settings.cuda_visible_devices == "1"

        settings = Settings(gpu_device=0)
        assert settings.cuda_visible_devices == "0"

    def test_gpu_device_validation(self):
        """Test GPU device validation."""
        # Valid GPU devices
        settings = Settings(gpu_device=0)
        assert settings.gpu_device == 0

        settings = Settings(gpu_device=3)
        assert settings.gpu_device == 3

        # Invalid GPU device
        with pytest.raises(ValidationError):
            Settings(gpu_device=-1)

    def test_file_size_validation(self):
        """Test file size validation."""
        # Valid file size
        settings = Settings(max_audio_file_size=10 * 1024 * 1024)  # 10MB
        assert settings.max_audio_file_size == 10 * 1024 * 1024

        # Invalid file size
        with pytest.raises(ValidationError):
            Settings(max_audio_file_size=0)

        with pytest.raises(ValidationError):
            Settings(max_audio_file_size=-1)