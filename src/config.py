"""Configuration management for parakeetv2API."""

from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server configuration
    host: str = Field(
        default="localhost",
        description="Host to bind the server to. Use '0.0.0.0' for external access.",
    )
    port: int = Field(
        default=8011,
        description="Port to bind the server to.",
        ge=1,
        le=65535,
    )

    # GPU configuration
    gpu_device: Optional[int] = Field(
        default=None,
        description="GPU device index to use (e.g., 0, 1, 2, 3). None for automatic selection.",
        ge=0,
    )

    # Model configuration
    model_name: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v2",
        description="Name of the ASR model to load.",
    )
    model_cache_dir: Optional[str] = Field(
        default=None,
        description="Directory to cache downloaded models.",
    )

    # Audio processing
    max_audio_file_size: int = Field(
        default=25 * 1024 * 1024,  # 25 MB
        description="Maximum audio file size in bytes.",
        gt=0,
    )
    temp_dir: Optional[str] = Field(
        default=None,
        description="Directory for temporary audio files. Uses system temp if not specified.",
    )

    # API configuration
    api_prefix: str = Field(
        default="/v1",
        description="API route prefix.",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins.",
    )
    max_concurrent_requests: int = Field(
        default=10,
        description="Maximum number of concurrent transcription requests.",
        gt=0,
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    log_format: str = Field(
        default="json",
        description="Log format (json or text).",
    )

    # Development
    debug: bool = Field(
        default=False,
        description="Enable debug mode.",
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development.",
    )

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host configuration."""
        valid_hosts = ["localhost", "127.0.0.1", "0.0.0.0"]
        if v not in valid_hosts and not v.startswith("192.168.") and not v.startswith("10."):
            raise ValueError(
                f"Host must be one of {valid_hosts} or a local network address"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        valid_formats = ["json", "text"]
        v_lower = v.lower()
        if v_lower not in valid_formats:
            raise ValueError(f"Log format must be one of {valid_formats}")
        return v_lower

    @property
    def cuda_visible_devices(self) -> Optional[str]:
        """Get CUDA_VISIBLE_DEVICES environment variable value."""
        if self.gpu_device is not None:
            return str(self.gpu_device)
        return None


# Global settings instance
settings = Settings()