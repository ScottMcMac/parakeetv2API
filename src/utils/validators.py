"""Validation utilities for parakeetv2API."""

import re
from pathlib import Path
from typing import Optional

from src.config import settings
from src.core.audio_processor import SUPPORTED_FORMATS
from src.core.exceptions import AudioValidationError


def validate_file_extension(filename: str) -> str:
    """
    Validate file extension is supported.
    
    Args:
        filename: Name of the file
        
    Returns:
        The file extension (without dot)
        
    Raises:
        AudioValidationError: If extension is not supported
    """
    path = Path(filename)
    extension = path.suffix.lower().lstrip(".")
    
    if not extension:
        raise AudioValidationError(
            "No file extension found. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    
    if extension not in SUPPORTED_FORMATS:
        raise AudioValidationError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    
    return extension


def validate_file_size(size: int) -> None:
    """
    Validate file size is within limits.
    
    Args:
        size: File size in bytes
        
    Raises:
        AudioValidationError: If file is too large
    """
    if size > settings.max_audio_file_size:
        max_mb = settings.max_audio_file_size / (1024 * 1024)
        size_mb = size / (1024 * 1024)
        raise AudioValidationError(
            f"File too large: {size_mb:.1f}MB. Maximum allowed: {max_mb:.1f}MB"
        )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove any path components
    filename = Path(filename).name
    
    # Remove potentially dangerous characters
    # Keep only alphanumeric, dots, hyphens, underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Ensure it has a valid extension
    if '.' not in filename:
        filename = f"{filename}.audio"
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        # Preserve extension
        name, ext = filename.rsplit('.', 1)
        max_name_length = max_length - len(ext) - 1
        filename = f"{name[:max_name_length]}.{ext}"
    
    return filename


def normalize_transcription(text: str) -> str:
    """
    Normalize transcription text for comparison in tests.
    
    Args:
        text: Transcription text
        
    Returns:
        Normalized text (lowercase, no punctuation)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and extra whitespace
    # Keep only letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compare_transcriptions(actual: str, expected: str, strict: bool = False) -> bool:
    """
    Compare two transcription texts.
    
    Args:
        actual: Actual transcription
        expected: Expected transcription
        strict: If True, require exact match. If False, normalize first.
        
    Returns:
        True if transcriptions match
    """
    if strict:
        return actual == expected
    
    return normalize_transcription(actual) == normalize_transcription(expected)