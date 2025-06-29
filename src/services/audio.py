"""Audio service for coordinating audio validation and conversion."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.core import audio_processor
from src.core.exceptions import AudioProcessingError, AudioValidationError
from src.utils import sanitize_filename, validate_file_extension, validate_file_size

logger = logging.getLogger(__name__)


class AudioService:
    """Service for coordinating audio validation and conversion operations."""

    def __init__(self):
        """Initialize the audio service."""
        self.audio_processor = audio_processor
        self.supported_formats = {
            "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
        }

    async def validate_and_save_file(
        self,
        file_content: bytes,
        filename: str,
        request_id: Optional[str] = None,
    ) -> Path:
        """
        Validate and save uploaded audio file.

        Args:
            file_content: Audio file content as bytes
            filename: Original filename
            request_id: Optional request ID for logging

        Returns:
            Path to saved file

        Raises:
            AudioValidationError: If validation fails
            AudioProcessingError: If saving fails
        """
        logger.info(
            f"Validating and saving audio file - filename: {filename}, "
            f"size: {len(file_content)} bytes, request_id: {request_id}"
        )

        # Sanitize filename
        safe_filename = sanitize_filename(filename)
        logger.debug(f"Sanitized filename: {safe_filename}")

        # Validate file extension
        try:
            extension = validate_file_extension(safe_filename)
            logger.debug(f"File extension validated: {extension}")
        except AudioValidationError as e:
            logger.warning(f"Invalid file extension: {e.message}")
            raise

        # Validate file size
        try:
            validate_file_size(len(file_content))
            logger.debug(f"File size validated: {len(file_content)} bytes")
        except AudioValidationError as e:
            logger.warning(f"File size validation failed: {e.message}")
            raise

        # Save the file
        try:
            file_path = await self.audio_processor.save_uploaded_file(
                file_content, safe_filename
            )
            logger.info(f"File saved successfully: {file_path}")
            return file_path
        except AudioProcessingError as e:
            logger.error(f"Failed to save file: {e.message}")
            raise

    async def process_for_transcription(
        self,
        file_path: Path,
        request_id: Optional[str] = None,
    ) -> Tuple[Path, bool]:
        """
        Process audio file for transcription (validate and convert if needed).

        Args:
            file_path: Path to audio file
            request_id: Optional request ID for logging

        Returns:
            Tuple of (processed_file_path, needs_cleanup)

        Raises:
            AudioValidationError: If audio validation fails
            AudioProcessingError: If processing fails
        """
        logger.info(
            f"Processing audio for transcription - file: {file_path}, "
            f"request_id: {request_id}"
        )

        try:
            # Process the audio file (validate and convert if needed)
            processed_path, needs_cleanup = await self.audio_processor.process_audio_file(
                file_path
            )

            logger.info(
                f"Audio processing complete - processed: {processed_path}, "
                f"needs_cleanup: {needs_cleanup}, request_id: {request_id}"
            )

            return processed_path, needs_cleanup

        except AudioValidationError as e:
            logger.warning(f"Audio validation failed: {e.message}")
            raise
        except AudioProcessingError as e:
            logger.error(f"Audio processing failed: {e.message}")
            raise

    async def get_audio_metadata(
        self,
        file_path: Path,
        request_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Get metadata for an audio file.

        Args:
            file_path: Path to audio file
            request_id: Optional request ID for logging

        Returns:
            Audio metadata dictionary

        Raises:
            AudioValidationError: If file is not valid audio
        """
        logger.info(
            f"Getting audio metadata - file: {file_path}, request_id: {request_id}"
        )

        try:
            metadata = await self.audio_processor.validate_audio_file(file_path)
            logger.debug(f"Audio metadata retrieved: {metadata}")
            return metadata
        except AudioValidationError as e:
            logger.warning(f"Failed to get audio metadata: {e.message}")
            raise

    async def cleanup_file(
        self,
        file_path: Path,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Clean up a temporary audio file.

        Args:
            file_path: Path to file to clean up
            request_id: Optional request ID for logging
        """
        logger.debug(f"Cleaning up file: {file_path}, request_id: {request_id}")

        try:
            await self.audio_processor.cleanup_temp_file(file_path)
            logger.debug(f"File cleaned up successfully: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {str(e)}")

    def is_format_supported(self, extension: str) -> bool:
        """
        Check if an audio format is supported.

        Args:
            extension: File extension (without dot)

        Returns:
            True if format is supported
        """
        return extension.lower() in self.supported_formats

    def get_supported_formats(self) -> list[str]:
        """
        Get list of supported audio formats.

        Returns:
            List of supported file extensions
        """
        return list(self.supported_formats)

    async def validate_audio_content(
        self,
        file_path: Path,
        request_id: Optional[str] = None,
    ) -> bool:
        """
        Validate that a file contains valid audio content.

        Args:
            file_path: Path to file to validate
            request_id: Optional request ID for logging

        Returns:
            True if file contains valid audio

        Raises:
            AudioValidationError: If validation fails
        """
        logger.debug(f"Validating audio content: {file_path}, request_id: {request_id}")

        try:
            # Use the audio processor to validate
            await self.audio_processor.validate_audio_file(file_path)
            logger.debug(f"Audio content validation passed: {file_path}")
            return True
        except AudioValidationError as e:
            logger.warning(f"Audio content validation failed: {e.message}")
            raise


# Global audio service instance
audio_service = AudioService()