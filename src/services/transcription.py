"""Transcription service for orchestrating audio transcription workflow."""

import logging
from pathlib import Path
from typing import Optional

from src.core import audio_processor, model_manager
from src.core.exceptions import (
    AudioProcessingError,
    AudioValidationError,
    ModelError,
    ModelNotLoadedError,
)
from src.models import TranscriptionRequest, TranscriptionResponse
from src.utils import sanitize_filename, validate_file_extension, validate_file_size

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service for handling audio transcription workflow."""

    def __init__(self):
        """Initialize the transcription service."""
        self.audio_processor = audio_processor
        self.model_manager = model_manager

    async def transcribe_audio(
        self,
        file_content: bytes,
        filename: str,
        request: TranscriptionRequest,
        request_id: Optional[str] = None,
    ) -> TranscriptionResponse:
        """
        Transcribe audio file to text.

        Args:
            file_content: Audio file content as bytes
            filename: Original filename
            request: Transcription request parameters
            request_id: Optional request ID for logging

        Returns:
            Transcription response

        Raises:
            AudioValidationError: If file validation fails
            AudioProcessingError: If audio processing fails
            ModelError: If transcription fails
            ModelNotLoadedError: If model is not loaded
        """
        logger.info(
            f"Starting transcription - model: {request.model}, "
            f"file: {filename}, request_id: {request_id}"
        )

        # Validate and sanitize filename
        safe_filename = sanitize_filename(filename)
        
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

        # Save uploaded file
        try:
            uploaded_file_path = await self.audio_processor.save_uploaded_file(
                file_content, safe_filename
            )
            logger.debug(f"File saved to: {uploaded_file_path}")
        except AudioProcessingError as e:
            logger.error(f"Failed to save uploaded file: {e.message}")
            raise

        # Process and transcribe
        processed_file_path = None
        needs_cleanup = False

        try:
            # Process audio file (validate and convert if needed)
            processed_file_path, needs_cleanup = await self.audio_processor.process_audio_file(
                uploaded_file_path
            )
            logger.debug(
                f"Audio processing complete - processed: {processed_file_path}, "
                f"needs_cleanup: {needs_cleanup}"
            )

            # Check if model is loaded
            if not self.model_manager.is_loaded:
                raise ModelNotLoadedError()

            # Transcribe using the model
            transcriptions = self.model_manager.transcribe(processed_file_path)
            
            # Get the first (and only) transcription
            text = transcriptions[0] if transcriptions else ""
            
            logger.info(
                f"Transcription successful - request_id: {request_id}, "
                f"text_length: {len(text)}, model: {request.model}"
            )

            # Create and return response
            return TranscriptionResponse(text=text)

        except (AudioValidationError, AudioProcessingError, ModelError, ModelNotLoadedError):
            # Re-raise expected exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error during transcription: {str(e)}", exc_info=True)
            raise ModelError(f"Transcription failed: {str(e)}")

        finally:
            # Cleanup temporary files
            await self._cleanup_files(uploaded_file_path, processed_file_path, needs_cleanup)

    async def _cleanup_files(
        self,
        uploaded_file_path: Path,
        processed_file_path: Optional[Path],
        needs_cleanup: bool,
    ) -> None:
        """
        Clean up temporary files.

        Args:
            uploaded_file_path: Path to uploaded file
            processed_file_path: Path to processed file (if different)
            needs_cleanup: Whether processed file needs cleanup
        """
        try:
            # Always cleanup uploaded file
            await self.audio_processor.cleanup_temp_file(uploaded_file_path)
            logger.debug(f"Cleaned up uploaded file: {uploaded_file_path}")

            # Cleanup processed file if it's different from uploaded
            if (
                needs_cleanup
                and processed_file_path
                and processed_file_path != uploaded_file_path
            ):
                await self.audio_processor.cleanup_temp_file(processed_file_path)
                logger.debug(f"Cleaned up processed file: {processed_file_path}")

        except Exception as e:
            logger.warning(f"Cleanup error: {str(e)}")

    def get_supported_models(self) -> list[str]:
        """
        Get list of supported model names.

        Returns:
            List of supported model names
        """
        return [
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe", 
            "parakeet-tdt-0.6b-v2",
            "whisper-1"
        ]

    def is_model_supported(self, model_name: str) -> bool:
        """
        Check if a model name is supported.

        Args:
            model_name: Model name to check

        Returns:
            True if model is supported
        """
        return model_name in self.get_supported_models()


# Global transcription service instance
transcription_service = TranscriptionService()