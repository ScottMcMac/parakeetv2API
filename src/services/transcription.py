"""Transcription service for orchestrating audio transcription workflow."""

import time
from pathlib import Path
from typing import Optional

from src.core import audio_processor, model_manager
from src.core.exceptions import (
    AudioProcessingError,
    AudioValidationError,
    ModelError,
    ModelNotLoadedError,
)
from src.core.logging import get_logger, performance_logger
from src.models import TranscriptionRequest, TranscriptionResponse
from src.utils import sanitize_filename, validate_file_extension, validate_file_size

logger = get_logger(__name__)


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
        start_time = time.time()
        
        logger.info(
            "transcription_started",
            model=request.model,
            filename=filename,
            file_size=len(file_content),
            request_id=request_id,
        )

        # Validate and sanitize filename
        safe_filename = sanitize_filename(filename)
        
        # Validate file extension
        try:
            extension = validate_file_extension(safe_filename)
            logger.debug("file_extension_validated", extension=extension)
        except AudioValidationError as e:
            logger.warning("invalid_file_extension", error=e.message, filename=filename)
            raise

        # Validate file size
        try:
            validate_file_size(len(file_content))
            logger.debug("file_size_validated", size_bytes=len(file_content))
        except AudioValidationError as e:
            logger.warning("file_size_validation_failed", error=e.message, size_bytes=len(file_content))
            raise

        # Save uploaded file
        try:
            uploaded_file_path = await self.audio_processor.save_uploaded_file(
                file_content, safe_filename
            )
            logger.debug("file_saved", path=str(uploaded_file_path))
        except AudioProcessingError as e:
            logger.error("file_save_failed", error=e.message)
            raise

        # Process and transcribe
        processed_file_path = None
        needs_cleanup = False

        try:
            # Process audio file (validate and convert if needed)
            processing_start = time.time()
            processed_file_path, needs_cleanup = await self.audio_processor.process_audio_file(
                uploaded_file_path
            )
            processing_duration_ms = (time.time() - processing_start) * 1000
            
            logger.debug(
                "audio_processing_complete",
                processed_path=str(processed_file_path),
                needs_cleanup=needs_cleanup,
                duration_ms=round(processing_duration_ms, 2),
            )
            
            # Log audio processing performance
            performance_logger.log_audio_processing(
                request_id=request_id,
                operation="process_audio",
                duration_ms=processing_duration_ms,
                input_format=extension,
            )

            # Check if model is loaded
            if not self.model_manager.is_loaded:
                raise ModelNotLoadedError()

            # Transcribe using the model
            inference_start = time.time()
            transcriptions = self.model_manager.transcribe(processed_file_path)
            inference_duration_ms = (time.time() - inference_start) * 1000
            
            # Debug logging
            logger.debug(
                "transcription_result",
                transcriptions_type=type(transcriptions).__name__,
                transcriptions_length=len(transcriptions) if transcriptions else 0,
                transcriptions_content=str(transcriptions)[:200] if transcriptions else None,
            )
            
            # Get the first (and only) transcription
            text = transcriptions[0] if transcriptions else ""
            
            # Ensure text is a string
            if not isinstance(text, str):
                logger.warning(
                    "transcription_not_string",
                    text_type=type(text).__name__,
                    text_value=str(text)[:200],
                )
                text = str(text)
            
            # Log model inference performance
            performance_logger.log_model_inference(
                request_id=request_id,
                duration_ms=inference_duration_ms,
                model=request.model,
            )
            
            total_duration_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "transcription_completed",
                request_id=request_id,
                text_length=len(text),
                model=request.model,
                total_duration_ms=round(total_duration_ms, 2),
                processing_duration_ms=round(processing_duration_ms, 2),
                inference_duration_ms=round(inference_duration_ms, 2),
            )

            # Create and return response
            return TranscriptionResponse(text=text)

        except (AudioValidationError, AudioProcessingError, ModelError, ModelNotLoadedError):
            # Re-raise expected exceptions
            raise
        except Exception as e:
            logger.error(
                "transcription_failed",
                error_type=type(e).__name__,
                error_message=str(e),
                request_id=request_id,
                exc_info=True,
            )
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
            logger.debug("cleaned_up_uploaded_file", path=str(uploaded_file_path))

            # Cleanup processed file if it's different from uploaded
            if (
                needs_cleanup
                and processed_file_path
                and processed_file_path != uploaded_file_path
            ):
                await self.audio_processor.cleanup_temp_file(processed_file_path)
                logger.debug("cleaned_up_processed_file", path=str(processed_file_path))

        except Exception as e:
            logger.warning("cleanup_error", error=str(e))

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