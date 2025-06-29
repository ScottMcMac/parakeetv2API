"""Audio processor for handling audio file conversions and validation."""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import aiofiles

from src.config import settings
from src.core.exceptions import AudioProcessingError, AudioValidationError

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {
    "flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"
}

# Target format for model
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = "wav"


class AudioProcessor:
    """Handle audio file validation and conversion using FFmpeg."""
    
    def __init__(self):
        """Initialize the audio processor."""
        self.temp_dir = Path(settings.temp_dir) if settings.temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    async def validate_audio_file(self, file_path: Path) -> Dict[str, any]:
        """
        Validate audio file using FFprobe.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio metadata dictionary
            
        Raises:
            AudioValidationError: If file is not valid audio
        """
        if not file_path.exists():
            raise AudioValidationError(f"File not found: {file_path}")
        
        # Check file extension
        extension = file_path.suffix.lower().lstrip(".")
        if extension not in SUPPORTED_FORMATS:
            raise AudioValidationError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )
        
        # Use ffprobe to validate and get metadata
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_streams",
                "-select_streams", "a:0",
                "-of", "json",
                str(file_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise AudioValidationError(f"Invalid audio file: {error_msg}")
            
            # Parse metadata
            metadata = json.loads(stdout.decode())
            streams = metadata.get("streams", [])
            
            if not streams:
                raise AudioValidationError("No audio stream found in file")
            
            audio_stream = streams[0]
            
            # Extract relevant metadata
            return {
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": int(audio_stream.get("channels", 0)),
                "codec_name": audio_stream.get("codec_name", "unknown"),
                "duration": float(audio_stream.get("duration", 0)),
                "bit_rate": int(audio_stream.get("bit_rate", 0)),
            }
            
        except asyncio.CancelledError:
            raise
        except AudioValidationError:
            raise
        except Exception as e:
            logger.error(f"FFprobe error: {str(e)}")
            raise AudioValidationError(f"Failed to validate audio file: {str(e)}")
    
    async def needs_conversion(self, file_path: Path, metadata: Dict[str, any]) -> bool:
        """
        Check if audio file needs conversion for the model.
        
        Args:
            file_path: Path to the audio file
            metadata: Audio metadata from validation
            
        Returns:
            True if conversion is needed
        """
        # Check if it's already in the correct format
        if (file_path.suffix.lower() in [".wav", ".flac"] and
            metadata["sample_rate"] == TARGET_SAMPLE_RATE and
            metadata["channels"] == TARGET_CHANNELS):
            return False
        
        return True
    
    async def convert_audio(
        self,
        input_path: Path,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Convert audio to target format using FFmpeg.
        
        Args:
            input_path: Path to input audio file
            output_path: Optional output path
            
        Returns:
            Path to converted audio file
            
        Raises:
            AudioProcessingError: If conversion fails
        """
        if output_path is None:
            # Generate temporary output file
            output_path = self.temp_dir / f"{input_path.stem}_converted.{TARGET_FORMAT}"
        
        try:
            cmd = [
                "ffmpeg",
                "-i", str(input_path),
                "-ar", str(TARGET_SAMPLE_RATE),  # Sample rate
                "-ac", str(TARGET_CHANNELS),      # Channels (mono)
                "-c:a", "pcm_s16le",              # 16-bit PCM
                "-y",                              # Overwrite output
                str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise AudioProcessingError(f"Audio conversion failed: {error_msg}")
            
            logger.info(f"Converted audio: {input_path} -> {output_path}")
            return output_path
            
        except asyncio.CancelledError:
            raise
        except AudioProcessingError:
            raise
        except Exception as e:
            logger.error(f"FFmpeg error: {str(e)}")
            raise AudioProcessingError(f"Failed to convert audio: {str(e)}")
    
    async def process_audio_file(self, file_path: Path) -> Tuple[Path, bool]:
        """
        Process audio file for transcription.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (processed_file_path, needs_cleanup)
            
        Raises:
            AudioValidationError: If validation fails
            AudioProcessingError: If processing fails
        """
        # Validate the audio file
        metadata = await self.validate_audio_file(file_path)
        
        # Check if conversion is needed
        if await self.needs_conversion(file_path, metadata):
            # Convert the audio
            converted_path = await self.convert_audio(file_path)
            return converted_path, True
        else:
            # No conversion needed
            return file_path, False
    
    async def cleanup_temp_file(self, file_path: Path) -> None:
        """
        Clean up temporary file.
        
        Args:
            file_path: Path to temporary file
        """
        try:
            if file_path.exists() and file_path.parent == self.temp_dir:
                file_path.unlink()
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")
    
    async def save_uploaded_file(self, content: bytes, filename: str) -> Path:
        """
        Save uploaded file content to temporary location.
        
        Args:
            content: File content
            filename: Original filename
            
        Returns:
            Path to saved file
        """
        # Generate safe filename
        safe_filename = f"{os.urandom(16).hex()}_{Path(filename).name}"
        file_path = self.temp_dir / safe_filename
        
        try:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            raise AudioProcessingError(f"Failed to save uploaded file: {str(e)}")


# Global audio processor instance
audio_processor = AudioProcessor()