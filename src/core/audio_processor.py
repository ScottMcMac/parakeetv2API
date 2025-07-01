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
import soundfile as sf

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
    
    async def get_audio_metadata(self, file_path: Path) -> Optional[Dict[str, any]]:
        """
        Get audio metadata using soundfile (fast, no subprocesses).
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio metadata dictionary or None if extraction fails
        """
        import time
        start_time = time.time()
        
        if not file_path.exists():
            return None
        
        try:
            # Use soundfile for fast metadata extraction (no subprocess)
            info = sf.info(str(file_path))
            
            # Calculate approximate bit rate (not always accurate but sufficient)
            # For PCM: bit_rate = sample_rate * channels * bits_per_sample
            # Assume 16-bit for most audio files
            estimated_bit_rate = info.samplerate * info.channels * 16
            
            result = {
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "codec_name": info.subtype.lower() if info.subtype else "unknown",
                "duration": info.duration,
                "bit_rate": estimated_bit_rate,
            }
            
            total_duration = (time.time() - start_time) * 1000
            logger.info(f"Metadata extraction for {file_path.name}: {total_duration:.2f}ms (soundfile)")
            logger.info(f"Audio format: {result['sample_rate']}Hz, {result['channels']} channels, {result['codec_name']}")
            
            return result
            
        except Exception as e:
            total_duration = (time.time() - start_time) * 1000
            logger.info(f"Failed to get metadata for {file_path.name} after {total_duration:.2f}ms: {str(e)}")
            
            # Fallback to FFprobe for unsupported formats
            logger.info(f"Falling back to FFprobe for {file_path.name}")
            return await self._get_metadata_ffprobe(file_path)
    
    async def _get_metadata_ffprobe(self, file_path: Path) -> Optional[Dict[str, any]]:
        """
        Fallback metadata extraction using FFprobe for unsupported formats.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio metadata dictionary or None if extraction fails
        """
        import time
        start_time = time.time()
        
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
                logger.info(f"FFprobe fallback failed for {file_path.name}")
                return None
            
            # Parse metadata
            metadata = json.loads(stdout.decode())
            streams = metadata.get("streams", [])
            
            if not streams:
                return None
            
            audio_stream = streams[0]
            
            # Extract relevant metadata
            result = {
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": int(audio_stream.get("channels", 0)),
                "codec_name": audio_stream.get("codec_name", "unknown"),
                "duration": float(audio_stream.get("duration", 0)),
                "bit_rate": int(audio_stream.get("bit_rate", 0)),
            }
            
            total_duration = (time.time() - start_time) * 1000
            logger.info(f"FFprobe fallback for {file_path.name}: {total_duration:.2f}ms")
            
            return result
            
        except Exception as e:
            total_duration = (time.time() - start_time) * 1000
            logger.info(f"FFprobe fallback failed for {file_path.name} after {total_duration:.2f}ms: {str(e)}")
            return None
    
    async def needs_conversion(self, file_path: Path, metadata: Optional[Dict[str, any]] = None) -> bool:
        """
        Check if audio file needs conversion for the model.
        
        Args:
            file_path: Path to the audio file
            metadata: Optional audio metadata 
            
        Returns:
            True if conversion is needed
        """
        # If no metadata, assume conversion is needed
        if metadata is None:
            return True
            
        # Check if it's already in the correct format
        if (file_path.suffix.lower() in [".wav", ".flac"] and
            metadata.get("sample_rate") == TARGET_SAMPLE_RATE and
            metadata.get("channels") == TARGET_CHANNELS):
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
            AudioProcessingError: If processing fails
        """
        import time
        start_time = time.time()
        
        # Get metadata to check if conversion is needed
        logger.info(f"Starting audio processing for {file_path.name}")
        metadata = await self.get_audio_metadata(file_path)
        
        # Check if conversion is needed (handles None metadata gracefully)
        needs_conv = await self.needs_conversion(file_path, metadata)
        conversion_check_time = (time.time() - start_time) * 1000
        
        if needs_conv:
            logger.info(f"Conversion needed for {file_path.name} (check took {conversion_check_time:.2f}ms)")
            try:
                # Convert the audio
                convert_start = time.time()
                converted_path = await self.convert_audio(file_path)
                convert_duration = (time.time() - convert_start) * 1000
                total_duration = (time.time() - start_time) * 1000
                logger.info(f"Audio conversion completed in {convert_duration:.2f}ms (total: {total_duration:.2f}ms)")
                return converted_path, True
            except AudioProcessingError:
                # Re-raise conversion errors
                raise
            except Exception as e:
                # Wrap unexpected errors
                raise AudioProcessingError(f"Unexpected error during conversion: {str(e)}")
        else:
            total_duration = (time.time() - start_time) * 1000
            logger.info(f"No conversion needed for {file_path.name} (total processing: {total_duration:.2f}ms)")
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
        import time
        start_time = time.time()
        
        # Generate safe filename
        safe_filename = f"{os.urandom(16).hex()}_{Path(filename).name}"
        file_path = self.temp_dir / safe_filename
        
        try:
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(content)
            
            duration = (time.time() - start_time) * 1000
            logger.info(f"Saved uploaded file {filename} ({len(content)} bytes) in {duration:.2f}ms")
            
            return file_path
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            logger.error(f"Failed to save uploaded file after {duration:.2f}ms: {str(e)}")
            raise AudioProcessingError(f"Failed to save uploaded file: {str(e)}")


# Global audio processor instance
audio_processor = AudioProcessor()