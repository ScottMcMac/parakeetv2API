"""Model manager for handling ASR model lifecycle."""

import logging
import threading
from pathlib import Path
from typing import List, Optional, Union

import nemo.collections.asr as nemo_asr
import torch

from src.config import settings
from src.core.exceptions import ModelError, ModelNotLoadedError

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton class for managing the ASR model."""
    
    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "ModelManager":
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model manager."""
        if hasattr(self, "_initialized"):
            return
        
        self._initialized = True
        self._model: Optional[nemo_asr.models.ASRModel] = None
        self._model_name = settings.model_name
        self._device: Optional[torch.device] = None
        self._loading_lock = threading.Lock()
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def device(self) -> torch.device:
        """Get the device for model inference."""
        if self._device is None:
            if torch.cuda.is_available():
                if settings.gpu_device is not None:
                    self._device = torch.device(f"cuda:{settings.gpu_device}")
                else:
                    self._device = torch.device("cuda")
                logger.info(f"Using device: {self._device}")
            else:
                self._device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU. Performance will be degraded.")
        return self._device
    
    def load_model(self) -> None:
        """Load the ASR model."""
        with self._loading_lock:
            if self._is_loaded:
                logger.info("Model already loaded")
                return
            
            try:
                logger.info(f"Loading model: {self._model_name}")
                
                # Set cache directory if specified
                cache_dir = None
                if settings.model_cache_dir:
                    cache_dir = Path(settings.model_cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Load the model
                self._model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=self._model_name,
                    map_location=self.device,
                    strict=False,
                )
                
                # Move model to device
                self._model = self._model.to(self.device)
                self._model.eval()
                
                # Warm up the model with test files to ensure it's ready
                self._warmup_model()
                
                self._is_loaded = True
                logger.info("Model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise ModelError(f"Failed to load model '{self._model_name}': {str(e)}")
    
    def _warmup_model(self) -> None:
        """Warm up the model with test transcriptions."""
        try:
            logger.info("Warming up model...")
            
            # Test file paths from README
            test_files = [
                "./tests/audio_files/test_wav_16000Hz_mono.wav",
                "./tests/audio_files/test_wav_16000Hz_mono_32bit.wav",
                "./tests/audio_files/test_flac_8000Hz_mono.flac",
            ]
            
            # Only use files that exist
            existing_files = [f for f in test_files if Path(f).exists()]
            
            if existing_files:
                logger.info(f"Running warmup with {len(existing_files)} test files")
                _ = self._model.transcribe(existing_files)
                logger.info("Model warmup completed")
            else:
                logger.warning("No test files found for warmup, skipping")
                
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {str(e)}")
    
    def transcribe(
        self,
        audio_paths: Union[str, Path, List[Union[str, Path]]],
        batch_size: int = 1,
    ) -> List[str]:
        """
        Transcribe audio files.
        
        Args:
            audio_paths: Path(s) to audio file(s)
            batch_size: Batch size for inference
            
        Returns:
            List of transcriptions
            
        Raises:
            ModelNotLoadedError: If model is not loaded
            ModelError: If transcription fails
        """
        if not self._is_loaded or self._model is None:
            raise ModelNotLoadedError()
        
        # Ensure paths are in a list
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [audio_paths]
        
        # Convert to strings
        audio_paths = [str(p) for p in audio_paths]
        
        try:
            with torch.no_grad():
                transcriptions = self._model.transcribe(
                    audio_paths,
                    batch_size=batch_size,
                    return_hypotheses=False,
                )
            
            # Ensure we always return a list
            if isinstance(transcriptions, str):
                transcriptions = [transcriptions]
            
            return transcriptions
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise ModelError(f"Transcription failed: {str(e)}")
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        with self._loading_lock:
            if self._model is not None:
                logger.info("Unloading model")
                del self._model
                self._model = None
                self._is_loaded = False
                
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("Model unloaded")


# Global model manager instance
model_manager = ModelManager()