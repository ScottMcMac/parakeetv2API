# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

parakeetv2API is a FastAPI-based server that provides an OpenAI-compatible audio transcription API using NVIDIA's parakeet-tdt-0.6b-v2 ASR model. The project is in initial development with no code implemented yet.

## Development Commands

### Environment Setup
```bash
conda activate nemo
```

### Running Tests (once implemented)
```bash
# Test runner not yet configured - likely pytest
# Test files located in ./tests/audio_files/ and ./tests/non_audio_files/
```

### Development Server (once implemented)
```bash
# FastAPI development server command to be determined
# Default port: 8011
```

## Architecture Overview

### API Endpoints
1. **POST /v1/audio/transcriptions** - Main transcription endpoint
   - Accepts audio files: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm
   - Returns OpenAI-compatible JSON response
   - Must handle audio format conversion via FFmpeg

2. **GET /v1/models** - List available models
3. **GET /v1/models/{model_id}** - Get specific model info

### Model Management
- The parakeet-tdt-0.6b-v2 model must be loaded on startup and kept in memory
- All model aliases (gpt-4o-transcribe, gpt-4o-mini-transcribe, parakeet-tdt-0.6b-v2, whisper-1) use the same backend
- Model expects 16kHz mono audio in WAV or FLAC format

### Audio Processing Pipeline
1. Validate file extension and audio content using FFmpeg
2. Convert non-compatible formats to 16kHz mono WAV
3. Pass to model for transcription
4. Return standardized JSON response with fixed token usage (1 input, 1 output)

### Error Handling Requirements
- Validate audio file types before processing
- Return informative errors for unsupported options (non-English language, non-JSON format, timestamp granularities)
- Handle non-audio files gracefully without crashing

### Testing Requirements
- All test audio files should transcribe to "The quick brown fox jumped over the lazy dog"
- Tests must be case and punctuation insensitive
- Test various audio formats, sample rates, and configurations
- Include error case testing with non-audio files

## Key Dependencies
- Python 3.11 with conda environment "nemo"
- nemo_toolkit["asr"] for the ASR model
- cuda-python>=12.3 for GPU acceleration
- FastAPI for the web framework
- FFmpeg for audio conversion
- NVIDIA GPU required for inference

## Development Principles
- Follow SOLID, KISS, and YAGNI principles
- Match OpenAI API specification for compatibility
- Ensure all features have comprehensive tests
- Keep model loaded for fast response times