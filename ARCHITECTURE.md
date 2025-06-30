# parakeetv2API Architecture & Implementation Plan

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              parakeetv2API Server                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                          FastAPI Application                          │  │
│  │                                                                       │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────┐   │  │
│  │  │   API Routes     │  │   Middleware    │  │  Exception        │   │  │
│  │  │                  │  │                 │  │  Handlers         │   │  │
│  │  │ • /v1/audio/    │  │ • CORS         │  │                   │   │  │
│  │  │   transcriptions│  │ • Rate Limiting │  │ • ValidationError │   │  │
│  │  │ • /v1/models    │  │ • Request ID    │  │ • HTTPException   │   │  │
│  │  │ • /v1/models/   │  │ • Logging       │  │ • Custom Errors   │   │  │
│  │  │   {model_id}    │  │                 │  │                   │   │  │
│  │  └────────┬─────────┘  └─────────────────┘  └───────────────────┘   │  │
│  │           │                                                           │  │
│  └───────────┼───────────────────────────────────────────────────────┘  │
│              │                                                            │
│  ┌───────────▼───────────────────────────────────────────────────────┐  │
│  │                        Service Layer                               │  │
│  │                                                                    │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │  │
│  │  │ Transcription   │  │ Model Service   │  │ Audio Service   │  │  │
│  │  │ Service         │  │                 │  │                 │  │  │
│  │  │                 │  │ • Model Info    │  │ • Validation    │  │  │
│  │  │ • Process Audio │  │ • Model List    │  │ • Conversion    │  │  │
│  │  │ • Orchestrate   │  │ • Model Aliases │  │ • FFmpeg Ops    │  │  │
│  │  └────────┬────────┘  └─────────────────┘  └────────┬────────┘  │  │
│  │           │                                           │           │  │
│  └───────────┼───────────────────────────────────────────┼──────────┘  │
│              │                                           │              │
│  ┌───────────▼───────────────────────────────────────────▼──────────┐  │
│  │                      Core Components                              │  │
│  │                                                                   │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │  │
│  │  │ Model Manager   │  │ Audio Processor │  │ Config Manager  │ │  │
│  │  │                 │  │                 │  │                 │ │  │
│  │  │ • Load Model    │  │ • FFmpeg Wrapper│  │ • Settings      │ │  │
│  │  │ • Inference     │  │ • Format Check  │  │ • GPU Config    │ │  │
│  │  │ • Keep in Mem   │  │ • Convert Audio │  │ • Port/Host     │ │  │
│  │  └────────┬────────┘  └─────────────────┘  └─────────────────┘ │  │
│  │           │                                                      │  │
│  └───────────┼──────────────────────────────────────────────────┘  │
│              │                                                       │
│  ┌───────────▼──────────────────────────────────────────────────┐  │
│  │                    External Dependencies                       │  │
│  │                                                                │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │  │
│  │  │ NVIDIA NeMo     │  │ FFmpeg Binary   │  │ CUDA Runtime │  │  │
│  │  │ Toolkit         │  │                 │  │              │  │  │
│  │  │                 │  │ • Audio Convert │  │ • GPU Access │  │  │
│  │  │ • ASR Model     │  │ • Format Check  │  │              │  │  │
│  │  └─────────────────┘  └─────────────────┘  └──────────────┘  │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Layered Architecture
- **API Layer**: FastAPI routes and middleware
- **Service Layer**: Business logic orchestration
- **Core Layer**: Model management and audio processing
- **External Layer**: Third-party dependencies

### 2. Separation of Concerns
- **Model Manager**: Handles model lifecycle (loading, inference, memory management)
- **Audio Processor**: FFmpeg operations and format validation
- **Transcription Service**: Orchestrates the transcription workflow
- **Config Manager**: Centralized configuration handling

### 3. Request Flow
1. Client sends multipart/form-data request to `/v1/audio/transcriptions`
2. FastAPI validates request structure
3. Audio Service validates file type and content
4. Audio Processor converts to required format (16kHz mono WAV)
5. Model Manager performs inference
6. Response formatted per OpenAI spec

## Phased Implementation Plan

### Phase 1: Project Foundation (Core Infrastructure)

#### 1.1 Project Structure Setup
```
parakeetv2API/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── transcription.py
│   │   │   └── models.py
│   │   └── dependencies.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── transcription.py
│   │   ├── model.py
│   │   └── audio.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model_manager.py
│   │   ├── audio_processor.py
│   │   └── exceptions.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py      # Pydantic models
│   │   └── responses.py
│   └── utils/
│       ├── __init__.py
│       └── validators.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── pyproject.toml
```

#### 1.2 Dependencies Installation
- FastAPI and Uvicorn
- Pydantic for data validation
- python-multipart for file uploads
- Additional testing/development tools

#### 1.3 Configuration System
- Environment variables support
- Settings class with validation
- GPU selection, host, and port configuration

### Phase 2: Core Components Implementation

#### 2.1 Model Manager
- Model loading on startup
- Singleton pattern for model instance
- GPU device selection
- Inference method with error handling

#### 2.2 Audio Processor
- FFmpeg wrapper class
- Audio format validation
- Conversion to 16kHz mono WAV
- Temporary file management

#### 2.3 Custom Exceptions
- ValidationError for input validation
- ModelError for inference issues
- AudioProcessingError for FFmpeg failures

### Phase 3: API Layer Development

#### 3.1 Pydantic Models
- Request models for transcription endpoint
- Response models matching OpenAI spec
- Model info schemas

#### 3.2 API Routes
- POST /v1/audio/transcriptions
- GET /v1/models
- GET /v1/models/{model_id}

#### 3.3 Request Validation
- File extension validation
- Audio content validation
- Parameter validation with informative errors

### Phase 4: Service Layer Integration

#### 4.1 Transcription Service
- Orchestrate audio processing and inference
- Handle temporary file cleanup
- Format response per OpenAI spec

#### 4.2 Model Service
- Manage model aliases
- Provide model information
- Handle model queries

#### 4.3 Audio Service
- Coordinate validation and conversion
- Error handling and recovery

### Phase 5: Testing Implementation

#### 5.1 Unit Tests
- Test each component in isolation
- Mock external dependencies
- Cover edge cases

#### 5.2 Integration Tests
- Test service interactions
- Audio processing pipeline
- Model inference flow

#### 5.3 End-to-End Tests
- Full API testing with test audio files
- Error case testing
- Performance benchmarks

### Phase 6: Production Readiness

#### 6.1 Logging and Monitoring
- Structured logging
- Request tracing
- Performance metrics

#### 6.2 Error Handling
- Comprehensive exception handling
- Graceful degradation
- User-friendly error messages

#### 6.3 Documentation
- API documentation (OpenAPI/Swagger)
- Deployment guide
- Configuration reference

### Phase 7: Optimization & Enhancement

#### 7.1 Performance Optimization
- Request batching consideration
- Memory usage optimization
- Response time improvements

#### 7.2 Security Hardening
- Input sanitization
- Rate limiting
- File size limits

#### 7.3 Deployment Preparation
- Health check endpoints
- Graceful shutdown handling

## Implementation Priority Order

1. **Core Infrastructure** (Phase 1)
2. **Model Manager & Config** (Phase 2.1, 2.3, 1.3)
3. **API Routes & Models** (Phase 3)
4. **Audio Processing** (Phase 2.2)
5. **Service Integration** (Phase 4)
6. **Testing** (Phase 5)
7. **Production Features** (Phase 6)
8. **Optimization** (Phase 7)

## Key Technical Considerations

1. **Model Loading**: Load once on startup, keep in memory
2. **File Handling**: Use context managers for automatic cleanup
3. **Error Messages**: Provide clear, actionable error messages
4. **Testing**: Test with all provided audio files
5. **Performance**: Monitor memory usage with loaded model
6. **Compatibility**: Strictly follow OpenAI API specification

## Success Criteria

- All test audio files transcribe correctly
- API matches OpenAI specification
- Graceful error handling for invalid inputs
- Fast response times (<2s for typical audio)
- Comprehensive test coverage (>90%)
- Clean, maintainable code following SOLID principles