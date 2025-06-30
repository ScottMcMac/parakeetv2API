"""Transcription API routes."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from src.api.dependencies import get_request_id, verify_api_key
from src.core.error_handler import error_handler, with_error_handling
from src.core.exceptions import (
    AudioProcessingError,
    AudioValidationError,
    ModelError,
    ModelNotLoadedError,
    ParakeetAPIException,
    UnsupportedParameterError,
)
from src.core.logging import get_logger
from src.models import TranscriptionRequest, TranscriptionResponse
from src.services import transcription_service

logger = get_logger(__name__)

router = APIRouter(prefix="/audio", tags=["Audio"])


@router.post(
    "/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe audio",
    description="Transcribes audio into the input language.",
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "example": {
                        "text": "The quick brown fox jumped over the lazy dog.",
                        "usage": {
                            "type": "tokens",
                            "input_tokens": 1,
                            "input_token_details": {
                                "text_tokens": 0,
                                "audio_tokens": 1
                            },
                            "output_tokens": 1,
                            "total_tokens": 2
                        }
                    }
                }
            }
        },
        400: {
            "description": "Bad request - invalid parameters or file",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "message": "Unsupported file format: txt. Supported formats: flac, m4a, mp3, mp4, mpeg, mpga, ogg, wav, webm",
                            "type": "invalid_request_error",
                            "param": "file",
                            "code": "invalid_file_format"
                        }
                    }
                }
            }
        },
        413: {
            "description": "File too large",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "message": "File too large: 30.5MB. Maximum allowed: 25.0MB",
                            "type": "invalid_request_error",
                            "param": "file",
                            "code": "file_too_large"
                        }
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "message": "Transcription failed",
                            "type": "server_error",
                            "code": "transcription_failed"
                        }
                    }
                }
            }
        }
    }
)
async def transcribe_audio(
    file: UploadFile = File(..., description="The audio file to transcribe"),
    model: Optional[str] = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    temperature: Optional[float] = Form(default=None),
    timestamp_granularities: Optional[str] = Form(default=None),  # Will be parsed as list
    chunking_strategy: Optional[str] = Form(default=None),
    include: Optional[str] = Form(default=None),  # Will be parsed as list
    stream: Optional[bool] = Form(default=False),
    api_key: Optional[str] = Depends(verify_api_key),
    request_id: Optional[str] = Depends(get_request_id),
) -> TranscriptionResponse:
    """
    Transcribe audio file to text.
    
    This endpoint is compatible with the OpenAI API specification.
    All model parameters use the same parakeet-tdt-0.6b-v2 backend.
    """
    logger.info(
        "transcription_request",
        model=model,
        filename=file.filename,
        request_id=request_id,
    )
    
    # Parse list parameters if provided as strings
    timestamp_granularities_list = None
    if timestamp_granularities:
        try:
            import json
            timestamp_granularities_list = json.loads(timestamp_granularities)
        except:
            timestamp_granularities_list = [timestamp_granularities]
    
    include_list = None
    if include:
        try:
            import json
            include_list = json.loads(include)
        except:
            include_list = [include]
    
    # Create request object for validation
    try:
        request = TranscriptionRequest(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities_list,
            chunking_strategy=chunking_strategy,
            include=include_list,
            stream=stream,
        )
    except UnsupportedParameterError as e:
        logger.warning("unsupported_parameter", parameter=e.details.get("parameter"), message=e.message)
        status_code = error_handler.get_status_code(e)
        response_data = error_handler.format_error_response(e, request_id=request_id)
        raise HTTPException(status_code=status_code, detail=response_data)
    except Exception as e:
        logger.error("request_validation_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": str(e),
                    "type": "invalid_request_error",
                    "code": "validation_error"
                }
            }
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "No filename provided",
                    "type": "invalid_request_error",
                    "param": "file",
                    "code": "missing_filename"
                }
            }
        )
    
    # Read file content
    content = await file.read()
    
    # Use transcription service to handle the entire workflow
    try:
        return await transcription_service.transcribe_audio(
            file_content=content,
            filename=file.filename,
            request=request,
            request_id=request_id,
        )
    except ParakeetAPIException as e:
        # Use centralized error handler for our custom exceptions
        status_code = error_handler.get_status_code(e)
        response_data = error_handler.format_error_response(e, request_id=request_id)
        raise HTTPException(status_code=status_code, detail=response_data)
    except Exception as e:
        # Log unexpected errors
        logger.error(
            "unexpected_transcription_error",
            error_type=type(e).__name__,
            error_message=str(e),
            request_id=request_id,
            exc_info=True,
        )
        # Use generic error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_handler.format_error_response(e, request_id=request_id)
        )