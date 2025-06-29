"""Shared dependencies for API routes."""

from typing import Optional

from fastapi import Header, HTTPException, status


async def verify_api_key(
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """
    Verify API key from Authorization header.
    
    Note: This is for OpenAI API compatibility only.
    The API key is not actually validated.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        The API key if provided, None otherwise
    """
    if not authorization:
        return None
    
    # Check for Bearer token format
    if authorization.startswith("Bearer "):
        return authorization[7:]
    
    # For compatibility, we don't actually validate the key
    return authorization


async def get_request_id(
    x_request_id: Optional[str] = Header(None),
) -> Optional[str]:
    """
    Get request ID from header.
    
    Args:
        x_request_id: Request ID from header
        
    Returns:
        Request ID if provided
    """
    return x_request_id