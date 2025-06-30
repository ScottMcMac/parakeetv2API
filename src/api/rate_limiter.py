"""Rate limiting for graceful degradation under load."""

import time
from collections import defaultdict
from typing import Dict, Optional

from fastapi import HTTPException, Request, status

from src.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ):
        """Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests: Dict[str, list] = defaultdict(list)
    
    def _clean_old_requests(self, client_id: str, current_time: float) -> None:
        """Remove requests older than 1 minute."""
        cutoff_time = current_time - 60
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > cutoff_time
        ]
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client.
        
        Args:
            client_id: Client identifier (IP or API key)
            
        Returns:
            True if request is allowed
        """
        current_time = time.time()
        self._clean_old_requests(client_id, current_time)
        
        recent_requests = self.requests[client_id]
        
        # Check burst limit
        if len(recent_requests) >= self.burst_size:
            # Check if all burst requests were in last second
            one_second_ago = current_time - 1
            burst_requests = [t for t in recent_requests if t > one_second_ago]
            if len(burst_requests) >= self.burst_size:
                return False
        
        # Check rate limit
        if len(recent_requests) >= self.requests_per_minute:
            return False
        
        # Allow request
        self.requests[client_id].append(current_time)
        return True
    
    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until next request is allowed.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Seconds to wait
        """
        current_time = time.time()
        self._clean_old_requests(client_id, current_time)
        
        if not self.requests[client_id]:
            return 0
        
        # Calculate when the oldest request will expire
        oldest_request = min(self.requests[client_id])
        retry_after = max(0, int(60 - (current_time - oldest_request)))
        
        return retry_after


class RateLimitMiddleware:
    """Middleware for rate limiting."""
    
    def __init__(self, rate_limiter: RateLimiter):
        """Initialize middleware.
        
        Args:
            rate_limiter: Rate limiter instance
        """
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/", "/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier (prefer API key, fallback to IP)
        client_id = request.headers.get("Authorization", "")
        if not client_id and request.client:
            client_id = request.client.host
        
        if not client_id:
            client_id = "anonymous"
        
        # Check rate limit
        if not self.rate_limiter.is_allowed(client_id):
            retry_after = self.rate_limiter.get_retry_after(client_id)
            
            logger.warning(
                "rate_limit_exceeded",
                client_id=client_id,
                retry_after=retry_after,
                path=request.url.path,
            )
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "message": "Rate limit exceeded. Please slow down.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                        "retry_after": retry_after,
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )
        
        return await call_next(request)


# Global rate limiter instances
transcription_rate_limiter = RateLimiter(
    requests_per_minute=settings.max_concurrent_requests * 10,
    burst_size=settings.max_concurrent_requests,
)

api_rate_limiter = RateLimiter(
    requests_per_minute=600,  # 10 requests per second average
    burst_size=20,
)