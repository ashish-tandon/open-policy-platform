"""
Performance Middleware
Provides caching, rate limiting, and response optimization for the API
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any
import time
import hashlib
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class PerformanceMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, cache_ttl: int = 300, rate_limit_per_minute: int = 100):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.rate_limit_per_minute = rate_limit_per_minute
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.rate_limit_store: Dict[str, list] = defaultdict(list)
        
    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()
        
        # Check rate limiting
        client_ip = request.client.host if request.client else "unknown"
        if not await self._check_rate_limit(client_ip):
            return Response(
                content=json.dumps({"error": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json"
            )
        
        # Check cache for GET requests
        if request.method == "GET":
            cache_key = await self._generate_cache_key(request)
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Cache hit for {request.url}")
                return cached_response
        
        # Process request
        response = await call_next(request)
        
        # Cache successful GET responses
        if request.method == "GET" and response.status_code == 200:
            cache_key = await self._generate_cache_key(request)
            # Buffer the response body so we can both cache and return it
            response, body = await self._buffer_response(response)
            await self._cache_response(cache_key, body, response)
        
        # Add performance headers
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Cache"] = "MISS" if request.method == "GET" else "N/A"
        
        # Log performance metrics
        logger.info(f"Request processed in {process_time:.3f}s: {request.method} {request.url}")
        
        return response
    
    async def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key based on request URL and query parameters"""
        url = str(request.url)
        query_params = dict(request.query_params)
        
        # Sort query parameters for consistent cache keys
        sorted_params = sorted(query_params.items())
        cache_string = f"{url}:{json.dumps(sorted_params)}"
        
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def _get_cached_response(self, cache_key: str) -> Response | None:
        """Get cached response if available and not expired"""
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if datetime.now() < cached_data["expires"]:
                response = Response(
                    content=cached_data["content"],
                    status_code=cached_data["status_code"],
                    headers=cached_data["headers"],
                    media_type=cached_data["media_type"]
                )
                response.headers["X-Cache"] = "HIT"
                return response
            else:
                # Remove expired cache entry
                del self.cache[cache_key]
        
        return None
    
    async def _cache_response(self, cache_key: str, content: bytes, response: Response):
        """Cache response for future use"""
        self.cache[cache_key] = {
            "content": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "media_type": response.media_type,
            "expires": datetime.now() + timedelta(seconds=self.cache_ttl)
        }
        
        # Clean up expired cache entries
        await self._cleanup_cache()
    
    async def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old entries
        self.rate_limit_store[client_ip] = [
            timestamp for timestamp in self.rate_limit_store[client_ip]
            if timestamp > minute_ago
        ]
        
        # Check if limit exceeded
        if len(self.rate_limit_store[client_ip]) >= self.rate_limit_per_minute:
            return False
        
        # Add current request
        self.rate_limit_store[client_ip].append(now)
        return True
    
    async def _cleanup_cache(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, data in self.cache.items()
            if now > data["expires"]
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _buffer_response(self, response: Response) -> tuple[Response, bytes]:
        """Ensure response body is fully buffered and return a new Response with bytes body.
        Handles StreamingResponse which lacks a .body() coroutine.
        """
        body_bytes: bytes = b""
        try:
            # StreamingResponse exposes an async iterator
            body_iterator = getattr(response, "body_iterator", None)
            if body_iterator is not None:
                async for chunk in body_iterator:
                    if chunk:
                        body_bytes += chunk
            else:
                # Standard Response supports awaiting .body()
                body_bytes = await response.body()
        except Exception:
            # Fallback to best-effort body attribute
            body_bytes = getattr(response, "body", b"") or b""

        new_response = Response(
            content=body_bytes,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type
        )
        return new_response, body_bytes
