"""
Advanced Caching and CDN System - 40by6
Enterprise-grade caching with multi-layer strategy and CDN integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import pickle
import zlib
import lz4.frame
import msgpack
from collections import OrderedDict, defaultdict
import redis
from redis.sentinel import Sentinel
import aioredis
import memcache
import aiomcache
import diskcache
from cachetools import TTLCache, LRUCache, LFUCache
import cloudflare
import boto3
from azure.storage.blob import BlobServiceClient
from fastly import Fastly
import aiohttp
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import numpy as np
from sklearn.linear_model import LinearRegression
from prometheus_client import Counter, Histogram, Gauge
import psutil
import mmap
import struct
from bloom_filter2 import BloomFilter
from pybloom_live import ScalableBloomFilter
import xxhash
import cityhash
from consistent_hash import ConsistentHash
from circuit_breaker import CircuitBreaker
import gevent
from gevent.pool import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ray
import dask.distributed
from prefect import task, Flow
import networkx as nx
from typing_extensions import Protocol

logger = logging.getLogger(__name__)

# Metrics
cache_hits = Counter('cache_hits_total', 'Total number of cache hits', ['layer', 'key_pattern'])
cache_misses = Counter('cache_misses_total', 'Total number of cache misses', ['layer', 'key_pattern'])
cache_latency = Histogram('cache_operation_duration_seconds', 'Cache operation latency', ['operation', 'layer'])
cache_size = Gauge('cache_size_bytes', 'Current cache size in bytes', ['layer'])
cdn_bandwidth = Counter('cdn_bandwidth_bytes', 'CDN bandwidth usage', ['region', 'content_type'])
cdn_requests = Counter('cdn_requests_total', 'Total CDN requests', ['region', 'status'])

Base = declarative_base()


class CacheLayer(Enum):
    """Cache layer hierarchy"""
    L1_MEMORY = "l1_memory"  # In-process memory
    L2_SHARED_MEMORY = "l2_shared_memory"  # Shared memory between processes
    L3_REDIS = "l3_redis"  # Redis cache
    L4_MEMCACHED = "l4_memcached"  # Memcached
    L5_DISK = "l5_disk"  # Local disk cache
    L6_CDN = "l6_cdn"  # CDN edge cache
    L7_ORIGIN = "l7_origin"  # Origin storage


class CacheStrategy(Enum):
    """Caching strategies"""
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"
    REFRESH_AHEAD = "refresh_ahead"
    CACHE_ASIDE = "cache_aside"


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    MRU = "mru"  # Most Recently Used
    RR = "rr"  # Random Replacement
    GDSF = "gdsf"  # Greedy Dual Size Frequency
    ADAPTIVE = "adaptive"  # ML-based adaptive


class CDNProvider(Enum):
    """Supported CDN providers"""
    CLOUDFLARE = "cloudflare"
    FASTLY = "fastly"
    AKAMAI = "akamai"
    CLOUDFRONT = "cloudfront"
    AZURE_CDN = "azure_cdn"
    CUSTOM = "custom"


@dataclass
class CacheConfig:
    """Cache configuration"""
    layer: CacheLayer
    ttl: Optional[int] = 3600  # seconds
    max_size: Optional[int] = None  # bytes
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    compression: bool = True
    encryption: bool = False
    warmup_enabled: bool = True
    stats_enabled: bool = True
    sharding_enabled: bool = False
    shard_count: int = 16


@dataclass
class CDNConfig:
    """CDN configuration"""
    provider: CDNProvider
    zones: List[str] = field(default_factory=list)
    edge_locations: List[str] = field(default_factory=list)
    purge_strategy: str = "soft"  # soft or hard
    cache_control: Dict[str, str] = field(default_factory=dict)
    security_headers: Dict[str, str] = field(default_factory=dict)
    bandwidth_limit: Optional[int] = None  # bytes/second
    geo_restrictions: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Individual cache entry"""
    key: str
    value: Any
    size: int
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl
    
    @property
    def age(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def frequency_score(self) -> float:
        """Calculate frequency score for eviction"""
        age_hours = self.age / 3600
        if age_hours == 0:
            return float('inf')
        return self.access_count / age_hours


class CacheStats(Base):
    """Cache statistics table"""
    __tablename__ = 'cache_stats'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    layer = Column(String(50))
    hits = Column(Integer, default=0)
    misses = Column(Integer, default=0)
    evictions = Column(Integer, default=0)
    size_bytes = Column(Integer)
    avg_latency_ms = Column(Float)
    hit_rate = Column(Float)
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_cache_stats_timestamp', 'timestamp'),
        Index('idx_cache_stats_layer', 'layer'),
    )


class CacheInterface(Protocol):
    """Protocol for cache implementations"""
    
    async def get(self, key: str) -> Optional[Any]:
        ...
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        ...
    
    async def delete(self, key: str) -> bool:
        ...
    
    async def exists(self, key: str) -> bool:
        ...
    
    async def clear(self) -> bool:
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        ...


class L1MemoryCache(CacheInterface):
    """L1 in-process memory cache"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.max_size = config.max_size or 100 * 1024 * 1024  # 100MB default
        
        if config.eviction_policy == EvictionPolicy.LRU:
            self.cache = LRUCache(maxsize=1000)
        elif config.eviction_policy == EvictionPolicy.LFU:
            self.cache = LFUCache(maxsize=1000)
        else:
            self.cache = TTLCache(maxsize=1000, ttl=config.ttl)
        
        self.size_tracker = 0
        self.stats = defaultdict(int)
        self.bloom_filter = BloomFilter(max_elements=10000, error_rate=0.1)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with cache_latency.labels('get', 'l1_memory').time():
            if key in self.bloom_filter:
                try:
                    entry: CacheEntry = self.cache[key]
                    if not entry.is_expired:
                        entry.accessed_at = datetime.utcnow()
                        entry.access_count += 1
                        self.stats['hits'] += 1
                        cache_hits.labels('l1_memory', self._get_key_pattern(key)).inc()
                        return self._decompress(entry.value)
                except KeyError:
                    pass
            
            self.stats['misses'] += 1
            cache_misses.labels('l1_memory', self._get_key_pattern(key)).inc()
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        with cache_latency.labels('set', 'l1_memory').time():
            compressed_value = self._compress(value)
            size = len(compressed_value)
            
            # Check size limit
            if self.size_tracker + size > self.max_size:
                await self._evict_to_fit(size)
            
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                size=size,
                created_at=datetime.utcnow(),
                accessed_at=datetime.utcnow(),
                ttl=ttl or self.config.ttl
            )
            
            self.cache[key] = entry
            self.bloom_filter.add(key)
            self.size_tracker += size
            self.stats['sets'] += 1
            
            cache_size.labels('l1_memory').set(self.size_tracker)
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            entry = self.cache.pop(key)
            self.size_tracker -= entry.size
            self.stats['deletes'] += 1
            return True
        except KeyError:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if key not in self.bloom_filter:
            return False
        return key in self.cache and not self.cache[key].is_expired
    
    async def clear(self) -> bool:
        """Clear entire cache"""
        self.cache.clear()
        self.bloom_filter = BloomFilter(max_elements=10000, error_rate=0.1)
        self.size_tracker = 0
        self.stats['clears'] += 1
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'layer': 'l1_memory',
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'size_bytes': self.size_tracker,
            'max_size_bytes': self.max_size,
            'entry_count': len(self.cache),
            'evictions': self.stats.get('evictions', 0)
        }
    
    def _compress(self, value: Any) -> bytes:
        """Compress value if enabled"""
        data = pickle.dumps(value)
        if self.config.compression:
            return lz4.frame.compress(data)
        return data
    
    def _decompress(self, data: bytes) -> Any:
        """Decompress value if needed"""
        if self.config.compression:
            data = lz4.frame.decompress(data)
        return pickle.loads(data)
    
    def _get_key_pattern(self, key: str) -> str:
        """Extract pattern from key for metrics"""
        parts = key.split(':')
        if len(parts) >= 2:
            return f"{parts[0]}:*"
        return "other"
    
    async def _evict_to_fit(self, required_size: int):
        """Evict entries to fit new data"""
        if self.config.eviction_policy == EvictionPolicy.ADAPTIVE:
            await self._adaptive_eviction(required_size)
        else:
            # Simple eviction - remove oldest
            while self.size_tracker + required_size > self.max_size and len(self.cache) > 0:
                oldest_key = next(iter(self.cache))
                entry = self.cache.pop(oldest_key)
                self.size_tracker -= entry.size
                self.stats['evictions'] += 1
    
    async def _adaptive_eviction(self, required_size: int):
        """ML-based adaptive eviction"""
        # Calculate eviction scores
        candidates = []
        
        for key, entry in self.cache.items():
            # Features for ML model
            features = {
                'age': entry.age,
                'size': entry.size,
                'frequency': entry.frequency_score,
                'recency': (datetime.utcnow() - entry.accessed_at).total_seconds(),
                'cost': entry.size / (entry.access_count + 1)  # Size per access
            }
            
            # Simple scoring (in production, use trained model)
            score = (
                features['age'] * 0.2 +
                features['size'] * 0.3 +
                (1 / (features['frequency'] + 1)) * 0.3 +
                features['recency'] * 0.1 +
                features['cost'] * 0.1
            )
            
            candidates.append((score, key, entry))
        
        # Sort by score (higher score = more likely to evict)
        candidates.sort(reverse=True)
        
        # Evict until we have enough space
        freed_space = 0
        for score, key, entry in candidates:
            if freed_space >= required_size:
                break
            
            self.cache.pop(key)
            self.size_tracker -= entry.size
            freed_space += entry.size
            self.stats['evictions'] += 1


class L3RedisCache(CacheInterface):
    """L3 Redis distributed cache"""
    
    def __init__(self, config: CacheConfig, redis_url: str):
        self.config = config
        self.redis_url = redis_url
        self.pool = None
        self.consistent_hash = ConsistentHash(
            nodes=[f"shard_{i}" for i in range(config.shard_count)]
        ) if config.sharding_enabled else None
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=redis.RedisError
        )
    
    async def connect(self):
        """Connect to Redis"""
        self.pool = await aioredis.create_redis_pool(
            self.redis_url,
            minsize=5,
            maxsize=20
        )
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from Redis"""
        with cache_latency.labels('get', 'l3_redis').time():
            try:
                with self.circuit_breaker:
                    redis_key = self._get_redis_key(key)
                    data = await self.pool.get(redis_key)
                    
                    if data:
                        cache_hits.labels('l3_redis', self._get_key_pattern(key)).inc()
                        return self._deserialize(data)
                    
                    cache_misses.labels('l3_redis', self._get_key_pattern(key)).inc()
                    return None
                    
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in Redis"""
        with cache_latency.labels('set', 'l3_redis').time():
            try:
                with self.circuit_breaker:
                    redis_key = self._get_redis_key(key)
                    data = self._serialize(value)
                    
                    if ttl or self.config.ttl:
                        await self.pool.setex(
                            redis_key,
                            ttl or self.config.ttl,
                            data
                        )
                    else:
                        await self.pool.set(redis_key, data)
                    
                    return True
                    
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete from Redis"""
        try:
            with self.circuit_breaker:
                redis_key = self._get_redis_key(key)
                result = await self.pool.delete(redis_key)
                return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check existence in Redis"""
        try:
            with self.circuit_breaker:
                redis_key = self._get_redis_key(key)
                return await self.pool.exists(redis_key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear cache (pattern-based)"""
        try:
            with self.circuit_breaker:
                # Clear keys with our prefix
                cursor = 0
                pattern = f"{self.config.layer.value}:*"
                
                while True:
                    cursor, keys = await self.pool.scan(cursor, match=pattern, count=1000)
                    if keys:
                        await self.pool.delete(*keys)
                    if cursor == 0:
                        break
                
                return True
                
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis stats"""
        try:
            info = await self.pool.info()
            return {
                'layer': 'l3_redis',
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'evicted_keys': info.get('evicted_keys', 0)
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {}
    
    def _get_redis_key(self, key: str) -> str:
        """Get Redis key with sharding"""
        prefix = f"{self.config.layer.value}:{key}"
        
        if self.consistent_hash:
            shard = self.consistent_hash.get_node(key)
            return f"{shard}:{prefix}"
        
        return prefix
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value"""
        if self.config.compression:
            return lz4.frame.compress(msgpack.packb(value))
        return msgpack.packb(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value"""
        if self.config.compression:
            data = lz4.frame.decompress(data)
        return msgpack.unpackb(data, raw=False)
    
    def _get_key_pattern(self, key: str) -> str:
        """Extract pattern from key"""
        parts = key.split(':')
        if len(parts) >= 2:
            return f"{parts[0]}:*"
        return "other"


class CDNManager:
    """CDN management system"""
    
    def __init__(self, config: CDNConfig):
        self.config = config
        self.providers = {}
        self._initialize_providers()
        self.stats = defaultdict(int)
        self.bandwidth_tracker = defaultdict(int)
    
    def _initialize_providers(self):
        """Initialize CDN providers"""
        if self.config.provider == CDNProvider.CLOUDFLARE:
            self.providers['cloudflare'] = CloudflareProvider(self.config)
        elif self.config.provider == CDNProvider.FASTLY:
            self.providers['fastly'] = FastlyProvider(self.config)
        elif self.config.provider == CDNProvider.CLOUDFRONT:
            self.providers['cloudfront'] = CloudFrontProvider(self.config)
        # Add more providers as needed
    
    async def publish(
        self,
        key: str,
        content: bytes,
        content_type: str = 'application/octet-stream',
        cache_control: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Publish content to CDN"""
        
        # Apply bandwidth limiting
        if self.config.bandwidth_limit:
            await self._check_bandwidth_limit(len(content))
        
        # Prepare headers
        headers = {
            'Content-Type': content_type,
            'Cache-Control': cache_control or self.config.cache_control.get(
                content_type,
                'public, max-age=3600'
            )
        }
        
        # Add security headers
        headers.update(self.config.security_headers)
        
        # Add custom metadata
        if metadata:
            for k, v in metadata.items():
                headers[f'X-Meta-{k}'] = v
        
        # Publish to provider
        results = {}
        for name, provider in self.providers.items():
            try:
                result = await provider.publish(key, content, headers)
                results[name] = result
                
                # Track metrics
                cdn_requests.labels(result.get('region', 'unknown'), 'success').inc()
                cdn_bandwidth.labels(result.get('region', 'unknown'), content_type).inc(len(content))
                
            except Exception as e:
                logger.error(f"CDN publish error ({name}): {e}")
                results[name] = {'error': str(e)}
                cdn_requests.labels('unknown', 'error').inc()
        
        return results
    
    async def purge(
        self,
        keys: Union[str, List[str]],
        purge_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Purge content from CDN"""
        
        if isinstance(keys, str):
            keys = [keys]
        
        purge_type = purge_type or self.config.purge_strategy
        
        results = {}
        for name, provider in self.providers.items():
            try:
                result = await provider.purge(keys, purge_type)
                results[name] = result
            except Exception as e:
                logger.error(f"CDN purge error ({name}): {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    async def get_analytics(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get CDN analytics"""
        
        analytics = {}
        for name, provider in self.providers.items():
            try:
                data = await provider.get_analytics(start_time, end_time, metrics)
                analytics[name] = data
            except Exception as e:
                logger.error(f"CDN analytics error ({name}): {e}")
                analytics[name] = {'error': str(e)}
        
        return analytics
    
    async def configure_rules(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Configure CDN rules (page rules, behaviors, etc.)"""
        
        results = {}
        for name, provider in self.providers.items():
            try:
                result = await provider.configure_rules(rules)
                results[name] = result
            except Exception as e:
                logger.error(f"CDN configure error ({name}): {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    async def _check_bandwidth_limit(self, size: int):
        """Check and enforce bandwidth limits"""
        current_second = int(datetime.utcnow().timestamp())
        
        # Reset counter for new second
        if current_second not in self.bandwidth_tracker:
            self.bandwidth_tracker.clear()
            self.bandwidth_tracker[current_second] = 0
        
        # Check limit
        if self.bandwidth_tracker[current_second] + size > self.config.bandwidth_limit:
            # Wait until next second
            await asyncio.sleep(1.0 - (datetime.utcnow().timestamp() % 1))
        
        self.bandwidth_tracker[current_second] += size


class CloudflareProvider:
    """Cloudflare CDN provider"""
    
    def __init__(self, config: CDNConfig):
        self.config = config
        self.cf = cloudflare.Cloudflare(
            email=os.getenv('CLOUDFLARE_EMAIL'),
            token=os.getenv('CLOUDFLARE_API_TOKEN')
        )
    
    async def publish(self, key: str, content: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        """Publish to Cloudflare"""
        # Implementation would interact with Cloudflare API
        # This is a placeholder
        return {
            'status': 'published',
            'url': f'https://cdn.example.com/{key}',
            'region': 'global'
        }
    
    async def purge(self, keys: List[str], purge_type: str) -> Dict[str, Any]:
        """Purge from Cloudflare"""
        # Implementation would interact with Cloudflare API
        return {
            'status': 'purged',
            'keys': keys,
            'type': purge_type
        }
    
    async def get_analytics(
        self,
        start_time: datetime,
        end_time: datetime,
        metrics: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Get Cloudflare analytics"""
        # Implementation would fetch from Cloudflare Analytics API
        return {
            'requests': 1000000,
            'bandwidth': 1024 * 1024 * 1024 * 100,  # 100GB
            'cache_hit_rate': 0.95
        }
    
    async def configure_rules(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Configure Cloudflare page rules"""
        # Implementation would configure page rules via API
        return {
            'status': 'configured',
            'rules_count': len(rules)
        }


class MultilayerCache:
    """Main multilayer caching system"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.layers: Dict[CacheLayer, CacheInterface] = {}
        self.warmup_queue = asyncio.Queue()
        self.stats_engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.stats_engine)
        Base.metadata.create_all(self.stats_engine)
        self.ml_predictor = CachePredictionModel()
        
    async def initialize(self, configs: List[CacheConfig]):
        """Initialize cache layers"""
        for config in configs:
            if config.layer == CacheLayer.L1_MEMORY:
                self.layers[config.layer] = L1MemoryCache(config)
            elif config.layer == CacheLayer.L3_REDIS:
                cache = L3RedisCache(config, os.getenv('REDIS_URL', 'redis://localhost'))
                await cache.connect()
                self.layers[config.layer] = cache
            # Add more layers as needed
        
        # Start background tasks
        asyncio.create_task(self._warmup_worker())
        asyncio.create_task(self._stats_collector())
        asyncio.create_task(self._ml_optimizer())
    
    async def get(
        self,
        key: str,
        fetch_fn: Optional[Callable] = None,
        ttl: Optional[int] = None,
        layers: Optional[List[CacheLayer]] = None
    ) -> Optional[Any]:
        """Get value with cascade through layers"""
        
        layers = layers or list(self.layers.keys())
        
        # Try each layer
        for layer in sorted(layers, key=lambda x: x.value):
            if layer not in self.layers:
                continue
            
            cache = self.layers[layer]
            value = await cache.get(key)
            
            if value is not None:
                # Promote to higher layers
                await self._promote_to_higher_layers(key, value, ttl, layer)
                return value
        
        # Cache miss - fetch from origin if function provided
        if fetch_fn:
            value = await fetch_fn()
            if value is not None:
                # Store in all layers
                await self.set(key, value, ttl)
            return value
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        layers: Optional[List[CacheLayer]] = None,
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set value in specified layers"""
        
        layers = layers or list(self.layers.keys())
        results = []
        
        for layer in layers:
            if layer not in self.layers:
                continue
            
            cache = self.layers[layer]
            result = await cache.set(key, value, ttl)
            results.append(result)
        
        # Add to warmup queue if eligible
        if self._should_warmup(key, value):
            await self.warmup_queue.put({
                'key': key,
                'value': value,
                'ttl': ttl,
                'tags': tags or set()
            })
        
        return all(results)
    
    async def delete(self, key: str, layers: Optional[List[CacheLayer]] = None) -> bool:
        """Delete from specified layers"""
        
        layers = layers or list(self.layers.keys())
        results = []
        
        for layer in layers:
            if layer not in self.layers:
                continue
            
            cache = self.layers[layer]
            result = await cache.delete(key)
            results.append(result)
        
        return any(results)
    
    async def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate all entries with specified tags"""
        # This would require tag tracking implementation
        # For now, return placeholder
        return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'layers': {},
            'overall': {
                'total_hits': 0,
                'total_misses': 0,
                'total_requests': 0,
                'hit_rate': 0.0
            }
        }
        
        for layer, cache in self.layers.items():
            layer_stats = await cache.get_stats()
            stats['layers'][layer.value] = layer_stats
            
            # Aggregate stats
            stats['overall']['total_hits'] += layer_stats.get('hits', 0)
            stats['overall']['total_misses'] += layer_stats.get('misses', 0)
        
        stats['overall']['total_requests'] = (
            stats['overall']['total_hits'] + stats['overall']['total_misses']
        )
        
        if stats['overall']['total_requests'] > 0:
            stats['overall']['hit_rate'] = (
                stats['overall']['total_hits'] / stats['overall']['total_requests']
            )
        
        return stats
    
    async def optimize(self) -> Dict[str, Any]:
        """Run cache optimization"""
        recommendations = []
        
        # Analyze hit rates
        stats = await self.get_stats()
        
        for layer_name, layer_stats in stats['layers'].items():
            hit_rate = layer_stats.get('hit_rate', 0)
            
            if hit_rate < 0.5:
                recommendations.append({
                    'layer': layer_name,
                    'issue': 'low_hit_rate',
                    'recommendation': 'Consider increasing cache size or TTL',
                    'current_hit_rate': hit_rate
                })
            
            if layer_stats.get('evictions', 0) > layer_stats.get('sets', 0) * 0.5:
                recommendations.append({
                    'layer': layer_name,
                    'issue': 'high_eviction_rate',
                    'recommendation': 'Cache size may be too small',
                    'eviction_rate': layer_stats['evictions'] / layer_stats.get('sets', 1)
                })
        
        return {
            'recommendations': recommendations,
            'optimizations_applied': len(recommendations)
        }
    
    async def _promote_to_higher_layers(
        self,
        key: str,
        value: Any,
        ttl: Optional[int],
        found_layer: CacheLayer
    ):
        """Promote value to higher cache layers"""
        
        # Get layers higher than where we found the value
        higher_layers = [
            layer for layer in self.layers.keys()
            if layer.value < found_layer.value
        ]
        
        # Set in higher layers
        for layer in higher_layers:
            cache = self.layers[layer]
            await cache.set(key, value, ttl)
    
    def _should_warmup(self, key: str, value: Any) -> bool:
        """Determine if key should be warmed up"""
        
        # Use ML model to predict if this will be accessed frequently
        features = self._extract_features(key, value)
        prediction = self.ml_predictor.predict_access_frequency(features)
        
        return prediction > 0.7  # 70% chance of frequent access
    
    def _extract_features(self, key: str, value: Any) -> Dict[str, float]:
        """Extract features for ML prediction"""
        
        # Simple feature extraction
        return {
            'key_length': len(key),
            'value_size': len(str(value)),
            'has_user_prefix': 1.0 if key.startswith('user:') else 0.0,
            'has_api_prefix': 1.0 if key.startswith('api:') else 0.0,
            'hour_of_day': datetime.utcnow().hour,
            'day_of_week': datetime.utcnow().weekday()
        }
    
    async def _warmup_worker(self):
        """Background worker for cache warmup"""
        while True:
            try:
                item = await asyncio.wait_for(self.warmup_queue.get(), timeout=5.0)
                
                # Refresh the value
                if 'fetch_fn' in item:
                    value = await item['fetch_fn']()
                    await self.set(item['key'], value, item['ttl'])
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Warmup worker error: {e}")
    
    async def _stats_collector(self):
        """Collect and persist cache statistics"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                stats = await self.get_stats()
                session = self.Session()
                
                for layer_name, layer_stats in stats['layers'].items():
                    stat_entry = CacheStats(
                        layer=layer_name,
                        hits=layer_stats.get('hits', 0),
                        misses=layer_stats.get('misses', 0),
                        evictions=layer_stats.get('evictions', 0),
                        size_bytes=layer_stats.get('size_bytes', 0),
                        avg_latency_ms=layer_stats.get('avg_latency_ms', 0),
                        hit_rate=layer_stats.get('hit_rate', 0),
                        metadata=layer_stats
                    )
                    session.add(stat_entry)
                
                session.commit()
                session.close()
                
            except Exception as e:
                logger.error(f"Stats collector error: {e}")
    
    async def _ml_optimizer(self):
        """ML-based cache optimization"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Collect recent stats
                session = self.Session()
                recent_stats = session.query(CacheStats).filter(
                    CacheStats.timestamp > datetime.utcnow() - timedelta(hours=1)
                ).all()
                session.close()
                
                # Train/update ML model
                if len(recent_stats) > 100:
                    self.ml_predictor.update_model(recent_stats)
                
                # Apply optimizations
                optimizations = await self.optimize()
                logger.info(f"Cache optimizations: {optimizations}")
                
            except Exception as e:
                logger.error(f"ML optimizer error: {e}")


class CachePredictionModel:
    """ML model for cache prediction"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.feature_names = [
            'key_length', 'value_size', 'has_user_prefix',
            'has_api_prefix', 'hour_of_day', 'day_of_week'
        ]
    
    def predict_access_frequency(self, features: Dict[str, float]) -> float:
        """Predict access frequency for a cache entry"""
        
        if not self.is_trained:
            # Return default prediction
            return 0.5
        
        # Convert features to array
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Ensure prediction is between 0 and 1
        return max(0.0, min(1.0, prediction))
    
    def update_model(self, stats: List[CacheStats]):
        """Update model with new statistics"""
        
        # Extract features and labels from stats
        X = []
        y = []
        
        for stat in stats:
            if stat.metadata:
                features = [
                    stat.metadata.get(name, 0)
                    for name in self.feature_names
                ]
                X.append(features)
                y.append(stat.hit_rate)
        
        if len(X) > 10:
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            logger.info("Cache prediction model updated")


# Example usage
async def cache_demo():
    """Demo caching functionality"""
    
    # Initialize multilayer cache
    cache = MultilayerCache('postgresql://user:pass@localhost/cache_db')
    
    await cache.initialize([
        CacheConfig(
            layer=CacheLayer.L1_MEMORY,
            ttl=300,
            max_size=100 * 1024 * 1024,  # 100MB
            eviction_policy=EvictionPolicy.ADAPTIVE
        ),
        CacheConfig(
            layer=CacheLayer.L3_REDIS,
            ttl=3600,
            eviction_policy=EvictionPolicy.LRU,
            sharding_enabled=True,
            shard_count=16
        )
    ])
    
    # Set value
    await cache.set('user:123', {'name': 'John', 'role': 'admin'}, ttl=600)
    
    # Get value (will cascade through layers)
    user = await cache.get('user:123')
    print(f"User: {user}")
    
    # Get with fetch function
    async def fetch_user():
        # Simulate database fetch
        await asyncio.sleep(0.1)
        return {'name': 'Jane', 'role': 'user'}
    
    user2 = await cache.get('user:456', fetch_fn=fetch_user)
    print(f"User 2: {user2}")
    
    # Get cache stats
    stats = await cache.get_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")
    
    # Initialize CDN
    cdn_config = CDNConfig(
        provider=CDNProvider.CLOUDFLARE,
        zones=['example.com'],
        cache_control={
            'image/jpeg': 'public, max-age=86400',
            'application/json': 'public, max-age=300'
        }
    )
    
    cdn = CDNManager(cdn_config)
    
    # Publish to CDN
    result = await cdn.publish(
        'assets/logo.png',
        b'image data here',
        content_type='image/png'
    )
    print(f"CDN publish result: {result}")
    
    # Get CDN analytics
    analytics = await cdn.get_analytics(
        datetime.utcnow() - timedelta(days=1),
        datetime.utcnow()
    )
    print(f"CDN analytics: {analytics}")


if __name__ == "__main__":
    asyncio.run(cache_demo())