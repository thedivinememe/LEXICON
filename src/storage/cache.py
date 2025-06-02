"""
Redis cache interface for LEXICON.
"""

import json
import pickle
from typing import Any, Dict, Optional, Union
import redis.asyncio as redis

class RedisCache:
    """Redis cache interface for LEXICON"""
    
    def __init__(self, redis_url: str):
        """Initialize Redis cache with connection URL"""
        self.redis_url = redis_url
        self.client = None
    
    async def connect(self):
        """Connect to Redis"""
        if self.client is None:
            self.client = redis.from_url(self.redis_url)
            # Test connection
            await self.client.ping()
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            self.client = None
    
    async def get(self, key: str) -> Any:
        """Get a value from cache"""
        if not self.client:
            await self.connect()
        
        value = await self.client.get(key)
        if value is None:
            return None
        
        try:
            # Try to deserialize as JSON first
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                # If not JSON, try pickle
                return pickle.loads(value)
            except:
                # If all else fails, return as string
                return value.decode('utf-8')
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set a value in cache with optional expiration in seconds"""
        if not self.client:
            await self.connect()
        
        try:
            # Try to serialize as JSON first
            serialized = json.dumps(value)
        except (TypeError, OverflowError):
            # If not JSON serializable, use pickle
            serialized = pickle.dumps(value)
        
        if expire:
            return await self.client.setex(key, expire, serialized)
        else:
            return await self.client.set(key, serialized)
    
    async def delete(self, key: str) -> int:
        """Delete a key from cache"""
        if not self.client:
            await self.connect()
        
        return await self.client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache"""
        if not self.client:
            await self.connect()
        
        return await self.client.exists(key) > 0
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in cache"""
        if not self.client:
            await self.connect()
        
        return await self.client.incrby(key, amount)
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key"""
        if not self.client:
            await self.connect()
        
        return await self.client.expire(key, seconds)
    
    async def publish(self, channel: str, message: Union[str, Dict]) -> int:
        """Publish a message to a channel"""
        if not self.client:
            await self.connect()
        
        if isinstance(message, dict):
            message = json.dumps(message)
        
        return await self.client.publish(channel, message)
    
    async def subscribe(self, channel: str):
        """Subscribe to a channel and yield messages"""
        if not self.client:
            await self.connect()
        
        pubsub = self.client.pubsub()
        await pubsub.subscribe(channel)
        
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                try:
                    # Try to parse as JSON
                    yield json.loads(message["data"])
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, return as string
                    if isinstance(message["data"], bytes):
                        yield message["data"].decode('utf-8')
                    else:
                        yield message["data"]
    
    async def flush_all(self) -> bool:
        """Clear the entire cache (use with caution)"""
        if not self.client:
            await self.connect()
        
        return await self.client.flushall()


class DummyCache:
    """Dummy cache implementation for when Redis is not available"""
    
    def __init__(self):
        """Initialize dummy cache"""
        self.cache = {}
    
    async def connect(self):
        """Connect to dummy cache (no-op)"""
        pass
    
    async def disconnect(self):
        """Disconnect from dummy cache (no-op)"""
        pass
    
    async def get(self, key: str) -> Any:
        """Get a value from dummy cache"""
        return self.cache.get(key)
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set a value in dummy cache (ignores expiration)"""
        self.cache[key] = value
        return True
    
    async def delete(self, key: str) -> int:
        """Delete a key from dummy cache"""
        if key in self.cache:
            del self.cache[key]
            return 1
        return 0
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in dummy cache"""
        return key in self.cache
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value in dummy cache"""
        if key not in self.cache:
            self.cache[key] = 0
        
        if not isinstance(self.cache[key], (int, float)):
            self.cache[key] = 0
        
        self.cache[key] += amount
        return self.cache[key]
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key (no-op in dummy cache)"""
        return key in self.cache
    
    async def publish(self, channel: str, message: Union[str, Dict]) -> int:
        """Publish a message to a channel (no-op in dummy cache)"""
        return 0
    
    async def subscribe(self, channel: str):
        """Subscribe to a channel (yields nothing in dummy cache)"""
        # In async functions, we can't use yield from
        # Just return an empty async generator
        if False:  # This will never execute but makes it an async generator
            yield None
        return
    
    async def flush_all(self) -> bool:
        """Clear the entire dummy cache"""
        self.cache.clear()
        return True
