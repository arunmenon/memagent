# Error Handling Guide

## Common Errors and Solutions

### 1. Memory Not Found
```python
# Problem
result = provider.get("non-existent-id")  # Returns None or raises

# Solution
try:
    result = provider.get(memory_id)
    if result is None:
        # Handle gracefully
        logger.warning(f"Memory {memory_id} not found")
        return default_memory
except NotFoundError:
    # Or handle exception
    logger.error(f"Memory {memory_id} does not exist")
    raise
```

### 2. Embedding Service Failures
```python
# Problem: API rate limits, network issues
embedding = embedder.embed(text)  # May fail

# Solution: Retry with exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def embed_with_retry(embedder, text):
    try:
        return await embedder.aembed(text)
    except RateLimitError:
        logger.warning("Rate limit hit, retrying...")
        raise
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        # Fallback to zero vector or cached embedding
        return np.zeros(1536)
```

### 3. Database Connection Issues
```python
# Problem: Connection drops, timeouts
provider = MongoProvider(uri)  # May fail to connect

# Solution: Connection management
class ResilientProvider(MongoProvider):
    async def __aenter__(self):
        retries = 3
        for i in range(retries):
            try:
                await self.connect()
                return self
            except ConnectionError as e:
                if i == retries - 1:
                    raise
                logger.warning(f"Connection attempt {i+1} failed, retrying...")
                await asyncio.sleep(2 ** i)
```

### 4. LLM Parsing Errors
```python
# Problem: LLM returns malformed JSON
response = llm.complete(prompt)
data = json.loads(response)  # May fail

# Solution: Structured parsing with validation
from pydantic import BaseModel, ValidationError

class InferResponse(BaseModel):
    operation: str
    importance: int
    reason: Optional[str] = None

def parse_llm_response(response: str) -> InferResponse:
    try:
        # Try JSON first
        data = json.loads(response)
        return InferResponse(**data)
    except (json.JSONDecodeError, ValidationError):
        # Fallback to regex parsing
        import re
        operation_match = re.search(r'operation["\s:]+(\w+)', response, re.I)
        importance_match = re.search(r'importance["\s:]+(\d+)', response, re.I)
        
        if operation_match and importance_match:
            return InferResponse(
                operation=operation_match.group(1).upper(),
                importance=int(importance_match.group(1))
            )
        
        # Final fallback
        logger.error(f"Failed to parse LLM response: {response}")
        return InferResponse(operation="ADD", importance=5)
```

### 5. Thread Safety Issues
```python
# Problem: SQLite "database is locked"
conn.execute("INSERT INTO memories...")  # May fail in threads

# Solution: Use connection pool or async
import aiosqlite

class AsyncSQLiteProvider:
    async def _get_connection(self):
        # New connection per operation
        async with aiosqlite.connect(self.db_path) as conn:
            yield conn
    
    async def aadd(self, record: MemRecord):
        async with self._get_connection() as conn:
            await conn.execute(...)
            await conn.commit()
```

### 6. Memory Overflow
```python
# Problem: Too many memories returned
results = provider.search(query, limit=10000)  # Memory issue

# Solution: Pagination and streaming
async def stream_search_results(provider, query, batch_size=100):
    offset = 0
    while True:
        batch = await provider.asearch(
            query,
            limit=batch_size,
            offset=offset
        )
        if not batch:
            break
        
        for record in batch:
            yield record
        
        offset += batch_size

# Usage
async for memory in stream_search_results(provider, "user preferences"):
    process(memory)
```

## Error Handling Patterns

### 1. Graceful Degradation
```python
async def get_memory_with_fallback(provider, memory_id):
    try:
        # Try primary provider
        return await provider.aget(memory_id)
    except ProviderError:
        # Fall back to cache
        try:
            return await cache_provider.aget(memory_id)
        except:
            # Final fallback
            return MemRecord(
                content="[Memory temporarily unavailable]",
                importance=5
            )
```

### 2. Circuit Breaker Pattern
```python
from circuitbreaker import circuit

class EmbedderWithCircuitBreaker:
    def __init__(self, embedder):
        self.embedder = embedder
        self.fallback_cache = {}
    
    @circuit(failure_threshold=5, recovery_timeout=30)
    async def aembed(self, text: str) -> np.ndarray:
        return await self.embedder.aembed(text)
    
    async def safe_embed(self, text: str) -> np.ndarray:
        try:
            return await self.aembed(text)
        except:
            # Use cached embedding if available
            if text in self.fallback_cache:
                return self.fallback_cache[text]
            # Or return zero vector
            return np.zeros(1536)
```

### 3. Validation and Sanitization
```python
def validate_memory_content(content: str) -> str:
    """Validate and sanitize memory content."""
    if not content or not isinstance(content, str):
        raise ValidationError("Content must be non-empty string")
    
    # Remove potentially harmful content
    content = content.strip()
    
    # Enforce length limits
    if len(content) > 10000:
        logger.warning(f"Content truncated from {len(content)} chars")
        content = content[:10000]
    
    # Remove null bytes
    content = content.replace('\x00', '')
    
    return content
```

## Debugging Failed Operations

### Enable Debug Logging
```python
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add context to logs
logger = logging.getLogger(__name__)

async def debug_search(provider, query):
    logger.debug(f"Search started: query='{query}'")
    
    try:
        results = await provider.asearch(query)
        logger.debug(f"Search completed: found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Search failed: query='{query}', error={e}", exc_info=True)
        raise
```

### Add Telemetry
```python
from prometheus_client import Counter, Histogram

# Metrics
memory_operations = Counter(
    'memagent_operations_total',
    'Total memory operations',
    ['operation', 'status']
)

operation_duration = Histogram(
    'memagent_operation_duration_seconds',
    'Memory operation duration'
)

# Track operations
async def track_operation(operation_type: str, func):
    with operation_duration.time():
        try:
            result = await func()
            memory_operations.labels(
                operation=operation_type,
                status='success'
            ).inc()
            return result
        except Exception as e:
            memory_operations.labels(
                operation=operation_type,
                status='error'
            ).inc()
            raise
```

### Debug Helper Functions
```python
async def diagnose_provider(provider: MemoryProvider):
    """Run diagnostic tests on provider."""
    results = {
        "connection": False,
        "read": False,
        "write": False,
        "search": False
    }
    
    # Test connection
    try:
        await provider.alist(limit=1)
        results["connection"] = True
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
    
    # Test write
    try:
        test_record = MemRecord(content="Diagnostic test", importance=1)
        written = await provider.aadd(test_record)
        results["write"] = written.id is not None
        
        # Test read
        if written.id:
            read = await provider.aget(written.id)
            results["read"] = read is not None
            
            # Cleanup
            await provider.adelete(written.id)
    except Exception as e:
        logger.error(f"Read/write test failed: {e}")
    
    # Test search
    try:
        search_results = await provider.asearch("test", limit=1)
        results["search"] = True
    except Exception as e:
        logger.error(f"Search test failed: {e}")
    
    return results
```

## Production Considerations

### 1. Always Log Errors with Context
```python
logger.error(
    "Memory operation failed",
    extra={
        "operation": "add",
        "memory_id": record.id,
        "content_length": len(record.content),
        "importance": record.importance,
        "error": str(e)
    }
)
```

### 2. Monitor Error Rates
```python
# Set up alerts for:
- Error rate > 5% over 5 minutes
- Connection failures > 3 in 1 minute  
- Embedding service timeout > 10 seconds
- Memory operation p99 latency > 1 second
```

### 3. Implement Health Checks
```python
async def health_check():
    """Health check endpoint."""
    checks = {
        "database": False,
        "embedder": False,
        "llm": False
    }
    
    try:
        # Check database
        await provider.alist(limit=1)
        checks["database"] = True
    except:
        pass
    
    try:
        # Check embedder
        await embedder.aembed("health check")
        checks["embedder"] = True
    except:
        pass
    
    try:
        # Check LLM
        await llm.complete("test", max_tokens=1)
        checks["llm"] = True
    except:
        pass
    
    status = "healthy" if all(checks.values()) else "unhealthy"
    return {"status": status, "checks": checks}
```