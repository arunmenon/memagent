# Performance Guide

## Key Performance Principles

1. **Batch Operations** - Process multiple items together
2. **Async Everything** - Never block the event loop
3. **Cache Wisely** - Cache embeddings and frequent queries
4. **Limit Searches** - Use appropriate limits and filters

## Optimization Patterns

### Batch Embedding
```python
# ❌ Slow - One at a time
embeddings = []
for text in texts:
    embedding = await embedder.aembed(text)
    embeddings.append(embedding)

# ✅ Fast - Batch processing
embeddings = await embedder.abatch_embed(texts)
```

### Concurrent Operations
```python
# ❌ Sequential
results = []
for query in queries:
    result = await provider.asearch(query)
    results.extend(result)

# ✅ Concurrent
tasks = [provider.asearch(query) for query in queries]
all_results = await asyncio.gather(*tasks)
results = [item for sublist in all_results for item in sublist]
```

### Caching Strategy
```python
from functools import lru_cache
from aiocache import cached

# Sync caching
@lru_cache(maxsize=1000)
def get_embedding_dimension(model: str) -> int:
    return {"ada-002": 1536, "voyage-01": 1024}.get(model, 1536)

# Async caching
@cached(ttl=300)  # Cache for 5 minutes
async def get_cached_embedding(embedder: Embedder, text: str):
    return await embedder.aembed(text)
```

### Search Optimization
```python
# ✅ Use filters to reduce search space
results = await provider.asearch(
    query="user preferences",
    filters={
        "memory_type": "semantic",
        "importance": {"$gte": 7},
        "created_at": {"$gte": datetime.utcnow() - timedelta(days=30)}
    },
    limit=10
)

# ✅ Use projections to reduce data transfer
results = await provider.asearch(
    query="recent conversations",
    projection=["content", "importance"],  # Only fetch needed fields
    limit=20
)
```

### Database Optimization

#### Indexes (MongoDB)
```python
# Create indexes for common queries
async def create_indexes(provider: MongoProvider):
    # Vector search index
    await provider.collections["semantic"].create_index([
        ("embedding", "vector"),
        ("importance", -1)
    ])
    
    # Compound index for filtering
    await provider.collections["episodic"].create_index([
        ("created_at", -1),
        ("importance", -1),
        ("memory_type", 1)
    ])
```

#### Connection Pooling
```python
# MongoDB connection pooling
class MongoProvider:
    def __init__(self, uri: str):
        self.client = AsyncIOMotorClient(
            uri,
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=30000
        )
```

### Memory Management

#### Streaming Large Results
```python
async def stream_memories(provider: MemoryProvider, query: str):
    """Stream results to avoid memory overflow."""
    cursor = provider.collections["semantic"].find(
        {"$text": {"$search": query}}
    ).limit(1000)
    
    async for document in cursor:
        yield MemRecord(**document)
        # Process one at a time to manage memory
```

#### Garbage Collection
```python
import gc

async def process_large_dataset(provider: MemoryProvider):
    batch_size = 100
    
    for offset in range(0, 10000, batch_size):
        batch = await provider.alist(limit=batch_size, offset=offset)
        
        # Process batch
        await process_batch(batch)
        
        # Explicit cleanup for large batches
        del batch
        if offset % 1000 == 0:
            gc.collect()
```

## Performance Monitoring

### Metrics to Track
```python
from prometheus_client import Histogram, Counter

# Latency metrics
operation_duration = Histogram(
    'memagent_operation_duration_seconds',
    'Duration of memory operations',
    ['operation_type']
)

# Throughput metrics
memories_processed = Counter(
    'memagent_memories_processed_total',
    'Total memories processed',
    ['operation_type']
)

# Usage
async def monitored_search(provider, query):
    with operation_duration.labels('search').time():
        results = await provider.asearch(query)
        memories_processed.labels('search').inc(len(results))
        return results
```

### Profiling
```python
import cProfile
import asyncio

# Profile async code
async def profile_memory_operations():
    memory = AsyncMemory(provider, embedder, llm)
    
    # Profile infer operations
    for i in range(100):
        await memory.ainfer(f"Test memory {i}")

# Run profiler
profiler = cProfile.Profile()
profiler.enable()
asyncio.run(profile_memory_operations())
profiler.disable()
profiler.print_stats(sort='cumulative')
```

## Common Performance Issues

### Issue: Slow Embedding Generation
**Solution**: Batch embeddings and implement caching
```python
class CachedBatchEmbedder:
    def __init__(self, embedder: Embedder, batch_size: int = 50):
        self.embedder = embedder
        self.batch_size = batch_size
        self.cache = {}
    
    async def abatch_embed(self, texts: List[str]) -> List[np.ndarray]:
        # Check cache
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self.cache:
                results.append(self.cache[text])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch embed uncached
        if uncached_texts:
            embeddings = await self.embedder.abatch_embed(uncached_texts)
            
            # Update cache and results
            for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
                self.cache[text] = embedding
                results[idx] = embedding
        
        return results
```

### Issue: Memory Leaks in Long-Running Processes
**Solution**: Implement connection recycling
```python
class ConnectionManager:
    def __init__(self, provider_factory):
        self.provider_factory = provider_factory
        self.provider = None
        self.operation_count = 0
        self.max_operations = 10000
    
    async def get_provider(self):
        if self.provider is None or self.operation_count >= self.max_operations:
            if self.provider:
                await self.provider.close()
            
            self.provider = await self.provider_factory()
            self.operation_count = 0
        
        self.operation_count += 1
        return self.provider
```

### Issue: Slow Search Performance
**Solution**: Use pre-filtering and early termination
```python
async def optimized_search(provider, query, required_importance=7):
    # Pre-filter to reduce search space
    high_importance_ids = await provider.db.memories.distinct(
        "_id",
        {"importance": {"$gte": required_importance}}
    )
    
    # Search only within filtered set
    results = await provider.asearch(
        query,
        filter={"_id": {"$in": high_importance_ids}},
        limit=10
    )
    
    return results
```

## Performance Checklist

Before deploying:
- [ ] Batch operations implemented where possible
- [ ] Appropriate indexes created
- [ ] Connection pooling configured
- [ ] Caching strategy in place
- [ ] Memory limits set for large operations
- [ ] Monitoring/metrics implemented
- [ ] Load testing performed
- [ ] Resource cleanup verified