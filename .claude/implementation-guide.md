# Implementation Guide

## Creating a New Storage Provider

### Step 1: Create the provider file
```bash
touch MemAgent/provider/dynamodb.py
```

### Step 2: Import required modules
```python
from typing import List, Optional, Dict, Any
import asyncio
import uuid
from datetime import datetime
import boto3

from MemAgent.provider.abc import MemoryProvider
from MemAgent.model.record import MemRecord
```

### Step 3: Implement the provider class
```python
class DynamoDBProvider(MemoryProvider):
    def __init__(self, table_name: str, region: str = "us-east-1"):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)
```

### Step 4: Implement all protocol methods
- Start with async methods (preferred)
- Add sync wrappers using asyncio.run()
- Include proper error handling
- Add history tracking

### Step 5: Create comprehensive tests
```bash
touch tests/test_dynamodb_provider.py
```

### Step 6: Update __init__.py to export the provider
```python
# In MemAgent/provider/__init__.py
from .dynamodb import DynamoDBProvider
```

## Example: Redis Provider Implementation

```python
from MemAgent.provider.abc import MemoryProvider
from MemAgent.model.record import MemRecord
from typing import List, Optional

class RedisProvider(MemoryProvider):
    def __init__(self, redis_url: str):
        """Initialize with Redis connection"""
        self.redis = redis.from_url(redis_url)
        self.embeddings_key = "memories:embeddings"
        self.metadata_key = "memories:metadata"
    
    async def aadd(self, record: MemRecord) -> MemRecord:
        # 1. Generate ID if not present
        if not record.id:
            record.id = str(uuid.uuid4())
        
        # 2. Store embeddings
        await self.redis.hset(
            self.embeddings_key,
            record.id,
            json.dumps(record.embedding.tolist())
        )
        
        # 3. Store metadata
        metadata = {
            "content": record.content,
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
            "metadata": record.metadata
        }
        await self.redis.hset(
            self.metadata_key,
            record.id,
            json.dumps(metadata)
        )
        
        # 4. Update history
        await self._add_to_history(record.id, "ADD")
        
        return record
```

## Adding a New Embedding Provider

### Step 1: Understand the Embedder protocol
```python
class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...
    async def aembed(self, text: str) -> np.ndarray: ...
```

### Step 2: Create implementation
```python
class CohereEmbedder(Embedder):
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)
    
    async def aembed(self, text: str) -> np.ndarray:
        response = await self.client.embed(
            texts=[text],
            model="embed-english-v3.0"
        )
        return np.array(response.embeddings[0])
```

### Step 3: Add caching if appropriate
### Step 4: Test with various text inputs
### Step 5: Ensure dimension compatibility

## Extending Memory with Custom Features

### Step 1: Subclass MemRecord for custom fields
```python
from MemAgent.model.record import MemRecord

class ExtendedMemRecord(MemRecord):
    tags: List[str] = []
    source: str = "user_input"
    confidence: float = 1.0
```

### Step 2: Update provider to handle new fields
### Step 3: Modify infer logic if needed
### Step 4: Update tests for new functionality

## Common Implementation Patterns

### Connection Management
```python
# ✅ ALWAYS use context managers or proper cleanup
class SQLiteProvider(MemoryProvider):
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self):
        self.conn = await aiosqlite.connect(self.db_path)
        self.chroma_client = chromadb.AsyncClient()
    
    async def disconnect(self):
        if self.conn:
            await self.conn.close()
```

### Batch Operations
```python
# ✅ Batch when possible
async def batch_infer(memory: Memory, contents: List[str]):
    tasks = [memory.ainfer(content) for content in contents]
    return await asyncio.gather(*tasks)
```

### Embedding Caching
```python
# ✅ Cache embeddings for repeated content
class CachedEmbedder(Embedder):
    def __init__(self, base_embedder: Embedder):
        self.base = base_embedder
        self.cache = {}
    
    async def aembed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        
        embedding = await self.base.aembed(text)
        self.cache[text] = embedding
        return embedding
```

## Database Schema Reference

### SQLite Schema (SQLiteProvider)
```sql
-- Memories table
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    importance INTEGER CHECK(importance >= 1 AND importance <= 10),
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    accessed_at TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    metadata TEXT -- JSON
);

-- History table
CREATE TABLE memory_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT,
    operation TEXT CHECK(operation IN ('ADD', 'UPDATE', 'DELETE')),
    timestamp TIMESTAMP,
    old_content TEXT,
    new_content TEXT,
    FOREIGN KEY(memory_id) REFERENCES memories(id)
);
```

### ChromaDB Collections
- Collection name: "memories"
- Metadata fields: All MemRecord fields except embedding
- Embedding dimension: 1536 (OpenAI default)