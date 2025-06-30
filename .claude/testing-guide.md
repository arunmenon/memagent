# Testing Guide

## Testing Philosophy
- Test both sync and async versions of every method
- Mock external services (OpenAI, databases)
- Use fixtures for setup/teardown
- Aim for >80% coverage

## Test Structure

### Basic Test Template
```python
import pytest
from MemAgent.provider.redis import RedisProvider
from MemAgent.model.record import MemRecord

@pytest.fixture
async def redis_provider():
    provider = RedisProvider("redis://localhost:6379/0")
    yield provider
    # Cleanup
    await provider.clear_all()

@pytest.mark.asyncio
async def test_add_memory(redis_provider):
    # Create test memory
    memory = MemRecord(
        content="Test memory",
        importance=5,
        embedding=np.random.rand(1536)
    )
    
    # Add memory
    result = await redis_provider.aadd(memory)
    
    # Verify
    assert result.id is not None
    assert result.content == "Test memory"
    assert result.importance == 5
    
    # Verify searchable
    results = await redis_provider.asearch("Test", limit=1)
    assert len(results) == 1
    assert results[0].id == result.id

# ALWAYS test both sync and async
def test_add_memory_sync(redis_provider):
    memory = MemRecord(content="Test", importance=5)
    result = redis_provider.add(memory)
    assert result.id is not None
```

## Mocking External Services

### Mock OpenAI Embeddings
```python
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_embedder():
    embedder = Mock()
    embedder.aembed = AsyncMock(return_value=np.random.rand(1536))
    embedder.embed = Mock(return_value=np.random.rand(1536))
    return embedder
```

### Mock LLM Responses
```python
@pytest.fixture
def mock_llm():
    llm = Mock()
    llm.chat.completions.create = Mock(
        return_value=Mock(
            choices=[Mock(
                message=Mock(
                    content='{"operation": "ADD", "importance": 7}'
                )
            )]
        )
    )
    return llm
```

### Mock Database Connections
```python
@pytest.fixture
async def mock_mongo_provider(monkeypatch):
    # Mock motor client
    mock_client = AsyncMock()
    mock_db = AsyncMock()
    mock_collection = AsyncMock()
    
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    
    monkeypatch.setattr(
        "motor.motor_asyncio.AsyncIOMotorClient",
        Mock(return_value=mock_client)
    )
    
    provider = MongoProvider("mongodb://mock")
    return provider
```

## Testing Patterns

### Test Error Handling
```python
@pytest.mark.asyncio
async def test_provider_connection_error(mock_provider):
    mock_provider.connect = AsyncMock(side_effect=ConnectionError())
    
    with pytest.raises(ConnectionError):
        async with mock_provider:
            pass
```

### Test Retry Logic
```python
@pytest.mark.asyncio
async def test_retry_on_rate_limit(mock_embedder):
    # First call fails, second succeeds
    mock_embedder.aembed.side_effect = [
        RateLimitError(),
        np.random.rand(1536)
    ]
    
    result = await embed_with_retry(mock_embedder, "test")
    assert result is not None
    assert mock_embedder.aembed.call_count == 2
```

### Test Batch Operations
```python
@pytest.mark.asyncio
async def test_batch_embed(mock_embedder):
    texts = ["text1", "text2", "text3"]
    mock_embedder.abatch_embed = AsyncMock(
        return_value=[np.random.rand(1536) for _ in texts]
    )
    
    results = await batch_embed(mock_embedder, texts)
    assert len(results) == 3
    mock_embedder.abatch_embed.assert_called_once_with(texts)
```

## Integration Tests

### Docker-Based Tests
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_mongo_integration():
    # Requires docker-compose up
    provider = MongoProvider("mongodb://localhost:27017/test")
    
    # Real operations
    memory = MemRecord(content="Integration test")
    result = await provider.aadd(memory)
    
    # Verify in real DB
    found = await provider.asearch("Integration", limit=1)
    assert len(found) == 1
```

### End-to-End Tests
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_memory_flow():
    # Real components
    provider = SQLiteProvider(":memory:")
    embedder = OpenAIEmbedder()  # Requires API key
    llm = OpenAI()
    
    memory = Memory(provider, embedder, llm)
    
    # Full flow
    result = await memory.ainfer("User likes Python")
    assert result.operation == "ADD"
    
    # Update
    update = await memory.ainfer("User loves Python")
    assert update.operation == "UPDATE"
```

## Test Configuration

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
markers =
    integration: Integration tests requiring external services
    e2e: End-to-end tests
    slow: Tests that take > 5 seconds
```

### Coverage Configuration
```toml
# pyproject.toml
[tool.coverage.run]
source = ["MemAgent"]
omit = ["tests/*", "*/migrations/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

## Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_memory.py

# Specific test
pytest tests/test_memory.py::test_infer_add

# With coverage
pytest --cov=MemAgent --cov-report=html

# Skip integration tests
pytest -m "not integration"

# Only fast tests
pytest -m "not slow"

# Parallel execution
pytest -n 4
```

## Common Test Issues

### Issue: Tests hit live APIs
**Solution**: Always mock external services
```python
# Wrong
def test_embed():
    embedder = OpenAIEmbedder()  # Hits API!
    
# Right  
def test_embed(mock_embedder):
    result = mock_embedder.embed("test")
```

### Issue: Async test failures
**Solution**: Use pytest-asyncio properly
```python
# Wrong
def test_async_method():
    asyncio.run(provider.aadd(record))
    
# Right
@pytest.mark.asyncio
async def test_async_method():
    await provider.aadd(record)
```

### Issue: Database state leakage
**Solution**: Use fixtures with cleanup
```python
@pytest.fixture
async def clean_db():
    db = Database(":memory:")
    yield db
    await db.clear_all()  # Always cleanup
```