# Code Standards & Style Guide

## 🚨 Critical Rules (Must Follow)

1. **Always use `infer()`** - Never call provider methods directly for memory operations
2. **Async first** - Always prefer `ainfer()` over `infer()`, `asearch()` over `search()`
3. **Follow protocols** - MemoryProvider and Embedder are contracts that must be implemented fully
4. **Test everything** - Both sync and async versions of every method must have tests

## Python Style

### General Rules
- Follow PEP 8
- Use type hints for all functions
- Write docstrings for all public methods
- Prefer descriptive names over comments

### Naming Conventions
```python
# Classes: PascalCase
class MemoryProvider:
    pass

# Functions/Methods: snake_case
def calculate_importance(content: str) -> int:
    pass

# Constants: UPPER_SNAKE_CASE
DEFAULT_IMPORTANCE = 5
MAX_RETRIES = 3

# Private methods: leading underscore
def _internal_helper(self) -> None:
    pass
```

### Type Hints
```python
# Always use type hints
from typing import List, Optional, Dict, Any, Protocol

def search(
    self,
    query: str,
    limit: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> List[MemRecord]:
    """Search for memories matching query."""
    pass
```

### Async Conventions
```python
# Async methods prefixed with 'a'
async def ainfer(self, content: str) -> MemRecord:
    pass

# Sync wrapper pattern
def infer(self, content: str) -> MemRecord:
    return asyncio.run(self.ainfer(content))
```

## Project-Specific Rules

### Memory Operations
```python
# ❌ NEVER do direct CRUD
provider.add(record)  # Wrong!

# ✅ ALWAYS use infer
memory.infer("User information")  # Right!
```

### Importance Scores
```python
# Always validate range
if not 1 <= importance <= 10:
    raise ValueError(f"Importance must be 1-10, got {importance}")
```

### Error Handling
```python
# Be specific with exceptions
try:
    result = await provider.asearch(query)
except ProviderError as e:
    logger.error(f"Provider error: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Re-raise or handle appropriately
```

## Documentation Standards

### Docstring Format
```python
def complex_method(
    self,
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """
    Brief description of what the method does.
    
    Longer explanation if needed, describing the algorithm,
    important considerations, or usage examples.
    
    Args:
        param1: Description of param1
        param2: Description of param2, defaults to None
        
    Returns:
        Description of the return value
        
    Raises:
        ValueError: When param1 is empty
        ProviderError: When storage operation fails
        
    Example:
        >>> result = obj.complex_method("test", param2=5)
        >>> print(result["status"])
        'success'
    """
    pass
```

### Inline Comments
```python
# Use sparingly, prefer descriptive names
embedding = await self.embedder.aembed(content)  # Good name, no comment needed

# Use for non-obvious logic
importance = min(10, base_score * 1.5)  # Cap at 10 to maintain range
```

## Security Standards

### Never Expose Secrets
```python
# ❌ NEVER
print(f"Using API key: {api_key}")
logger.info(f"Connection string: {db_uri}")

# ✅ ALWAYS
logger.info("API key configured")
logger.info(f"Connecting to database at {urlparse(db_uri).hostname}")
```

### Input Validation
```python
def add_memory(self, content: str) -> MemRecord:
    # Validate input
    if not content or not content.strip():
        raise ValueError("Content cannot be empty")
    
    # Sanitize if needed
    content = content.strip()[:1000]  # Limit length
```

## Performance Standards

### Batch Operations
```python
# ❌ Inefficient
for text in texts:
    embedding = await embedder.aembed(text)

# ✅ Efficient
embeddings = await embedder.abatch_embed(texts)
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(input: str) -> float:
    # Cache results of expensive operations
    pass
```

## Testing Standards

### Test Naming
```python
def test_provider_adds_memory_successfully():
    """Test names should be descriptive sentences."""
    pass

def test_search_returns_empty_list_when_no_matches():
    """Describe the expected behavior."""
    pass
```

### Test Structure
```python
def test_memory_operation():
    # Arrange
    provider = MockProvider()
    memory = MemRecord(content="test")
    
    # Act
    result = provider.add(memory)
    
    # Assert
    assert result.id is not None
    assert result.content == "test"
```

## Git & PR Standards

### Commit Messages
```
feat: Add MongoDB provider implementation
fix: Handle connection timeout in embedder
docs: Update API reference for search method
test: Add integration tests for decay task
refactor: Extract validation logic to helper
```

### PR Checklist
Before submitting a PR, ensure:
- [ ] All tests pass (`pytest`)
- [ ] Type hints added
- [ ] Docstrings for public methods
- [ ] No hardcoded values
- [ ] No exposed secrets
- [ ] Error handling added
- [ ] Tests added/updated
- [ ] Documentation updated

## Code Review Focus

When reviewing PRs, check for:
1. **Protocol compliance** - Does it follow MemoryProvider?
2. **Async safety** - No blocking calls in async methods?
3. **Error handling** - Graceful failures?
4. **Test coverage** - Both sync and async tested?
5. **Performance** - Batch operations where possible?
6. **Security** - No exposed secrets or injection risks?

## Common Mistakes to Avoid

### ❌ DON'T
```python
# Direct provider calls
provider.add(memory)

# Hardcoded importance
memory.importance = 10

# Sync in async
async def bad():
    time.sleep(1)  # Blocks!

# Print debugging
print(f"Memory: {memory}")

# Ignore errors
try:
    risky_operation()
except:
    pass  # Never do this!
```

### ✅ DO
```python
# Use infer
memory.infer(content)

# Dynamic importance
memory.importance = llm.rate_importance(content)

# Async sleep
async def good():
    await asyncio.sleep(1)

# Proper logging
logger.debug(f"Processing memory ID: {memory.id}")

# Handle errors
try:
    risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise
```