# Technical Debt & Known Issues

## Overview
This document tracks known technical debt and issues in the MemAgent codebase. Each item includes priority, impact, and suggested fixes.

## 🔴 Critical Issues

### 1. Thread Safety Problems
**Files**: `MemAgent/provider/sqlite.py`  
**Issue**: SQLite connections are not thread-safe, ChromaDB client not async-safe  
**Impact**: Potential data corruption, connection errors in concurrent environments  
**TODO Location**: `MemAgent/memory/base.py` line ~45

**Fix**:
```python
# Use aiosqlite with proper locking
import aiosqlite
import asyncio

class SQLiteProvider:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock()
    
    async def _get_connection(self):
        async with self._lock:
            async with aiosqlite.connect(self.db_path) as conn:
                yield conn
```

### 2. Tests Hit Live APIs
**Files**: `tests/test_memory.py`, `tests/test_sqlite_provider.py`  
**Issue**: Tests require live OpenAI API and ChromaDB  
**Impact**: Tests fail without API keys, consume API credits, slow test runs  
**TODO Location**: `tests/test_memory.py` line ~20

**Fix**: Implement comprehensive mocking (see Testing Guide)

### 3. No Retry Logic for API Calls
**Files**: `MemAgent/memory/main.py`, `MemAgent/embedding/provider.py`  
**Issue**: OpenAI API calls fail without retry on rate limits  
**Impact**: Transient failures cause operation failures  
**TODO Location**: `MemAgent/memory/base.py` line ~120

**Fix**:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def call_with_retry(func, *args, **kwargs):
    return await func(*args, **kwargs)
```

## 🟡 Important Issues

### 4. Inefficient Embedding Generation
**Files**: `MemAgent/memory/base.py`  
**Issue**: Embeddings fetched one at a time instead of batched  
**Impact**: 10x slower than necessary for bulk operations  
**TODO Location**: `MemAgent/memory/base.py` line ~89

### 5. No Cascade Delete
**Files**: `MemAgent/provider/sqlite.py`  
**Issue**: Deleting from SQLite doesn't remove from ChromaDB  
**Impact**: Orphaned data in vector store  
**TODO Location**: `MemAgent/memory/base.py` line ~156

### 6. Free-form LLM Response Parsing
**Files**: `MemAgent/memory/main.py`  
**Issue**: Parsing LLM responses with string matching instead of structured output  
**Impact**: Parsing failures, inconsistent results  
**TODO Location**: `MemAgent/memory/main.py` line ~78

### 7. Using print() Instead of Logging
**Files**: Throughout codebase  
**Issue**: Debug output uses print() instead of proper logging  
**Impact**: No log levels, can't disable in production  
**TODO Location**: `MemAgent/memory/main.py` line ~142

## 🟢 Minor Issues

### 8. Metadata Split Across Statements
**Files**: `MemAgent/provider/sqlite.py`  
**Issue**: Memory metadata written in two SQL statements  
**Impact**: Potential inconsistency if one fails  
**TODO Location**: `MemAgent/memory/base.py` line ~134

### 9. Raw Datetime Storage
**Files**: `MemAgent/model/record.py`  
**Issue**: Timestamps stored as raw datetime objects  
**Impact**: Timezone issues, serialization problems  
**TODO Location**: `MemAgent/memory/base.py` line ~145

### 10. No PII Handling
**Files**: All storage providers  
**Issue**: No mechanism to tag or encrypt PII data  
**Impact**: Compliance risks  
**Status**: Not addressed in current sprint

## Migration-Related Debt

### 11. Dual Provider Maintenance
**Status**: Temporary (until Week 8)  
**Issue**: Must maintain both SQLite and MongoDB providers  
**Impact**: Double the work for bug fixes  
**Resolution**: Week 8 - deprecate SQLiteProvider

### 12. Incomplete MongoDB Implementation
**Files**: `MemAgent/provider/mongo.py`  
**Issue**: Several methods still raise NotImplementedError  
**Impact**: Can't fully migrate yet  
**TODO**: Implement remaining methods

## Code Quality Issues

### 13. Missing Type Hints
**Files**: Various older files  
**Issue**: Incomplete type annotations  
**Impact**: Harder to catch type errors  
**Status**: Ongoing improvement

### 14. No GitHub Actions Workflow
**Files**: `.github/workflows/`  
**Issue**: No CI/CD pipeline  
**Impact**: No automated testing/linting  
**TODO Location**: `.github/workflows/main.yml`

## Proposed Solutions Priority

### Phase 1 (Weeks 8-9)
1. Fix thread safety issues
2. Implement proper mocking for tests
3. Add retry logic for API calls

### Phase 2 (Weeks 10-11)  
1. Batch embedding operations
2. Fix cascade delete
3. Switch to structured LLM outputs
4. Replace print() with logging

### Phase 3 (Week 12)
1. Add GitHub Actions CI/CD
2. Complete type hints
3. Document remaining issues

## How to Address Technical Debt

When working on any of these issues:

1. **Check the TODO location** in the code
2. **Follow the suggested fix** if provided
3. **Add tests** for your fix
4. **Update this document** when resolved

## Tracking Progress

```python
# Add this comment when fixing a TODO
# TODO: [RESOLVED] <your_github_username> <date>
# Original issue: <brief description>
# Fix: <what you did>
```

## Recent Resolutions

- Week 7: Moved decay logic to separate tasks module
- Week 6: Added basic match_tools implementation
- Week 5: Created Context Manager for prompt assembly