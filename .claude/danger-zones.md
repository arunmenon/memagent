# ⚠️ Danger Zones - Handle with Care

## Critical Areas

### 1. Memory Provider Methods
**DANGER**: Never call provider methods directly
```python
# ❌ WRONG
provider.add(record)
provider.update(id, record)

# ✅ CORRECT
memory.infer(content)
await memory.ainfer(content)
```

### 2. Async/Sync Safety
**DANGER**: Blocking operations in async context
```python
# ❌ WRONG - Blocks event loop
async def bad():
    time.sleep(1)
    result = collection.query(...)  # Sync ChromaDB call

# ✅ CORRECT
async def good():
    await asyncio.sleep(1)
    result = await asyncio.to_thread(collection.query, ...)
```

### 3. API Key Management
**DANGER**: Exposed credentials
```python
# ❌ WRONG
openai.api_key = "sk-abc123..."
print(f"Using key: {api_key}")

# ✅ CORRECT
openai.api_key = os.getenv("OPENAI_API_KEY")
logger.info("API key configured")
```

### 4. Database Connections
**DANGER**: Thread safety issues
```python
# ❌ WRONG
self.conn = sqlite3.connect(db_path)  # Shared across threads

# ✅ CORRECT (for now)
self.conn = sqlite3.connect(db_path, check_same_thread=False)
# TODO: Migrate to aiosqlite for proper async support
```

### 5. Importance Score Validation
**DANGER**: Invalid importance ranges
```python
# ❌ WRONG
importance = float(llm_response)  # Could be any value

# ✅ CORRECT
importance = max(1, min(10, int(float(llm_response))))
```

## Red Flags in Code Review
1. Direct provider method calls (add, update, delete)
2. Sync operations in async methods
3. Hardcoded values that should be configurable
4. Missing error handling in LLM calls
5. Uncached embeddings for duplicate content