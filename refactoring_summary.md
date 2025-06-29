## Refactoring Summary

### Changes Made:

1.  **`MemAgent/memory/base.py` Refactoring:**
    *   Moved common synchronous logic from `Memory` to `BaseMemory`.
    *   Introduced methods for `_get_embedding`, `_get_llm_response`, `_update_metadata`, and `_log_history` to `BaseMemory`.
    *   Added methods for direct interaction with the ChromaDB collection: `_add_memory_to_collection`, `_update_memory_in_collection`, `_delete_memory_from_collection`, `_get_memory_from_collection`, `_get_all_memories_from_collection`, and `_delete_all_memories_from_collection`.
    *   The `calculate_strength` and `decay_memories` methods were also moved to `BaseMemory`.

2.  **`MemAgent/memory/main.py` Updates:**
    *   Modified the `Memory` class to call the newly introduced methods in `BaseMemory`, significantly reducing code duplication.
    *   Updated the `AsyncMemory` class to wrap calls to `BaseMemory` methods with `asyncio.to_thread` for asynchronous execution, while retaining its own `_get_embedding` and `_get_llm_response` for `openai.AsyncOpenAI` calls.

### Issues Encountered:

*   **Package Installation:** Faced difficulties installing `pydantic-settings` due to an `externally-managed-environment` error and issues with `poetry` commands not being directly accessible.
*   **Test Execution:** Unable to run `pytest` successfully after the refactoring, likely stemming from the package installation issues and environment configuration.