"""Main Memory and AsyncMemory classes for MemAgent."""

from typing import Any, List, Optional
from .model.record import MemRecord
from .provider.abc import MemoryProvider


class Memory:
    """Main interface for memory operations with intelligent inference capabilities."""
    
    def __init__(self, provider: MemoryProvider, embedder: Any, llm: Any):
        """Initialize Memory with provider, embedder, and LLM.
        
        Args:
            provider: Memory storage provider implementing MemoryProvider protocol
            embedder: Embedding service for text->vector conversion
            llm: Language model for intelligent operations
        """
        self.provider = provider
        self.embedder = embedder
        self.llm = llm
    
    def infer(self, content: str) -> MemRecord:
        """Intelligently process memory content to determine operation type.
        
        Analyzes the content to determine if it should be:
        - Added as a new memory
        - Used to update an existing memory
        - Used to delete a memory
        
        Args:
            content: The memory content to process
            
        Returns:
            MemRecord with appropriate operation and metadata
        """
        # Sync implementation - actual logic would use LLM to determine operation
        # For now, return a placeholder implementation
        operation = "ADD"  # Would be determined by LLM
        importance = 5  # Would be calculated by LLM
        
        record = MemRecord(
            content=content,
            importance=importance,
            operation=operation
        )
        
        # Execute the determined operation
        if operation == "ADD":
            return self.provider.add(record)
        elif operation == "UPDATE":
            # Would find related record first
            return self.provider.update("placeholder_id", record)
        elif operation == "DELETE":
            # Would find record to delete first
            self.provider.delete("placeholder_id")
            return record
        
        return record
    
    def search(self, query: str, limit: int = 10) -> List[MemRecord]:
        """Search for memories matching the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            
        Returns:
            List of matching MemRecord objects
        """
        return self.provider.search(query, limit=limit)


class AsyncMemory:
    """Async version of Memory with intelligent inference capabilities."""
    
    def __init__(self, provider: MemoryProvider, embedder: Any, llm: Any):
        """Initialize AsyncMemory with provider, embedder, and LLM.
        
        Args:
            provider: Memory storage provider implementing MemoryProvider protocol
            embedder: Embedding service for text->vector conversion
            llm: Language model for intelligent operations
        """
        self.provider = provider
        self.embedder = embedder
        self.llm = llm
    
    async def ainfer(self, content: str) -> MemRecord:
        """Async version of infer - intelligently process memory content.
        
        Analyzes the content to determine if it should be:
        - Added as a new memory
        - Used to update an existing memory
        - Used to delete a memory
        
        Args:
            content: The memory content to process
            
        Returns:
            MemRecord with appropriate operation and metadata
        """
        # Async implementation - actual logic would use LLM to determine operation
        # For now, return a placeholder implementation
        operation = "ADD"  # Would be determined by LLM
        importance = 5  # Would be calculated by LLM
        
        record = MemRecord(
            content=content,
            importance=importance,
            operation=operation
        )
        
        # Execute the determined operation
        if operation == "ADD":
            return await self.provider.aadd(record)
        elif operation == "UPDATE":
            # Would find related record first
            return await self.provider.aupdate("placeholder_id", record)
        elif operation == "DELETE":
            # Would find record to delete first
            await self.provider.adelete("placeholder_id")
            return record
        
        return record
    
    async def asearch(self, query: str, limit: int = 10) -> List[MemRecord]:
        """Async search for memories matching the query.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            
        Returns:
            List of matching MemRecord objects
        """
        return await self.provider.asearch(query, limit=limit)