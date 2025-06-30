from typing import Protocol, List, Optional, Dict, Any
from MemAgent.model.record import MemRecord, ToolRecord, Persona


class MemoryProvider(Protocol):
    """Protocol defining the interface for memory storage providers."""
    
    # Async CRUD operations
    async def aadd(self, record: MemRecord) -> MemRecord:
        """Add a new memory record."""
        ...
    
    async def aupdate(self, id: str, record: MemRecord) -> MemRecord:
        """Update an existing memory record."""
        ...
    
    async def adelete(self, id: str) -> bool:
        """Delete a memory record by ID."""
        ...
    
    async def aget(self, id: str) -> Optional[MemRecord]:
        """Get a memory record by ID."""
        ...
    
    async def asearch(self, query: str, limit: int = 10) -> List[MemRecord]:
        """Search for memory records."""
        ...
    
    async def alist(self, limit: int = 100, offset: int = 0, filters: Optional[Dict[str, Any]] = None) -> List[MemRecord]:
        """List memory records with pagination and filters."""
        ...
    
    # Sync CRUD operations
    def add(self, record: MemRecord) -> MemRecord:
        """Sync version of add."""
        ...
    
    def update(self, id: str, record: MemRecord) -> MemRecord:
        """Sync version of update."""
        ...
    
    def delete(self, id: str) -> bool:
        """Sync version of delete."""
        ...
    
    def get(self, id: str) -> Optional[MemRecord]:
        """Sync version of get."""
        ...
    
    def search(self, query: str, limit: int = 10) -> List[MemRecord]:
        """Sync version of search."""
        ...
    
    def list(self, limit: int = 100, offset: int = 0, filters: Optional[Dict[str, Any]] = None) -> List[MemRecord]:
        """Sync version of list."""
        ...
    
    # Specialized methods
    async def log_turn(self, turn_data: Dict[str, Any]) -> MemRecord:
        """Log a conversation turn."""
        ...
    
    async def log_workflow_step(self, workflow_id: str, step: dict):
        """Log a workflow step."""
        ...
    
    async def stream_conversation(self, conv_id: str):
        """Stream conversation history."""
        ...
    
    async def entity_facts(self, entity_id: str, k: int = 10):
        """Get facts about an entity."""
        ...
    
    async def match_tools(self, msg: str, k: int = 4) -> List[ToolRecord]:
        """Match tools for a query."""
        ...
    
    async def persona(self, persona_id: str) -> Persona:
        """Get persona information."""
        ...
    
    # Legacy method for backward compatibility
    async def upsert(self, rec: MemRecord) -> None:
        """Legacy upsert method - use aadd or aupdate instead."""
        ...