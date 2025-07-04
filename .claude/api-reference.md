# API Reference

## Core Classes

### Memory Class
```python
class Memory:
    def __init__(
        self,
        provider: MemoryProvider,
        embedder: Embedder,
        llm: Any
    ):
        """
        Initialize Memory with storage, embedding, and LLM providers.
        
        Args:
            provider: Storage backend implementing MemoryProvider
            embedder: Text embedding service
            llm: Language model for intelligent operations
        """
    
    def infer(self, content: str) -> MemRecord:
        """
        Intelligently process memory content.
        
        Automatically determines whether to ADD, UPDATE, or DELETE
        based on content analysis and existing memories.
        
        Args:
            content: Text content to process
            
        Returns:
            MemRecord with operation details
        """
    
    async def ainfer(self, content: str) -> MemRecord:
        """Async version of infer."""
    
    def search(
        self,
        query: str,
        limit: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> List[MemRecord]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            limit: Maximum results to return
            memory_types: Filter by memory types
            
        Returns:
            List of matching MemRecord objects
        """
```

### AsyncMemory Class
```python
class AsyncMemory:
    """Async-first version of Memory class with same interface."""
    # All methods are async versions of Memory methods
```

## Protocols

### MemoryProvider Protocol
```python
class MemoryProvider(Protocol):
    """Storage backend interface."""
    
    def add(self, record: MemRecord) -> MemRecord:
        """Add new memory record."""
    
    def update(self, id: str, record: MemRecord) -> MemRecord:
        """Update existing memory."""
    
    def delete(self, id: str) -> bool:
        """Delete memory by ID."""
    
    def get(self, id: str) -> Optional[MemRecord]:
        """Retrieve memory by ID."""
    
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemRecord]:
        """Search memories by semantic similarity."""
    
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MemRecord]:
        """List memories with pagination."""
    
    # Async versions (prefix with 'a')
    async def aadd(self, record: MemRecord) -> MemRecord: ...
    async def aupdate(self, id: str, record: MemRecord) -> MemRecord: ...
    async def adelete(self, id: str) -> bool: ...
    async def aget(self, id: str) -> Optional[MemRecord]: ...
    async def asearch(...) -> List[MemRecord]: ...
    async def alist(...) -> List[MemRecord]: ...
    
    # Specialized methods (MongoDB migration)
    async def log_turn(self, conversation_id: str, turn: Dict[str, Any]): ...
    async def log_workflow_step(self, workflow_id: str, step: Dict[str, Any]): ...
    async def stream_conversation(self, conversation_id: str): ...
    async def entity_facts(self, entity_id: str, k: int = 10): ...
    async def match_tools(self, msg: str, k: int = 4) -> List[ToolRecord]: ...
    async def persona(self, persona_id: str) -> Persona: ...
```

### Embedder Protocol
```python
class Embedder(Protocol):
    """Text embedding service interface."""
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (numpy array)
        """
    
    async def aembed(self, text: str) -> np.ndarray:
        """Async version of embed."""
    
    def batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts efficiently."""
    
    async def abatch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Async batch embedding."""
```

## Data Models

### MemRecord
```python
@dataclass
class MemRecord:
    """Core memory record."""
    
    # Required fields
    content: str                    # Memory content
    importance: int                 # 1-10 importance score
    
    # Auto-generated fields
    id: Optional[str] = None       # Unique identifier
    embedding: Optional[np.ndarray] = None  # Vector embedding
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Metadata
    memory_type: str = "general"   # Type classification
    sub_type: Optional[str] = None # Sub-classification
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Usage tracking
    access_count: int = 0
    decay_rate: float = 0.1
    
    # Operation tracking
    operation: Optional[str] = None  # ADD, UPDATE, DELETE
    previous_id: Optional[str] = None  # For updates
```

### ToolRecord
```python
@dataclass
class ToolRecord:
    """Available tool/function definition."""
    
    name: str                      # Tool name
    description: str               # What it does
    parameters: Dict[str, Any]     # Parameter schema
    examples: List[str] = field(default_factory=list)
    
    # Metadata
    id: Optional[str] = None
    category: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
```

### Persona
```python
@dataclass
class Persona:
    """Agent personality configuration."""
    
    name: str                      # Persona name
    description: str               # Personality description
    traits: List[str]              # Key traits
    guidelines: List[str]          # Behavioral guidelines
    
    # Metadata
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

## Context Management

### ContextManager
```python
async def build_prompt(
    user_msg: str,
    provider: MemoryProvider,
    session: SessionInfo,
    max_tokens: int = 8000
) -> str:
    """
    Assemble context-aware prompt.
    
    Args:
        user_msg: Current user message
        provider: Memory provider instance
        session: Session information
        max_tokens: Token budget
        
    Returns:
        Formatted prompt string
    """
```

## Background Tasks

### Decay Task
```python
async def decay_memories(
    provider: MemoryProvider,
    decay_rate: float = 0.1,
    min_importance: int = 1
) -> int:
    """
    Apply time-based decay to memories.
    
    Args:
        provider: Memory provider instance
        decay_rate: Decay factor per day
        min_importance: Minimum importance threshold
        
    Returns:
        Number of memories affected
    """
```

### Reflection Task
```python
async def reflect_on_memories(
    provider: MemoryProvider,
    llm: Any,
    batch_size: int = 10
) -> List[MemRecord]:
    """
    Synthesize insights from memories.
    
    Args:
        provider: Memory provider instance
        llm: Language model for synthesis
        batch_size: Memories per reflection
        
    Returns:
        New synthetic memory records
    """
```

## Exceptions

```python
class MemAgentError(Exception):
    """Base exception for MemAgent."""

class ProviderError(MemAgentError):
    """Storage provider errors."""

class EmbeddingError(MemAgentError):
    """Embedding service errors."""

class ValidationError(MemAgentError):
    """Input validation errors."""

class ConnectionError(ProviderError):
    """Database connection errors."""

class NotFoundError(ProviderError):
    """Memory not found errors."""
```

## Usage Examples

### Basic Usage
```python
from MemAgent.memory import Memory
from MemAgent.provider.sqlite import SQLiteProvider
from MemAgent.embedding.provider import OpenAIEmbedder

# Initialize
provider = SQLiteProvider("memories.db")
embedder = OpenAIEmbedder()
llm = OpenAI()

memory = Memory(provider, embedder, llm)

# Add memory
result = memory.infer("User's favorite color is blue")
print(f"Operation: {result.operation}")  # "ADD"

# Update memory
update = memory.infer("Actually, user prefers green")
print(f"Operation: {update.operation}")  # "UPDATE"

# Search memories
results = memory.search("color preferences", limit=5)
for mem in results:
    print(f"{mem.content} (importance: {mem.importance})")
```

### Async Usage
```python
from MemAgent.memory import AsyncMemory

# Initialize
memory = AsyncMemory(provider, embedder, llm)

# Async operations
async def process_memories():
    # Add memory
    result = await memory.ainfer("User works at OpenAI")
    
    # Batch operations
    memories = [
        "User likes Python",
        "User prefers VS Code",
        "User uses macOS"
    ]
    
    tasks = [memory.ainfer(m) for m in memories]
    results = await asyncio.gather(*tasks)
    
    # Search
    relevant = await memory.asearch("programming preferences")
    return relevant

# Run
memories = asyncio.run(process_memories())
```