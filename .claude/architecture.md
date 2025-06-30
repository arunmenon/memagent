# MemAgent Architecture

## Overview
MemAgent is an intelligent memory management system for AI agents that combines traditional database capabilities with vector search and LLM-powered decision making. The system automatically determines whether to add, update, or delete memories based on content analysis rather than explicit CRUD operations.

## Core Architecture

### Project Structure
```
MemAgent/
├── memory/          # Core memory management
│   ├── __init__.py
│   └── main.py     # Memory and AsyncMemory classes
├── provider/        # Storage backends
│   ├── abc.py      # MemoryProvider protocol
│   ├── sqlite.py   # SQLite + ChromaDB implementation
│   └── mongo.py    # MongoDB implementation
├── model/          # Data models
│   └── record.py   # MemRecord, ToolRecord, Persona
├── embedding/      # Text embedding services
│   └── provider.py # Embedder protocol, OpenAIEmbedder
├── context/        # Context management
│   └── manager.py  # Prompt assembly
└── tasks/          # Background tasks
    ├── decay.py    # Memory decay
    └── reflect.py  # Memory reflection
```

## Key Components

### Memory Layer (`memory/main.py`)
- **Memory/AsyncMemory Classes**: Main API for memory operations
- Intelligent decision making using LLM to infer ADD/UPDATE/DELETE operations
- Importance scoring (1-10) for each memory
- Support for both sync and async operations

### Provider Layer (`provider/`)
- **MemoryProvider Protocol**: Defines storage interface
- **SQLiteProvider**: 
  - ChromaDB for vector storage and similarity search
  - SQLite for metadata and audit trail
  - Full history tracking of all operations
- **MongoProvider**: Alternative MongoDB-based storage (in development)

### Model Layer (`model/record.py`)
- **MemRecord**: Core memory structure with:
  - Content, metadata, embeddings
  - Importance score, timestamps
  - Access tracking
- **ToolRecord**: Available agent tools/functions
- **Persona**: Agent personality information

### Embedding Layer (`embedding/provider.py`)
- **Embedder Protocol**: Abstract interface for embedding services
- **OpenAIEmbedder**: Default implementation using OpenAI embeddings
- Extensible for other embedding providers

### Context Management (`context/manager.py`)
- Assembles prompts with relevant memories
- Integrates conversation history, tools, and persona
- Manages context window limits

### Background Tasks (`tasks/`)
- **Memory Decay**: Automatic importance reduction based on:
  - Time since creation
  - Access frequency
  - Current importance
- **Reflection**: Placeholder for memory synthesis

## Key Design Patterns

1. **Protocol-Oriented Design**: Uses Python Protocols for clean interfaces
2. **Repository Pattern**: Storage providers abstract data persistence
3. **Strategy Pattern**: Swappable embedding and storage providers
4. **Async/Sync Duality**: Both patterns supported throughout
5. **Audit Trail**: Complete history of all memory operations

## Memory Operations Flow

1. **Adding Memory**:
   ```python
   memory = Memory(provider, embedder, llm)
   await memory.ainfer("User's favorite color is blue")
   # LLM determines this is ADD operation
   # Generates importance score
   # Stores with embeddings
   ```

2. **Updating Memory**:
   ```python
   await memory.ainfer("Actually, user prefers green")
   # LLM identifies conflict with existing memory
   # Determines UPDATE operation
   # Updates existing record
   ```

3. **Semantic Search**:
   ```python
   memories = await memory.provider.asearch("color preferences", limit=5)
   # Uses vector similarity search
   # Returns relevant memories
   ```

## Architecture Principles

1. **Intelligence First**: LLM-driven decision making rather than explicit CRUD
2. **Extensibility**: Protocol-based design for easy provider swapping
3. **Hybrid Storage**: Combines vector and traditional databases
4. **Async Native**: Built for high-performance async applications
5. **Audit Complete**: Full history tracking for all operations
6. **Context Aware**: Smart prompt assembly with token management

## MongoDB Migration Architecture

The system is transitioning to a MongoDB-centric design with 8 typed collections:
- `conversation_mem`: Recent dialog turns
- `episodic_mem`: Event sequences
- `semantic_mem`: Facts and concepts
- `procedural_mem`: How-to knowledge
- `persona`: Agent personality
- `toolbox`: Available tools
- `workflow`: Multi-step processes
- `cache`: Temporary storage

Each collection has specific schemas, TTLs, and access patterns optimized for its memory type.