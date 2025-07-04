# Setup Guide

## Environment Setup

### Required Environment Variables
```bash
# OpenAI API Key (Required)
export OPENAI_API_KEY="sk-..."
```

### Optional Environment Variables
```bash
# MongoDB Connection (for MongoProvider)
export MONGODB_URI="mongodb://localhost:27017"

# Custom embedder settings
export EMBEDDING_MODEL="text-embedding-ada-002"
export EMBEDDING_DIMENSION="1536"

# API endpoints (if using proxies)
export OPENAI_API_BASE="https://api.openai.com/v1"

# Development settings
export MEMAGENT_LOG_LEVEL="DEBUG"
export MEMAGENT_ENV="development"
```

## Installation

### Using Poetry (Recommended)
```bash
# Install dependencies
poetry install

# Install with development dependencies
poetry install --with dev

# Update dependencies
poetry update
```

### Using pip
```bash
# Install from pyproject.toml
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Quick Start Commands

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sqlite_provider.py

# Run with coverage
pytest --cov=MemAgent

# Run only unit tests (skip integration)
pytest -m "not integration"

# Run tests in parallel
pytest -n 4
```

### Development Commands
```bash
# Format code
ruff format MemAgent tests

# Lint code
ruff check MemAgent tests

# Type checking
mypy MemAgent

# Run all checks
make check  # if Makefile exists
```

## Key Imports Reference

### Core Classes
```python
from MemAgent.memory import Memory, AsyncMemory
from MemAgent.provider.sqlite import SQLiteProvider
from MemAgent.provider.mongo import MongoProvider
from MemAgent.embedding.provider import OpenAIEmbedder
```

### Models
```python
from MemAgent.model.record import MemRecord, ToolRecord, Persona
```

### Protocols
```python
from MemAgent.provider.abc import MemoryProvider
from MemAgent.embedding.provider import Embedder
```

### Context Management
```python
from MemAgent.context.manager import ContextManager, build_prompt
```

### Background Tasks
```python
from MemAgent.tasks.decay import DecayTasks, decay_memories
from MemAgent.tasks.reflect import ReflectTasks
```

### Exceptions
```python
from MemAgent.exceptions import (
    MemAgentError,
    ProviderError,
    EmbeddingError,
    ValidationError
)
```

## Common Usage Patterns

### Basic Setup
```python
# Minimal setup
provider = SQLiteProvider("memories.db")
embedder = OpenAIEmbedder()
llm = OpenAI()
memory = Memory(provider, embedder, llm)
```

### Async Setup
```python
# Async setup with context manager
async def main():
    async with MongoProvider("mongodb://localhost") as provider:
        embedder = OpenAIEmbedder()
        llm = AsyncOpenAI()
        memory = AsyncMemory(provider, embedder, llm)
        
        # Use memory
        await memory.ainfer("Important information")
```

### Development Setup
```python
# Development with mock providers
from tests.mocks import MockProvider, MockEmbedder

provider = MockProvider()
embedder = MockEmbedder()
llm = MockLLM()
memory = Memory(provider, embedder, llm)
```

## Docker Setup (Optional)

### MongoDB for Development
```yaml
# docker-compose.yml
version: '3.8'
services:
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

### Run Services
```bash
# Start MongoDB
docker-compose up -d mongodb

# Check logs
docker-compose logs -f mongodb

# Stop services
docker-compose down
```

## Troubleshooting Setup Issues

### Issue: Import errors
```bash
# Ensure you're in the project root
cd /path/to/MemAgent

# Reinstall
poetry install
```

### Issue: OpenAI API key not found
```bash
# Check environment
echo $OPENAI_API_KEY

# Set in .env file
echo "OPENAI_API_KEY=sk-..." > .env

# Load .env (if using python-dotenv)
python -c "from dotenv import load_dotenv; load_dotenv()"
```

### Issue: MongoDB connection failed
```bash
# Check MongoDB is running
docker ps | grep mongo

# Test connection
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://localhost:27017'); print(client.server_info())"
```

## IDE Setup

### VS Code
```json
// .vscode/settings.json
{
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

### PyCharm
1. Set Python interpreter to Poetry environment
2. Mark `MemAgent` as Sources Root
3. Configure pytest as test runner
4. Enable type checking