# Migration Status

## ⚠️ Important: Build for MongoDB, Not SQLite!

**All new features should target MongoProvider**. SQLiteProvider is being deprecated and will be removed after the migration completes.

## Current Status: Week 7 of 12 ✅

### Overview
The codebase is undergoing a major architectural shift from SQLite+ChromaDB to MongoDB Atlas with 8 typed collections.

### Migration Timeline

| Week | Status | Focus Area |
|------|--------|------------|
| 0 | ✅ Complete | Extract MemoryProvider protocol |
| 1 | ✅ Complete | Implement MongoProvider skeleton |
| 2 | ✅ Complete | Create Embedder abstraction |
| 3 | ✅ Complete | Add typed memory records |
| 4 | ✅ Complete | Add conversation & workflow helpers |
| 5 | ✅ Complete | Create Context Manager |
| 6 | ✅ Complete | Add reranker & tool-match |
| 7 | ✅ Complete | Move decay & reflection to tasks |
| 8 | 🔄 In Progress | Deprecate SQLiteProvider |
| 9-10 | ⏳ Pending | Add observability & config |
| 11 | ⏳ Pending | Documentation & notebooks |
| 12 | ⏳ Pending | Cut v0.3 release |

### Architecture Changes

#### From (Legacy)
```
BaseMemory
├── SQLite (metadata)
├── ChromaDB (vectors)
└── Monolithic operations
```

#### To (Target)
```
MemoryProvider Protocol
├── MongoProvider
│   ├── conversation_mem
│   ├── episodic_mem
│   ├── semantic_mem
│   ├── procedural_mem
│   ├── persona
│   ├── toolbox
│   ├── workflow
│   └── cache
└── Pluggable architecture
```

### What This Means for Contributors

#### ✅ DO
- Implement new features for MongoProvider
- Follow the MemoryProvider protocol
- Add tests with mocked dependencies
- Use async methods (ainfer, asearch, etc.)

#### ❌ DON'T
- Add new features to SQLiteProvider
- Use ChromaDB directly
- Create tight coupling to storage
- Skip protocol methods

### Migration Helpers

#### Using Both Providers (Temporary)
```python
# During migration, both providers coexist
if use_mongo:
    provider = MongoProvider(mongodb_uri)
else:
    provider = SQLiteProvider(db_path)

memory = Memory(provider, embedder, llm)
```

#### Feature Flags
```python
# Check which provider to use
FEATURES = {
    "use_mongodb": os.getenv("USE_MONGODB", "false").lower() == "true",
    "enable_reranker": True,
    "batch_embeddings": True,
}
```

### Known Issues During Migration

1. **Both providers must be maintained** until Week 8
2. **Tests may need updates** when switching providers
3. **Performance differences** between SQLite and MongoDB
4. **Schema mismatches** being resolved incrementally

### Data Migration Script (Week 8)
```bash
# Will be available in Week 8
python scripts/migrate_sqlite_to_mongo.py \
    --source-db memories.db \
    --target-uri mongodb://localhost:27017
```

### Post-Migration Benefits

1. **Better scalability** with MongoDB Atlas
2. **Native vector search** (no separate ChromaDB)
3. **Typed collections** for different memory types
4. **Built-in TTL** and expiration
5. **Cloud-native** deployment options