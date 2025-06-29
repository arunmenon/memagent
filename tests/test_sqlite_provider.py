import pytest
import os
import asyncio
from MemAgent.provider.sqlite import SQLiteProvider
from MemAgent.model.record import MemRecord, ToolRecord, Persona
import datetime

@pytest.fixture
def sqlite_provider():
    db_path = "test_memory.db"
    provider = SQLiteProvider(db_path=db_path, collection_name="test_collection")
    yield provider
    provider.reset()
    os.remove(db_path)

@pytest.mark.asyncio
async def test_upsert_and_search(sqlite_provider):
    # Test upsert
    rec = MemRecord(
        id="test_id_1",
        memory_type="episodic",
        text="This is a test memory.",
        meta={"user": "test_user"},
        ts=datetime.datetime.now().timestamp(),
        salience=0.8
    )
    await sqlite_provider.upsert(rec)

    # Test search
    results = await sqlite_provider.search(query="test memory", mem_types=["episodic"], k=1)
    assert len(results) == 1
    assert results[0].id == "test_id_1"
    assert results[0].text == "This is a test memory."

@pytest.mark.asyncio
async def test_reset(sqlite_provider):
    rec = MemRecord(
        id="test_id_2",
        memory_type="semantic",
        text="Another test memory.",
        meta={"agent": "test_agent"},
        ts=datetime.datetime.now().timestamp(),
        salience=0.5
    )
    await sqlite_provider.upsert(rec)
    
    results_before_reset = await sqlite_provider.search(query="test memory", mem_types=["semantic"], k=1)
    assert len(results_before_reset) == 1

    sqlite_provider.reset()

    results_after_reset = await sqlite_provider.search(query="test memory", mem_types=["semantic"], k=1)
    assert len(results_after_reset) == 0

@pytest.mark.asyncio
async def test_history(sqlite_provider):
    rec = MemRecord(
        id="test_id_3",
        memory_type="conversation",
        text="Conversation entry.",
        meta={"conv_id": "conv123"},
        ts=datetime.datetime.now().timestamp(),
        salience=0.7
    )
    await sqlite_provider.upsert(rec)
    history = sqlite_provider.history("test_id_3")
    assert len(history) == 1
    assert history[0][0] == "UPSERT"

@pytest.mark.asyncio
async def test_calculate_strength(sqlite_provider):
    rec = MemRecord(
        id="test_id_4",
        memory_type="procedural",
        text="How to do something.",
        meta={},
        ts=datetime.datetime.now().timestamp(),
        salience=0.9
    )
    await sqlite_provider.upsert(rec)
    strength = sqlite_provider.calculate_strength("test_id_4")
    assert strength > 0

# Test for unimplemented methods (should pass without error, but not do anything)
@pytest.mark.asyncio
async def test_unimplemented_methods(sqlite_provider):
    await sqlite_provider.stream_conversation("some_conv_id")
    await sqlite_provider.log_workflow_step("some_workflow_id", {"step": 1})
    await sqlite_provider.entity_facts("some_entity_id")
    await sqlite_provider.match_tools("some message")
    await sqlite_provider.persona("some_persona_id")
