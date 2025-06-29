import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from MemAgent.memory.main import Memory

@pytest.fixture
def memory_instance():
    mem = Memory(db_path=":memory:") # Use in-memory SQLite for testing
    mem.reset() # Ensure a clean state for each test
    return mem

def test_add_and_get_memory(memory_instance):
    data = "This is a test memory."
    memory_instance.add(data, user_id="test_user", infer=False)
    
    # Retrieve the memory (assuming we know the content, we can search for it)
    results = memory_instance.search("test memory", user_id="test_user")
    assert len(results) > 0
    assert results[0]["document"] == data

def test_add_and_get_memory_with_agent_id(memory_instance):
    data = "This is a test memory for an agent."
    memory_instance.add(data, agent_id="test_agent", infer=False)
    
    results = memory_instance.search("test memory for agent", agent_id="test_agent")
    assert len(results) > 0
    assert results[0]["document"] == data

def test_history_tracking(memory_instance):
    data = "Initial memory."
    memory_instance.add(data, user_id="test_user", infer=False)
    
    results = memory_instance.search("Initial memory", user_id="test_user")
    memory_id = results[0]["id"]
    
    updated_data = "Updated memory."
    memory_instance.update(memory_id, updated_data)
    
    history = memory_instance.history(memory_id)
    assert len(history) == 2
    assert history[0][0] == "ADD"
    assert history[1][0] == "UPDATE"
    assert history[1][1] == updated_data

def test_delete_memory(memory_instance):
    data = "Memory to be deleted."
    memory_instance.add(data, user_id="test_user", infer=False)
    
    results = memory_instance.search("Memory to be deleted", user_id="test_user")
    memory_id = results[0]["id"]
    
    memory_instance.delete(memory_id)
    
    results_after_delete = memory_instance.search("Memory to be deleted", user_id="test_user")
    assert len(results_after_delete) == 0
    
    history = memory_instance.history(memory_id)
    assert len(history) == 2
    assert history[1][0] == "DELETE"

def test_reset_memory(memory_instance):
    memory_instance.add("Some data", user_id="user1", infer=False)
    memory_instance.add("More data", user_id="user2", infer=False)
    
    assert len(memory_instance.get_all()) == 2
    
    memory_instance.reset()
    
    assert len(memory_instance.get_all()) == 0