import chromadb
import sqlite3
import uuid
import datetime
import re
import openai
from typing import Optional, List, Dict, Any, Tuple


class BaseMemory:
    """
    Base class for memory management, handling interactions with ChromaDB and SQLite.
    """

    def __init__(self, db_path: str = ":memory:", collection_name: str = "mem0") -> None:
        """
        Initializes the BaseMemory with a ChromaDB client and SQLite connection.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to an in-memory database.
            collection_name (str): The name of the ChromaDB collection. Defaults to "mem0".
        """
        self.client = chromadb.Client() # TODO: Consider a dedicated worker or use Chroma’s HTTP API for async-safety.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)  # TODO: Use aiosqlite or asyncio.Lock for more robust thread-safety
        self.collection_name = collection_name
        self._create_tables()

    def _create_tables(self) -> None:
        """
        Creates the necessary tables in the SQLite database for memory history and metadata.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_history (
                id TEXT PRIMARY KEY,
                memory_id TEXT,
                action TEXT,
                data TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_metadata (
                memory_id TEXT PRIMARY KEY,
                importance REAL,
                access_count INTEGER,
                last_accessed_timestamp DATETIME
            )
        """)
        self.conn.commit()

    def _parse_importance(self, text: str) -> float:
        """
        Parses the importance score from a given text.

        Args:
            text (str): The text containing the importance score.

        Returns:
            float: The parsed importance score.
        """
        match = re.search(r"\d+\.?\d*", text)
        if match:
            return float(match.group())
        return 0.0

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using OpenAI's embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding  # TODO: Batch calls or cache duplicates (simple LRU).

    def _get_llm_response(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """
        Gets a response from the OpenAI LLM.

        Args:
            prompt (str): The prompt to send to the LLM.
            model (str): The LLM model to use.

        Returns:
            str: The LLM's response.
        """
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content # TODO: Add exponential back-off and error handling. Centralise in a LLMClient helper with retry, timeout and telemetry.

    def _update_metadata(
        self, memory_id: str, importance: Optional[float] = None, access_count: Optional[int] = None
    ) -> None:
        """
        Updates the metadata for a given memory, including importance and access count.

        Args:
            memory_id (str): The ID of the memory to update.
            importance (Optional[float]): The new importance score.
            access_count (Optional[int]): The new access count.
        """
        cursor = self.conn.cursor()
        if importance is not None:
            cursor.execute(
                "INSERT OR REPLACE INTO memory_metadata (memory_id, importance, access_count, last_accessed_timestamp) VALUES (?, ?, ?, ?)",
                (memory_id, importance, 1, datetime.datetime.now().isoformat()),
            )
        if access_count is not None:
            cursor.execute(
                "UPDATE memory_metadata SET access_count = ?, last_accessed_timestamp = ? WHERE memory_id = ?",
                (access_count, datetime.datetime.now().isoformat(), memory_id),
            )
        self.conn.commit() # TODO: Add FK constraints or reconcile in a single unit-of-work abstraction for cascade delete.

    def _log_history(self, memory_id: str, action: str, data: str) -> None:
        """
        Logs an action performed on a memory to the history table.

        Args:
            memory_id (str): The ID of the memory.
            action (str): The action performed (e.g., "ADD", "UPDATE", "DELETE").
            data (str): The data associated with the action.
        """
        cursor = self.conn.cursor()
        history_id = str(uuid.uuid4())
        cursor.execute(
            "INSERT INTO memory_history (id, memory_id, action, data) VALUES (?, ?, ?, ?)",
            (history_id, memory_id, action, data),
        )
        self.conn.commit()

    def _add_memory_to_collection(
        self, data: str, user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
    ) -> str:
        """
        Adds a new memory to the ChromaDB collection.

        Args:
            data (str): The content of the memory.
            user_id (Optional[str]): The ID of the user associated with the memory.
            agent_id (Optional[str]): The ID of the agent associated with the memory.
            run_id (Optional[str]): The ID of the run associated with the memory.

        Returns:
            str: The ID of the newly added memory.
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        memory_id = str(uuid.uuid4())
        embedding = self._get_embedding(data)
        collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[data],
            metadatas=[
                {
                    k: v
                    for k, v in {
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "run_id": run_id,
                    }.items()
                    if v is not None
                }
            ],
        )
        return memory_id

    def _update_memory_in_collection(self, memory_id: str, data: str) -> None:
        """
        Updates an existing memory in the ChromaDB collection.

        Args:
            memory_id (str): The ID of the memory to update.
            data (str): The new content of the memory.
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        embedding = self._get_embedding(data)
        collection.update(ids=[memory_id], embeddings=[embedding], documents=[data])

    def _delete_memory_from_collection(self, memory_id: str) -> None:
        """
        Deletes a memory from the ChromaDB collection.

        Args:
            memory_id (str): The ID of the memory to delete.
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        collection.delete(ids=[memory_id])

    def _get_memory_from_collection(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a memory from the ChromaDB collection by its ID.

        Args:
            memory_id (str): The ID of the memory to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The memory details if found, otherwise None.
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        result = collection.get(ids=[memory_id])
        if result and result["documents"]:
            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0],
            }
        return None

    def _get_all_memories_from_collection(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all memories from the ChromaDB collection, optionally filtered by user, agent, or run ID.

        Args:
            user_id (Optional[str]): Filter memories by user ID.
            agent_id (Optional[str]): Filter memories by agent ID.
            run_id (Optional[str]): Filter memories by run ID.

        Returns:
            List[Dict[str, Any]]: A list of matching memories.
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        where_clause = {}
        if user_id:
            where_clause["user_id"] = user_id
        if agent_id:
            where_clause["agent_id"] = agent_id
        if run_id:
            where_clause["run_id"] = run_id

        results = collection.get(where=where_clause if where_clause else None)

        all_memories = []
        if results and results["documents"]:
            for i in range(len(results["ids"])):
                all_memories.append(
                    {
                        "id": results["ids"][i],
                        "document": results["documents"][i],
                        "metadata": results["metadatas"][i],
                    }
                )
        return all_memories

    def _delete_all_memories_from_collection(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> None:
        """
        Deletes all memories from the ChromaDB collection, optionally filtered by user, agent, or run ID.

        Args:
            user_id (Optional[str]): Filter memories by user ID.
            agent_id (Optional[str]): Filter memories by agent ID.
            run_id (Optional[str]): Filter memories by run ID.
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        where_clause = {}
        if user_id:
            where_clause["user_id"] = user_id
        if agent_id:
            where_clause["agent_id"] = agent_id
        if run_id:
            where_clause["run_id"] = run_id

        collection.delete(where=where_clause if where_clause else None)

    def history(self, memory_id: str) -> List[Tuple[str, str, str]]:
        """
        Retrieves the history of actions performed on a specific memory.

        Args:
            memory_id (str): The ID of the memory to retrieve history for.

        Returns:
            List[Tuple[str, str, str]]: A list of (action, data, timestamp) tuples.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT action, data, timestamp FROM memory_history WHERE memory_id = ? ORDER BY timestamp ASC",
            (memory_id,),
        )
        return cursor.fetchall()

    def reset(self) -> None:
        """
        Resets the memory system by deleting the ChromaDB collection and clearing SQLite tables.
        """
        try:
            self.client.delete_collection("mem0")
        except chromadb.errors.NotFoundError:
            pass
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS memory_history")
        cursor.execute("DROP TABLE IF EXISTS memory_metadata")
        self.conn.commit()
        self._create_tables()

    def calculate_strength(self, memory_id: str) -> float:
        """
        Calculates the strength of a memory based on its importance, access count, and recency.

        Args:
            memory_id (str): The ID of the memory.

        Returns:
            float: The calculated memory strength.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT importance, access_count, last_accessed_timestamp FROM memory_metadata WHERE memory_id = ?",
            (memory_id,),
        )
        row = cursor.fetchone()
        if row:
            importance, access_count, last_accessed_timestamp = row
            recency = (
                datetime.datetime.now()
                - datetime.datetime.fromisoformat(last_accessed_timestamp)
            ).total_seconds() / 3600  # Time in hours # TODO: Timestamps stored as raw datetime – SQLite driver turns into string silently. Store .isoformat() explicitly; consider integer epoch for speed.
            return importance * (1 + access_count) * (1 / (1 + recency))
        return 0.0

    def decay_memories(self, threshold: float = 0.5) -> None:
        """
        Decays memories with strength below a given threshold.

        Args:
            threshold (float): The strength threshold for decaying memories.
        """
        all_memories = self._get_all_memories_from_collection()
        for memory in all_memories:
            strength = self.calculate_strength(memory["id"])
            if strength < threshold:
                self._delete_memory_from_collection(memory["id"])
                self._log_history(memory["id"], "DECAY", "")
