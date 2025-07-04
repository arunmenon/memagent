import chromadb
import sqlite3
import uuid
import datetime
import re
import openai
from typing import Optional, List, Dict, Any, Tuple


from MemAgent.provider.abc import MemoryProvider
from MemAgent.model.record import MemRecord, ToolRecord, Persona

class SQLiteProvider(MemoryProvider):
    """
    Base class for memory management, handling interactions with ChromaDB and SQLite.
    """

    def __init__(self, db_path: str = ":memory:", collection_name: str = "mem0") -> None:
        super().__init__()
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

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for the given text using OpenAI's embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding  # TODO: Batch calls or cache duplicates (simple LRU).

    async def _get_llm_response(self, prompt: str, model: str = "gpt-4o-mini") -> str:
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

    

    async def upsert(self, rec: MemRecord) -> None:
        """
        Adds or updates a memory in the ChromaDB collection and its metadata in SQLite.
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        embedding = await self._get_embedding(rec.text)
        
        # Update ChromaDB
        collection.upsert(
            ids=[rec.id],
            embeddings=[embedding],
            documents=[rec.text],
            metadatas=[rec.meta],
        )

        # Update SQLite metadata
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO memory_metadata (memory_id, importance, access_count, last_accessed_timestamp) VALUES (?, ?, ?, ?)",
            (rec.id, rec.salience if rec.salience is not None else 0.0, 1, datetime.datetime.now().isoformat()),
        )
        self.conn.commit()
        self._log_history(rec.id, "UPSERT", rec.model_dump_json())

    

    async def search(self, query: str, mem_types: list[str], k: int = 8) -> List[MemRecord]:
        """
        Searches for memories in the ChromaDB collection based on a query.
        """
        collection = self.client.get_or_create_collection(self.collection_name)
        embedding = await self._get_embedding(query)
        results = collection.query(query_embeddings=[embedding], n_results=k)

        mem_records = []
        if results and results["documents"]:
            for i in range(len(results["ids"])):
                # Assuming metadata contains memory_type and sub_type, and document is text
                mem_records.append(MemRecord(
                    id=results["ids"][i],
                    memory_type=results["metadatas"][i].get("memory_type", "unknown"),
                    sub_type=results["metadatas"][i].get("sub_type"),
                    text=results["documents"][i],
                    embedding=results["embeddings"][i],
                    meta=results["metadatas"][i],
                    ts=datetime.datetime.now().timestamp(), # Placeholder, actual timestamp should be stored in metadata
                    salience=0.0 # Placeholder, salience should be stored in metadata
                ))
        return mem_records

    async def stream_conversation(self, conv_id: str):
        pass

    async def log_workflow_step(self, workflow_id: str, step: dict):
        pass

    async def entity_facts(self, entity_id: str, k: int = 10):
        pass

    async def match_tools(self, msg: str, k: int = 4) -> List[ToolRecord]:
        pass

    async def persona(self, persona_id: str) -> Persona:
        pass

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
            self.client.delete_collection(self.collection_name)
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
            ).total_seconds() / 3600  # Time in hours
            return importance * (1 + access_count) * (1 / (1 + recency))
        return 0.0

    
