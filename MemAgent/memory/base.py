import chromadb
import sqlite3
import uuid
import datetime
import re
import openai

class BaseMemory:
    def __init__(self, db_path=":memory:"):
        self.client = chromadb.Client()
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
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

    def _parse_importance(self, text):
        match = re.search(r"\d+\.?\d*", text)
        if match:
            return float(match.group())
        return 0.0

    def _get_embedding(self, text):
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def _get_llm_response(self, prompt, model="gpt-4o-mini"):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    def _update_metadata(self, memory_id, importance=None, access_count=None):
        cursor = self.conn.cursor()
        if importance is not None:
            cursor.execute("INSERT OR REPLACE INTO memory_metadata (memory_id, importance, access_count, last_accessed_timestamp) VALUES (?, ?, ?, ?)",
                           (memory_id, importance, 1, datetime.datetime.now()))
        if access_count is not None:
            cursor.execute("UPDATE memory_metadata SET access_count = ?, last_accessed_timestamp = ? WHERE memory_id = ?",
                           (access_count, datetime.datetime.now(), memory_id))
        self.conn.commit()

    def _log_history(self, memory_id, action, data):
        cursor = self.conn.cursor()
        history_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO memory_history (id, memory_id, action, data) VALUES (?, ?, ?, ?)",
                       (history_id, memory_id, action, data))
        self.conn.commit()

    def _add_memory_to_collection(self, data, user_id, agent_id, run_id):
        collection = self.client.get_or_create_collection("mem0")
        memory_id = str(uuid.uuid4())
        embedding = self._get_embedding(data)
        collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[data],
            metadatas=[{k: v for k, v in {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}.items() if v is not None}]
        )
        return memory_id

    def _update_memory_in_collection(self, memory_id, data):
        collection = self.client.get_or_create_collection("mem0")
        embedding = self._get_embedding(data)
        collection.update(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[data]
        )

    def _delete_memory_from_collection(self, memory_id):
        collection = self.client.get_or_create_collection("mem0")
        collection.delete(ids=[memory_id])

    def _get_memory_from_collection(self, memory_id):
        collection = self.client.get_or_create_collection("mem0")
        result = collection.get(ids=[memory_id])
        if result and result["documents"]:
            return {
                "id": result["ids"][0],
                "document": result["documents"][0],
                "metadata": result["metadatas"][0]
            }
        return None

    def _get_all_memories_from_collection(self, user_id=None, agent_id=None, run_id=None):
        collection = self.client.get_or_create_collection("mem0")
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
                all_memories.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i]
                })
        return all_memories

    def _delete_all_memories_from_collection(self, user_id=None, agent_id=None, run_id=None):
        collection = self.client.get_or_create_collection("mem0")
        where_clause = {}
        if user_id:
            where_clause["user_id"] = user_id
        if agent_id:
            where_clause["agent_id"] = agent_id
        if run_id:
            where_clause["run_id"] = run_id
        
        collection.delete(where=where_clause if where_clause else None)

    def history(self, memory_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT action, data, timestamp FROM memory_history WHERE memory_id = ? ORDER BY timestamp ASC", (memory_id,))
        return cursor.fetchall()

    def reset(self):
        try:
            self.client.delete_collection("mem0")
        except chromadb.errors.NotFoundError:
            pass
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS memory_history")
        cursor.execute("DROP TABLE IF EXISTS memory_metadata")
        self.conn.commit()
        self._create_tables()

    def calculate_strength(self, memory_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT importance, access_count, last_accessed_timestamp FROM memory_metadata WHERE memory_id = ?", (memory_id,))
        row = cursor.fetchone()
        if row:
            importance, access_count, last_accessed_timestamp = row
            recency = (datetime.datetime.now() - datetime.datetime.fromisoformat(last_accessed_timestamp)).total_seconds() / 3600  # Time in hours
            return importance * (1 + access_count) * (1 / (1 + recency))
        return 0

    def decay_memories(self, threshold=0.5):
        all_memories = self._get_all_memories_from_collection()
        for memory in all_memories:
            strength = self.calculate_strength(memory["id"])
            if strength < threshold:
                self._delete_memory_from_collection(memory["id"])
                self._log_history(memory["id"], "DECAY", "")
