import asyncio
import datetime
from .base import BaseMemory

class Memory(BaseMemory):
    def __init__(self, db_path=":memory:"):
        super().__init__(db_path)

    def add(self, data, user_id=None, agent_id=None, run_id=None, infer=True):
        if not infer:
            self._add_memory(data, user_id, agent_id, run_id)
        else:
            self._infer_and_add(data, user_id, agent_id, run_id)

    def _add_memory(self, data, user_id, agent_id, run_id):
        memory_id = self._add_memory_to_collection(data, user_id, agent_id, run_id)
        self._log_history(memory_id, "ADD", data)
        importance_prompt = f"""Rate the importance of this memory on a scale from 0 to 1. 
        Memory: {data}
        Importance:"""
        importance_text = self._get_llm_response(importance_prompt).strip()
        importance = self._parse_importance(importance_text)
        self._update_metadata(memory_id, importance=importance)

    def _infer_and_add(self, data, user_id, agent_id, run_id):
        fact_extraction_prompt = f"""Extract key facts from the following text. Return them as a comma-separated list.
        Text: {data}
        Facts:"""
        facts = self._get_llm_response(fact_extraction_prompt).strip().split(", ")

        related_memories = []
        for fact in facts:
            search_results = self.search(fact, user_id, agent_id, run_id)
            related_memories.extend(search_results)
        
        unique_related_memories = {mem["id"]: mem for mem in related_memories}.values()

        decision_prompt = f"""Given the new information: "{data}"
        And existing related memories: {unique_related_memories}
        Decide whether to ADD this new information as a new memory, UPDATE an existing memory, or DELETE an existing memory if it's outdated.
        Respond with 'ADD', 'UPDATE:<memory_id>', or 'DELETE:<memory_id>'. If UPDATE, specify the memory_id to update. If DELETE, specify the memory_id to delete.
        Decision:"""
        decision = self._get_llm_response(decision_prompt).strip()

        if decision == "ADD":
            self._add_memory(data, user_id, agent_id, run_id)
        elif decision.startswith("UPDATE:"):
            memory_id_to_update = decision.split(":")[1]
            self.update(memory_id_to_update, data)
        elif decision.startswith("DELETE:"):
            memory_id_to_delete = decision.split(":")[1]
            self.delete(memory_id_to_delete)
        else:
            print(f"LLM made an unrecognised decision: {decision}. Defaulting to ADD.")
            self._add_memory(data, user_id, agent_id, run_id)

    def search(self, query, user_id=None, agent_id=None, run_id=None, threshold=0.8):
        query_embedding = self._get_embedding(query)
        where_clause = {}
        if user_id:
            where_clause["user_id"] = user_id
        if agent_id:
            where_clause["agent_id"] = agent_id
        if run_id:
            where_clause["run_id"] = run_id

        collection = self.client.get_or_create_collection("mem0")
        if where_clause:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                where=where_clause,
            )
        else:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
            )
        filtered_results = []
        if results and results["distances"]:
            for i, distance in enumerate(results["distances"][0]):
                if distance < threshold:
                    filtered_results.append({
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distance
                    })
        return filtered_results

    def get(self, memory_id):
        result = self._get_memory_from_collection(memory_id)
        if result:
            cursor = self.conn.cursor()
            cursor.execute("SELECT access_count FROM memory_metadata WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()
            if row:
                access_count = row[0] + 1
                self._update_metadata(memory_id, access_count=access_count)
            return result
        return None

    def get_all(self, user_id=None, agent_id=None, run_id=None):
        return self._get_all_memories_from_collection(user_id, agent_id, run_id)

    def update(self, memory_id, data):
        self._update_memory_in_collection(memory_id, data)
        self._log_history(memory_id, "UPDATE", data)

    def delete(self, memory_id):
        self._delete_memory_from_collection(memory_id)
        self._log_history(memory_id, "DELETE", "")

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        self._delete_all_memories_from_collection(user_id, agent_id, run_id)

    def decay_memories(self, threshold=0.5):
        super().decay_memories(threshold)

class AsyncMemory(BaseMemory):
    def __init__(self, db_path=":memory:"):
        super().__init__(db_path)

    async def _get_embedding(self, text):
        client = openai.AsyncOpenAI()
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    async def _get_llm_response(self, prompt, model="gpt-4o-mini"):
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def add(self, data, user_id=None, agent_id=None, run_id=None, infer=True):
        if not infer:
            await self._add_memory(data, user_id, agent_id, run_id)
        else:
            await self._infer_and_add(data, user_id, agent_id, run_id)

    async def _add_memory(self, data, user_id, agent_id, run_id):
        memory_id = await asyncio.to_thread(self._add_memory_to_collection, data, user_id, agent_id, run_id)
        await asyncio.to_thread(self._log_history, memory_id, "ADD", data)
        importance_prompt = f"""Rate the importance of this memory on a scale from 0 to 1. 
        Memory: {data}
        Importance:"""
        importance_text = (await self._get_llm_response(importance_prompt)).strip()
        importance = self._parse_importance(importance_text)
        await asyncio.to_thread(self._update_metadata, memory_id, importance=importance)

    async def _infer_and_add(self, data, user_id, agent_id, run_id):
        fact_extraction_prompt = f"""Extract key facts from the following text. Return them as a comma-separated list.
        Text: {data}
        Facts:"""
        facts = (await self._get_llm_response(fact_extraction_prompt)).strip().split(", ")

        related_memories = []
        for fact in facts:
            search_results = await self.search(fact, user_id, agent_id, run_id)
            related_memories.extend(search_results)
        
        unique_related_memories = {mem["id"]: mem for mem in related_memories}.values()

        decision_prompt = f"""Given the new information: "{data}"
        And existing related memories: {unique_related_memories}
        Decide whether to ADD this new information as a new memory, UPDATE an existing memory, or DELETE an existing memory if it's outdated.
        Respond with 'ADD', 'UPDATE:<memory_id>', or 'DELETE:<memory_id>'. If UPDATE, specify the memory_id to update. If DELETE, specify the memory_id to delete.
        Decision:"""
        decision = (await self._get_llm_response(decision_prompt)).strip()

        if decision == "ADD":
            await self._add_memory(data, user_id, agent_id, run_id)
        elif decision.startswith("UPDATE:"):
            memory_id_to_update = decision.split(":")[1]
            await self.update(memory_id_to_update, data)
        elif decision.startswith("DELETE:"):
            memory_id_to_delete = decision.split(":")[1]
            await self.delete(memory_id_to_delete)
        else:
            print(f"LLM made an unrecognised decision: {decision}. Defaulting to ADD.")
            await self._add_memory(data, user_id, agent_id, run_id)

    async def search(self, query, user_id=None, agent_id=None, run_id=None, threshold=0.8):
        query_embedding = await self._get_embedding(query)
        where_clause = {}
        if user_id:
            where_clause["user_id"] = user_id
        if agent_id:
            where_clause["agent_id"] = agent_id
        if run_id:
            where_clause["run_id"] = run_id

        collection = await asyncio.to_thread(self.client.get_or_create_collection, "mem0")
        results = await asyncio.to_thread(collection.query,
            query_embeddings=[query_embedding],
            n_results=10,
            where=where_clause if where_clause else None,
        )
        filtered_results = []
        if results and results["distances"]:
            for i, distance in enumerate(results["distances"][0]):
                if distance < threshold:
                    filtered_results.append({
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distance
                    })
        return filtered_results

    async def get(self, memory_id):
        result = await asyncio.to_thread(self._get_memory_from_collection, memory_id)
        if result:
            cursor = self.conn.cursor()
            cursor.execute("SELECT access_count FROM memory_metadata WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()
            if row:
                access_count = row[0] + 1
                await asyncio.to_thread(self._update_metadata, memory_id, access_count=access_count)
            return result
        return None

    async def get_all(self, user_id=None, agent_id=None, run_id=None):
        return await asyncio.to_thread(self._get_all_memories_from_collection, user_id, agent_id, run_id)

    async def update(self, memory_id, data):
        await asyncio.to_thread(self._update_memory_in_collection, memory_id, data)
        await asyncio.to_thread(self._log_history, memory_id, "UPDATE", data)

    async def delete(self, memory_id):
        await asyncio.to_thread(self._delete_memory_from_collection, memory_id)
        await asyncio.to_thread(self._log_history, memory_id, "DELETE", "")

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        await asyncio.to_thread(self._delete_all_memories_from_collection, user_id, agent_id, run_id)

    async def decay_memories(self, threshold=0.5):
        await asyncio.to_thread(super().decay_memories, threshold)

