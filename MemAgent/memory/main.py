import asyncio
import datetime
from typing import Optional, List, Dict, Any
from .base import BaseMemory


class Memory(BaseMemory):
    """
    Synchronous memory management class, extending BaseMemory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """
        Initializes the Memory with a SQLite database path.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to an in-memory database.
        """
        super().__init__(db_path)

    def add(
        self, data: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None, infer: bool = True
    ) -> None:
        """
        Adds a new memory, with optional inference for intelligent decision-making.

        Args:
            data (str): The content of the memory.
            user_id (Optional[str]): The ID of the user associated with the memory.
            agent_id (Optional[str]): The ID of the agent associated with the memory.
            run_id (Optional[str]): The ID of the run associated with the memory.
            infer (bool): Whether to infer ADD/UPDATE/DELETE actions. Defaults to True.
        """
        if not infer:
            self._add_memory(data, user_id, agent_id, run_id)
        else:
            self._infer_and_add(data, user_id, agent_id, run_id)

    def _add_memory(
        self, data: str, user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
    ) -> None:
        """
        Internal method to add a memory to the collection and log its history.

        Args:
            data (str): The content of the memory.
            user_id (Optional[str]): The ID of the user associated with the memory.
            agent_id (Optional[str]): The ID of the agent associated with the memory.
            run_id (Optional[str]): The ID of the run associated with the memory.
        """
        memory_id = self._add_memory_to_collection(data, user_id, agent_id, run_id)
        self._log_history(memory_id, "ADD", data)
        importance_prompt = f"""Rate the importance of this memory on a scale from 0 to 1. 
        Memory: {data}
        Importance:"""
        importance_text = self._get_llm_response(importance_prompt).strip()
        importance = self._parse_importance(importance_text)
        self._update_metadata(memory_id, importance=importance)

    def _infer_and_add(
        self, data: str, user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
    ) -> None:
        """
        Infers the appropriate action (ADD, UPDATE, DELETE) based on new information and existing memories.

        Args:
            data (str): The new information.
            user_id (Optional[str]): The ID of the user associated with the memory.
            agent_id (Optional[str]): The ID of the agent associated with the memory.
            run_id (Optional[str]): The ID of the run associated with the memory.
        """
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
            # TODO: Request response_format={"type":"json_object"} and validate with pydantic.
            memory_id_to_update = decision.split(":")[1]
            self.update(memory_id_to_update, data)
        elif decision.startswith("DELETE:"):
            # TODO: Request response_format={"type":"json_object"} and validate with pydantic.
            memory_id_to_delete = decision.split(":")[1]
            self.delete(memory_id_to_delete)
        else:
            # TODO: Replace with proper logging (e.g., `logging.warning`).
            print(f"LLM made an unrecognised decision: {decision}. Defaulting to ADD.")
            self._add_memory(data, user_id, agent_id, run_id)

    def search(
        self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None, distance_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Searches for memories related to a query.

        Args:
            query (str): The search query.
            user_id (Optional[str]): Filter memories by user ID.
            agent_id (Optional[str]): Filter memories by agent ID.
            run_id (Optional[str]): Filter memories by run ID.
            threshold (float): The similarity threshold for search results. Defaults to 0.8.

        Returns:
            List[Dict[str, Any]]: A list of matching memories with their distances.
        """
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
                if distance < distance_threshold: # TODO: Decide on distance vs. similarity; expose similarity_cutoff to the caller.
                    filtered_results.append(
                        {
                            "id": results["ids"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": distance,
                        }
                    )
        return filtered_results

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific memory by its ID and updates its access count.

        Args:
            memory_id (str): The ID of the memory to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The memory details if found, otherwise None.
        """
        result = self._get_memory_from_collection(memory_id)
        if result:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT access_count FROM memory_metadata WHERE memory_id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row:
                access_count = row[0] + 1
                self._update_metadata(memory_id, access_count=access_count)
            return result
        return None

    def get_all(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all memories, optionally filtered by user, agent, or run ID.

        Args:
            user_id (Optional[str]): Filter memories by user ID.
            agent_id (Optional[str]): Filter memories by agent ID.
            run_id (Optional[str]): Filter memories by run ID.

        Returns:
            List[Dict[str, Any]]: A list of all matching memories.
        """
        return self._get_all_memories_from_collection(user_id, agent_id, run_id)

    def update(self, memory_id: str, data: str) -> None:
        """
        Updates an existing memory.

        Args:
            memory_id (str): The ID of the memory to update.
            data (str): The new content of the memory.
        """
        self._update_memory_in_collection(memory_id, data)
        self._log_history(memory_id, "UPDATE", data)

    def delete(self, memory_id: str) -> None:
        """
        Deletes a memory by its ID.

        Args:
            memory_id (str): The ID of the memory to delete.
        """
        self._delete_memory_from_collection(memory_id)
        self._log_history(memory_id, "DELETE", "")

    def delete_all(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> None:
        """
        Deletes all memories, optionally filtered by user, agent, or run ID.

        Args:
            user_id (Optional[str]): Filter memories by user ID.
            agent_id (Optional[str]): Filter memories by agent ID.
            run_id (Optional[str]): Filter memories by run ID.
        """
        self._delete_all_memories_from_collection(user_id, agent_id, run_id)

    def decay_memories(self, threshold: float = 0.5) -> None:
        """
        Decays memories with strength below a given threshold.

        Args:
            threshold (float): The strength threshold for decaying memories.
        """
        super().decay_memories(threshold) # TODO: Add Prometheus counters (`memagent_memory_decayed_total`, etc.).


class AsyncMemory(BaseMemory):
    """
    Asynchronous memory management class, extending BaseMemory.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        """
        Initializes the AsyncMemory with a SQLite database path.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to an in-memory database.
        """
        super().__init__(db_path)

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Asynchronously generates an embedding for the given text using OpenAI's embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        client = openai.AsyncOpenAI()
        response = await client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    async def _get_llm_response(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """
        Asynchronously gets a response from the OpenAI LLM.

        Args:
            prompt (str): The prompt to send to the LLM.
            model (str): The LLM model to use.

        Returns:
            str: The LLM's response.
        """
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def add(
        self, data: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None, infer: bool = True
    ) -> None:
        """
        Asynchronously adds a new memory, with optional inference for intelligent decision-making.

        Args:
            data (str): The content of the memory.
            user_id (Optional[str]): The ID of the user associated with the memory.
            agent_id (Optional[str]): The ID of the agent associated with the memory.
            run_id (Optional[str]): The ID of the run associated with the memory.
            infer (bool): Whether to infer ADD/UPDATE/DELETE actions. Defaults to True.
        """
        if not infer:
            await self._add_memory(data, user_id, agent_id, run_id)
        else:
            await self._infer_and_add(data, user_id, agent_id, run_id)

    async def _add_memory(
        self, data: str, user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
    ) -> None:
        """
        Internal asynchronous method to add a memory to the collection and log its history.

        Args:
            data (str): The content of the memory.
            user_id (Optional[str]): The ID of the user associated with the memory.
            agent_id (Optional[str]): The ID of the agent associated with the memory.
            run_id (Optional[str]): The ID of the run associated with the memory.
        """
        memory_id = await asyncio.to_thread(
            self._add_memory_to_collection, data, user_id, agent_id, run_id
        )
        await asyncio.to_thread(self._log_history, memory_id, "ADD", data)
        importance_prompt = f"""Rate the importance of this memory on a scale from 0 to 1. 
        Memory: {data}
        Importance:"""
        importance_text = (await self._get_llm_response(importance_prompt)).strip()
        importance = self._parse_importance(importance_text)
        await asyncio.to_thread(self._update_metadata, memory_id, importance=importance)

    async def _infer_and_add(
        self, data: str, user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]
    ) -> None:
        """
        Asynchronously infers the appropriate action (ADD, UPDATE, DELETE) based on new information and existing memories.

        Args:
            data (str): The new information.
            user_id (Optional[str]): The ID of the user associated with the memory.
            agent_id (Optional[str]): The ID of the agent associated with the memory.
            run_id (Optional[str]): The ID of the run associated with the memory.
        """
        fact_extraction_prompt = f"""Extract key facts from the following text. Return them as a comma-separated list.
        Text: {data}
        Facts:"""
        facts = (
            (await self._get_llm_response(fact_extraction_prompt)).strip().split(", ")
        )

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
            # TODO: Replace with proper logging
            print(f"LLM made an unrecognised decision: {decision}. Defaulting to ADD.")
            await self._add_memory(data, user_id, agent_id, run_id)

    async def search(
        self, query: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None, threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously searches for memories related to a query.

        Args:
            query (str): The search query.
            user_id (Optional[str]): Filter memories by user ID.
            agent_id (Optional[str]): Filter memories by agent ID.
            run_id (Optional[str]): Filter memories by run ID.
            threshold (float): The similarity threshold for search results. Defaults to 0.8.

        Returns:
            List[Dict[str, Any]]: A list of matching memories with their distances.
        """
        query_embedding = await self._get_embedding(query)
        where_clause = {}
        if user_id:
            where_clause["user_id"] = user_id
        if agent_id:
            where_clause["agent_id"] = agent_id
        if run_id:
            where_clause["run_id"] = run_id

        collection = await asyncio.to_thread(
            self.client.get_or_create_collection, "mem0"
        )
        results = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=10,
            where=where_clause if where_clause else None,
        )
        filtered_results = []
        if results and results["distances"]:
            for i, distance in enumerate(results["distances"][0]):
                if distance < distance_threshold: # TODO: Decide on distance vs. similarity; expose similarity_cutoff to the caller.
                    filtered_results.append(
                        {
                            "id": results["ids"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "distance": distance,
                        }
                    )
        return filtered_results

    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Asynchronously retrieves a specific memory by its ID and updates its access count.

        Args:
            memory_id (str): The ID of the memory to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The memory details if found, otherwise None.
        """
        result = await asyncio.to_thread(self._get_memory_from_collection, memory_id)
        if result:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT access_count FROM memory_metadata WHERE memory_id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row:
                access_count = row[0] + 1
                await asyncio.to_thread(
                    self._update_metadata, memory_id, access_count=access_count
                )
            return result
        return None

    async def get_all(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieves all memories, optionally filtered by user, agent, or run ID.

        Args:
            user_id (Optional[str]): Filter memories by user ID.
            agent_id (Optional[str]): Filter memories by agent ID.
            run_id (Optional[str]): Filter memories by run ID.

        Returns:
            List[Dict[str, Any]]: A list of all matching memories.
        """
        return await asyncio.to_thread(
            self._get_all_memories_from_collection, user_id, agent_id, run_id
        )

    async def update(self, memory_id: str, data: str) -> None:
        """
        Asynchronously updates an existing memory.

        Args:
            memory_id (str): The ID of the memory to update.
            data (str): The new content of the memory.
        """
        await asyncio.to_thread(self._update_memory_in_collection, memory_id, data)
        await asyncio.to_thread(self._log_history, memory_id, "UPDATE", data)

    async def delete(self, memory_id: str) -> None:
        """
        Asynchronously deletes a memory by its ID.

        Args:
            memory_id (str): The ID of the memory to delete.
        """
        await asyncio.to_thread(self._delete_memory_from_collection, memory_id)
        await asyncio.to_thread(self._log_history, memory_id, "DELETE", "")

    async def delete_all(
        self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None
    ) -> None:
        """
        Asynchronously deletes all memories, optionally filtered by user, agent, or run ID.

        Args:
            user_id (Optional[str]): Filter memories by user ID.
            agent_id (Optional[str]): Filter memories by agent ID.
            run_id (Optional[str]): Filter memories by run ID.
        """
        await asyncio.to_thread(
            self._delete_all_memories_from_collection, user_id, agent_id, run_id
        )

    async def decay_memories(self, threshold: float = 0.5) -> None:
        """
        Asynchronously decays memories with strength below a given threshold.

        Args:
            threshold (float): The strength threshold for decaying memories.
        """
        await asyncio.to_thread(super().decay_memories, threshold)
