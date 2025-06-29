from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class MemRecord(BaseModel):
    id: str
    memory_type: str
    sub_type: Optional[str] = None
    text: str
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any]
    ts: float
    salience: Optional[float] = None

class ToolRecord(BaseModel):
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]

class Persona(BaseModel):
    id: str
    name: str
    description: str
    attributes: Dict[str, Any]
