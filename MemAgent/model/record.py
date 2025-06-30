from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np


class MemRecord(BaseModel):
    """Memory record with all required fields according to API specification."""
    
    # Core fields
    id: str
    content: str
    memory_type: str
    sub_type: Optional[str] = None
    
    # Embedding and importance
    embedding: Optional[np.ndarray] = None
    importance: int = 5  # 1-10 scale
    
    # Metadata
    metadata: Dict[str, Any] = {}
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime
    
    # Usage tracking
    access_count: int = 0
    decay_rate: float = 0.0
    
    # Operation tracking
    operation: Optional[str] = None  # ADD, UPDATE, DELETE
    previous_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True  # Allow np.ndarray


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