import json
from datetime import datetime, date
from uuid import UUID
from typing import Any
from pydantic import BaseModel

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that extends support for datetimes, UUIDs, and Pydantic models.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def to_json(obj: Any) -> str:
    """
    Serializes any object to a JSON-formatted string using the CustomJSONEncoder.
    """
    return json.dumps(obj, cls=CustomJSONEncoder)

def from_json(json_str: str) -> Any:
    """
    Deserializes a JSON-formatted string back into python objects.
    """
    if not json_str:
        return None
    return json.loads(json_str)
