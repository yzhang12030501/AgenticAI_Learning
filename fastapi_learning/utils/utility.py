import inspect
from typing import get_type_hints

def function_to_json_schema(func):
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    properties = {}
    required = []
    type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean"
    }
    for name, param in sig.parameters.items():
        py_type = hints.get(name, str)
        json_type = type_map.get(py_type, "string")
        properties[name] = {"type": json_type}
        required.append(name)
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.strip() if func.__doc__ else "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }