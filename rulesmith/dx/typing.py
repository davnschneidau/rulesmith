"""Strict typing support for rulebooks."""

import inspect
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints, get_origin, get_args
from dataclasses import dataclass
from inspect import signature

from pydantic import BaseModel, create_model, ValidationError


@dataclass
class TypeInfo:
    """Type information for a field."""
    
    name: str
    type: Type[Any]
    required: bool = True
    default: Any = None


class TypeValidator:
    """Validates inputs and outputs against type hints."""
    
    def __init__(self, strict: bool = True):
        """
        Initialize type validator.
        
        Args:
            strict: If True, enforce strict type checking (no coercion)
        """
        self.strict = strict
    
    def validate_inputs(self, func: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate inputs against function signature.
        
        Args:
            func: Function to validate against
            inputs: Input dictionary
        
        Returns:
            Validated inputs (may be coerced if not strict)
        
        Raises:
            TypeError: If validation fails
        """
        sig = signature(func)
        type_hints = get_type_hints(func)
        
        validated = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            
            param_type = type_hints.get(param_name)
            
            if param_name not in inputs:
                if param.default != inspect.Parameter.empty:
                    # Has default, use it
                    validated[param_name] = param.default
                    continue
                elif not param.required:
                    # Not required, skip
                    continue
                else:
                    raise TypeError(f"Missing required parameter: {param_name}")
            
            value = inputs[param_name]
            
            if param_type:
                validated[param_name] = self._validate_type(value, param_type, param_name)
            else:
                validated[param_name] = value
        
        return validated
    
    def validate_outputs(self, func: Any, outputs: Dict[str, Any], expected_outputs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate outputs against return type.
        
        Args:
            func: Function that produced outputs
            outputs: Output dictionary
            expected_outputs: Optional list of expected output field names
        
        Returns:
            Validated outputs
        
        Raises:
            TypeError: If validation fails
        """
        type_hints = get_type_hints(func)
        return_type = type_hints.get("return")
        
        if return_type:
            # Check if return type is Dict
            if get_origin(return_type) == dict:
                # Validate dictionary structure
                args = get_args(return_type)
                if args:
                    # Dict[str, Any] or similar
                    key_type, value_type = args
                    if key_type != str:
                        raise TypeError(f"Return type must be Dict[str, ...], got Dict[{key_type}, ...]")
                    
                    # Validate values
                    validated = {}
                    for key, value in outputs.items():
                        validated[key] = self._validate_type(value, value_type, f"output.{key}")
                    return validated
            
            # For other return types, validate the entire output
            if isinstance(outputs, dict) and len(outputs) == 1:
                # Single value output
                value = list(outputs.values())[0]
                return {list(outputs.keys())[0]: self._validate_type(value, return_type, "return")}
        
        return outputs
    
    def _validate_type(self, value: Any, expected_type: Type[Any], field_name: str) -> Any:
        """
        Validate a value against an expected type.
        
        Args:
            value: Value to validate
            expected_type: Expected type
            field_name: Field name for error messages
        
        Returns:
            Validated value (may be coerced if not strict)
        
        Raises:
            TypeError: If validation fails
        """
        # Handle Optional types
        if get_origin(expected_type) is Union:
            args = get_args(expected_type)
            # Check for None in union (Optional)
            if type(None) in args:
                non_none_types = [t for t in args if t is not type(None)]
                if not non_none_types:
                    # Union[None, None] - weird but allow it
                    return value
                expected_type = Union[tuple(non_none_types)]  # Try each type
        
        # Handle generic types
        origin = get_origin(expected_type)
        if origin:
            if origin == list:
                args = get_args(expected_type)
                if args:
                    item_type = args[0]
                    if not isinstance(value, list):
                        if self.strict:
                            raise TypeError(f"{field_name}: expected list, got {type(value).__name__}")
                        value = [value]  # Coerce to list
                    return [self._validate_type(item, item_type, f"{field_name}[{i}]") for i, item in enumerate(value)]
                return value
            elif origin == dict:
                args = get_args(expected_type)
                if args:
                    # Dict[str, int] or similar
                    if not isinstance(value, dict):
                        if self.strict:
                            raise TypeError(f"{field_name}: expected dict, got {type(value).__name__}")
                        raise TypeError(f"Cannot coerce {type(value).__name__} to dict")
                    
                    key_type, value_type = args
                    validated = {}
                    for k, v in value.items():
                        validated_key = self._validate_type(k, key_type, f"{field_name}.key")
                        validated_value = self._validate_type(v, value_type, f"{field_name}.value")
                        validated[validated_key] = validated_value
                    return validated
                return value
        
        # Check if value matches type
        if isinstance(value, expected_type):
            return value
        
        # Try type coercion if not strict
        if not self.strict:
            try:
                if expected_type == int:
                    return int(value)
                elif expected_type == float:
                    return float(value)
                elif expected_type == str:
                    return str(value)
                elif expected_type == bool:
                    return bool(value)
            except (ValueError, TypeError):
                pass
        
        raise TypeError(
            f"{field_name}: expected {expected_type.__name__}, got {type(value).__name__}"
        )


def create_input_schema(func: Any) -> Type[BaseModel]:
    """
    Create a Pydantic model from function signature.
    
    Args:
        func: Function to create schema for
    
    Returns:
        Pydantic model class
    """
    sig = signature(func)
    type_hints = get_type_hints(func)
    
    fields = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        
        param_type = type_hints.get(param_name, Any)
        
        # Handle Optional
        if get_origin(param_type) is Union:
            args = get_args(param_type)
            if type(None) in args:
                non_none_types = [t for t in args if t is not type(None)]
                if non_none_types:
                    param_type = Optional[non_none_types[0]]
                else:
                    param_type = Optional[Any]
        
        # Determine if required
        if param.default != inspect.Parameter.empty:
            fields[param_name] = (param_type, param.default)
        else:
            fields[param_name] = (param_type, ...)
    
    return create_model(f"{func.__name__}_Input", **fields)


def create_output_schema(func: Any, output_fields: Optional[List[str]] = None) -> Type[BaseModel]:
    """
    Create a Pydantic model for function outputs.
    
    Args:
        func: Function to create schema for
        output_fields: Optional list of output field names
    
    Returns:
        Pydantic model class
    """
    type_hints = get_type_hints(func)
    return_type = type_hints.get("return", Dict[str, Any])
    
    # If return type is Dict[str, Any], create fields from output_fields
    if output_fields:
        fields = {field: (Any, ...) for field in output_fields}
    else:
        # Infer from return type
        if get_origin(return_type) == dict:
            args = get_args(return_type)
            if args and len(args) >= 2:
                # Dict[str, int] or similar
                value_type = args[1]
                fields = {"result": (value_type, ...)}
            else:
                fields = {"result": (Any, ...)}
        else:
            fields = {"result": (return_type, ...)}
    
    return create_model(f"{func.__name__}_Output", **fields)


# Global type validator
type_validator = TypeValidator(strict=True)

