"""Schema validation and contracts."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError


class FieldSchema(BaseModel):
    """Schema definition for a single field."""

    name: str = Field(..., description="Field name")
    type: str = Field(..., description="Field type (str, int, float, bool, dict, list)")
    required: bool = Field(default=True, description="Whether field is required")
    description: Optional[str] = Field(None, description="Field description")
    default: Optional[Any] = Field(None, description="Default value")
    enum: Optional[List[Any]] = Field(None, description="Allowed enum values")
    min_value: Optional[float] = Field(None, description="Minimum value (for numeric types)")
    max_value: Optional[float] = Field(None, description="Maximum value (for numeric types)")
    min_length: Optional[int] = Field(None, description="Minimum length (for string/list types)")
    max_length: Optional[int] = Field(None, description="Maximum length (for string/list types)")


class SchemaContract(BaseModel):
    """Schema contract for input/output validation."""

    name: str = Field(..., description="Schema name")
    version: str = Field(default="1.0.0", description="Schema version")
    fields: List[FieldSchema] = Field(default_factory=list, description="List of field schemas")
    strict: bool = Field(default=False, description="If True, reject extra fields")

    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate data against schema.

        Args:
            data: Data dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check for extra fields if strict
        if self.strict:
            allowed_fields = {field.name for field in self.fields}
            extra_fields = set(data.keys()) - allowed_fields
            if extra_fields:
                errors.append(f"Extra fields not allowed: {', '.join(extra_fields)}")

        # Validate each field
        for field_schema in self.fields:
            field_name = field_schema.name
            field_value = data.get(field_name)

            # Check required
            if field_schema.required and field_name not in data:
                errors.append(f"Required field '{field_name}' is missing")
                continue

            # Use default if not provided
            if field_value is None and field_schema.default is not None:
                field_value = field_schema.default

            if field_value is None:
                continue

            # Type validation
            expected_type = field_schema.type
            actual_type = type(field_value).__name__

            type_map = {
                "str": "str",
                "int": "int",
                "float": "float",
                "bool": "bool",
                "dict": "dict",
                "list": "list",
            }

            if expected_type not in type_map:
                errors.append(f"Unknown field type '{expected_type}' for '{field_name}'")
                continue

            if actual_type != type_map[expected_type]:
                # Try to coerce
                try:
                    if expected_type == "int":
                        field_value = int(field_value)
                    elif expected_type == "float":
                        field_value = float(field_value)
                    elif expected_type == "bool":
                        field_value = bool(field_value)
                    elif expected_type == "str":
                        field_value = str(field_value)
                    else:
                        errors.append(
                            f"Field '{field_name}' has wrong type: expected {expected_type}, got {actual_type}"
                        )
                        continue
                except (ValueError, TypeError):
                    errors.append(
                        f"Field '{field_name}' cannot be coerced to {expected_type} from {actual_type}"
                    )
                    continue

            # Enum validation
            if field_schema.enum and field_value not in field_schema.enum:
                errors.append(
                    f"Field '{field_name}' value '{field_value}' not in allowed values: {field_schema.enum}"
                )

            # Numeric range validation
            if expected_type in ("int", "float"):
                if field_schema.min_value is not None and field_value < field_schema.min_value:
                    errors.append(
                        f"Field '{field_name}' value {field_value} is less than minimum {field_schema.min_value}"
                    )
                if field_schema.max_value is not None and field_value > field_schema.max_value:
                    errors.append(
                        f"Field '{field_name}' value {field_value} is greater than maximum {field_schema.max_value}"
                    )

            # Length validation
            if expected_type in ("str", "list"):
                length = len(field_value) if field_value else 0
                if field_schema.min_length is not None and length < field_schema.min_length:
                    errors.append(
                        f"Field '{field_name}' length {length} is less than minimum {field_schema.min_length}"
                    )
                if field_schema.max_length is not None and length > field_schema.max_length:
                    errors.append(
                        f"Field '{field_name}' length {length} is greater than maximum {field_schema.max_length}"
                    )

        return len(errors) == 0, errors


def validate_with_schema(data: Dict[str, Any], schema: SchemaContract) -> tuple[bool, List[str]]:
    """
    Validate data with schema contract.

    Args:
        data: Data to validate
        schema: Schema contract

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    return schema.validate(data)


def infer_schema(data: Dict[str, Any], name: str = "inferred") -> SchemaContract:
    """
    Infer schema from sample data.

    Args:
        data: Sample data dictionary
        name: Schema name

    Returns:
        Inferred SchemaContract
    """
    fields = []

    for key, value in data.items():
        field_type = type(value).__name__

        # Map Python types to schema types
        type_mapping = {
            "str": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "dict": "dict",
            "list": "list",
        }

        schema_type = type_mapping.get(field_type, "str")

        field = FieldSchema(
            name=key,
            type=schema_type,
            required=True,
        )

        # Add length constraints for strings/lists
        if isinstance(value, str):
            field.min_length = len(value)  # Use actual length as min
            field.max_length = len(value)  # Use actual length as max
        elif isinstance(value, list):
            field.min_length = len(value)

        fields.append(field)

    return SchemaContract(name=name, fields=fields)
