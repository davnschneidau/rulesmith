"""Tests for schema validation."""

import pytest

from rulesmith.io.schemas import FieldSchema, SchemaContract, infer_schema, validate_with_schema


class TestSchemaContract:
    """Test schema contracts."""

    def test_schema_validation(self):
        """Test schema validation."""
        schema = SchemaContract(
            name="test_schema",
            fields=[
                FieldSchema(name="age", type="int", required=True, min_value=0, max_value=150),
                FieldSchema(name="name", type="str", required=True, min_length=1),
            ],
        )

        # Valid data
        is_valid, errors = schema.validate({"age": 30, "name": "John"})
        assert is_valid is True

        # Invalid data
        is_valid, errors = schema.validate({"age": -5})
        assert is_valid is False
        assert len(errors) > 0

    def test_schema_strict_mode(self):
        """Test strict mode."""
        schema = SchemaContract(
            name="test",
            fields=[FieldSchema(name="allowed", type="str")],
            strict=True,
        )

        is_valid, errors = schema.validate({"allowed": "value", "extra": "field"})
        assert is_valid is False
        assert any("Extra fields" in err for err in errors)

    def test_infer_schema(self):
        """Test schema inference."""
        data = {
            "name": "test",
            "age": 30,
            "active": True,
        }

        schema = infer_schema(data, name="inferred")
        assert len(schema.fields) == 3
        assert any(f.name == "name" and f.type == "str" for f in schema.fields)

