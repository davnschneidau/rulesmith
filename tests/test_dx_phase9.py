"""Tests for Phase 9: Developer Experience features."""

import pytest
from pathlib import Path
import tempfile
import json

from rulesmith.dx.errors import (
    RulebookError,
    RulebookErrorHandler,
    ValidationError,
    MappingError,
    error_handler,
)
from rulesmith.dx.typing import (
    TypeValidator,
    create_input_schema,
    create_output_schema,
    type_validator,
)
from rulesmith.dx.auto_mapping import AutoMapper, auto_mapper
from rulesmith.dx.single_file import RulebookDSL, load_rsm_file, save_rsm_file
from rulesmith.dag.graph import Rulebook
from rulesmith.dag.decorators import rule


class TestErrorErgonomics:
    """Test error handling and ergonomics."""
    
    def test_error_handler(self):
        """Test error handler with context."""
        handler = RulebookErrorHandler()
        
        error = ValueError("Invalid input")
        error_info = handler.handle_error(
            error=error,
            node_name="test_node",
            rule_name="test_rule",
            input_data={"x": 5},
        )
        
        assert error_info.error_type == "ValueError"
        assert error_info.node_name == "test_node"
        assert error_info.rule_name == "test_rule"
        assert len(error_info.suggestions) > 0
    
    def test_validation_error(self):
        """Test validation error with field errors."""
        error = ValidationError(
            message="Validation failed",
            field_errors={"age": ["Must be >= 18"], "income": ["Must be > 0"]},
        )
        
        assert error.error_type == "ValidationError"
        assert "age" in error.field_errors
        assert len(error.field_errors["age"]) > 0
    
    def test_mapping_error(self):
        """Test mapping error."""
        error = MappingError(
            message="Missing fields",
            source_node="node1",
            target_node="node2",
            missing_fields=["age", "income"],
        )
        
        assert error.error_type == "MappingError"
        assert error.source_node == "node1"
        assert error.target_node == "node2"
        assert "age" in error.missing_fields


class TestStrictTyping:
    """Test strict typing support."""
    
    def test_type_validator(self):
        """Test type validator."""
        validator = TypeValidator(strict=True)
        
        def test_func(age: int, name: str) -> dict:
            return {"result": f"{name} is {age}"}
        
        # Valid inputs
        inputs = {"age": 25, "name": "Alice"}
        validated = validator.validate_inputs(test_func, inputs)
        assert validated["age"] == 25
        assert validated["name"] == "Alice"
        
        # Invalid type
        with pytest.raises(TypeError):
            validator.validate_inputs(test_func, {"age": "25", "name": "Alice"})
    
    def test_type_validator_coercion(self):
        """Test type validator with coercion."""
        validator = TypeValidator(strict=False)
        
        def test_func(age: int) -> dict:
            return {"age": age}
        
        # Coerce string to int
        inputs = {"age": "25"}
        validated = validator.validate_inputs(test_func, inputs)
        assert validated["age"] == 25
    
    def test_create_input_schema(self):
        """Test creating Pydantic input schema."""
        def test_func(age: int, name: str = "Default") -> dict:
            return {"result": f"{name} is {age}"}
        
        schema = create_input_schema(test_func)
        
        # Valid data
        instance = schema(age=25, name="Alice")
        assert instance.age == 25
        assert instance.name == "Alice"
        
        # Invalid data
        with pytest.raises(Exception):
            schema(age="25")  # Wrong type


class TestAutoMapping:
    """Test automatic field mapping."""
    
    def test_infer_mapping(self):
        """Test inferring field mappings."""
        mapper = AutoMapper(strict=False)
        
        source_outputs = {"age": 25, "income": 50000, "name": "Alice"}
        target_inputs = ["age", "income"]
        
        mapping = mapper.infer_mapping(source_outputs, target_inputs)
        
        assert mapping["age"] == "age"  # Exact match
        assert mapping["income"] == "income"  # Exact match
    
    def test_infer_mapping_case_insensitive(self):
        """Test case-insensitive mapping."""
        mapper = AutoMapper(strict=False)
        
        source_outputs = {"UserAge": 25, "UserIncome": 50000}
        target_inputs = ["age", "income"]
        
        mapping = mapper.infer_mapping(source_outputs, target_inputs)
        
        # Should find matches (case-insensitive)
        assert len(mapping) > 0
    
    def test_validate_mapping(self):
        """Test mapping validation."""
        mapper = AutoMapper()
        
        source_outputs = {"age": 25, "income": 50000}
        target_inputs = ["age", "income"]
        mapping = {"age": "age", "income": "income"}
        
        is_valid, missing = mapper.validate_mapping(source_outputs, target_inputs, mapping)
        
        assert is_valid is True
        assert len(missing) == 0
    
    def test_validate_mapping_missing(self):
        """Test mapping validation with missing fields."""
        mapper = AutoMapper()
        
        source_outputs = {"age": 25}
        target_inputs = ["age", "income"]
        mapping = {"age": "age"}
        
        is_valid, missing = mapper.validate_mapping(source_outputs, target_inputs, mapping)
        
        assert is_valid is False
        assert "income" in missing


class TestSingleFileRulebooks:
    """Test single-file rulebook format."""
    
    def test_rulebook_dsl(self):
        """Test RulebookDSL."""
        dsl = RulebookDSL()
        
        @dsl.rule(name="check_age", inputs=["age"], outputs=["eligible"])
        def check_age(age: int) -> dict:
            return {"eligible": age >= 18}
        
        dsl.rulebook(name="test_rulebook", version="1.0.0")
        dsl.add_node(kind="rule", name="check_age", rule_ref="check_age")
        
        rb = dsl.build()
        assert rb.name == "test_rulebook"
        assert rb.version == "1.0.0"
    
    def test_save_and_load_rsm_json(self):
        """Test saving and loading .rsm file (JSON)."""
        # Create a simple rulebook
        rb = Rulebook(name="test", version="1.0.0")
        
        @rule(name="test_rule", inputs=["x"], outputs=["y"])
        def test_rule(x: int) -> dict:
            return {"y": x * 2}
        
        rb.add_rule(test_rule, as_name="test_rule")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rsm', delete=False) as f:
            temp_path = f.name
        
        try:
            save_rsm_file(rb, temp_path, format="json")
            
            # Load back
            loaded_rb = load_rsm_file(temp_path)
            
            assert loaded_rb.name == "test"
            assert loaded_rb.version == "1.0.0"
        finally:
            Path(temp_path).unlink()


class TestDXIntegration:
    """Integration tests for DX features."""
    
    def test_rulebook_with_auto_mapping(self):
        """Test rulebook with auto-mapping."""
        rb = Rulebook(name="test", version="1.0.0")
        
        @rule(name="rule1", inputs=["age"], outputs=["eligible"])
        def rule1(age: int) -> dict:
            return {"eligible": age >= 18}
        
        @rule(name="rule2", inputs=["eligible"], outputs=["result"])
        def rule2(eligible: bool) -> dict:
            return {"result": "approved" if eligible else "rejected"}
        
        rb.add_rule(rule1, as_name="rule1")
        rb.add_rule(rule2, as_name="rule2")
        
        # Connect with auto-mapping
        rb.connect("rule1", "rule2", auto_map=True)
        
        # Execute
        result = rb.run({"age": 25}, return_decision_result=True)
        assert result.value["result"] == "approved"
    
    def test_error_handling_in_execution(self):
        """Test error handling in execution."""
        rb = Rulebook(name="test", version="1.0.0")
        
        @rule(name="error_rule", inputs=["x"])
        def error_rule(x: int) -> dict:
            if x < 0:
                raise ValueError("x must be >= 0")
            return {"result": x * 2}
        
        rb.add_rule(error_rule, as_name="error_rule")
        
        # Should handle error gracefully
        result = rb.run({"x": -1}, return_decision_result=True)
        assert len(result.warnings) > 0

