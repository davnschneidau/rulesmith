"""Enhanced error handling and ergonomics."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import traceback


@dataclass
class RulebookError:
    """Enhanced error information for rulebook execution."""
    
    error_type: str
    message: str
    node_name: Optional[str] = None
    rule_name: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "node_name": self.node_name,
            "rule_name": self.rule_name,
            "input_data": self.input_data,
            "context": self.context,
            "stack_trace": self.stack_trace,
            "suggestions": self.suggestions,
        }
    
    def __str__(self) -> str:
        """Human-readable error message."""
        parts = [f"{self.error_type}: {self.message}"]
        
        if self.node_name:
            parts.append(f"Node: {self.node_name}")
        if self.rule_name:
            parts.append(f"Rule: {self.rule_name}")
        
        if self.suggestions:
            parts.append("\nSuggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  - {suggestion}")
        
        return "\n".join(parts)


class RulebookErrorHandler:
    """Enhanced error handler with context and suggestions."""
    
    def __init__(self, include_stack_trace: bool = True):
        """
        Initialize error handler.
        
        Args:
            include_stack_trace: Whether to include stack traces in errors
        """
        self.include_stack_trace = include_stack_trace
        self._error_context: List[Dict[str, Any]] = []
    
    def handle_error(
        self,
        error: Exception,
        node_name: Optional[str] = None,
        rule_name: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RulebookError:
        """
        Handle an error with enhanced context.
        
        Args:
            error: Exception that occurred
            node_name: Optional node name where error occurred
            rule_name: Optional rule name where error occurred
            input_data: Optional input data that caused the error
            context: Optional additional context
        
        Returns:
            RulebookError with suggestions
        """
        error_type = type(error).__name__
        message = str(error)
        
        # Get stack trace if requested
        stack_trace = None
        if self.include_stack_trace:
            stack_trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        
        # Generate suggestions based on error type
        suggestions = self._generate_suggestions(error, node_name, rule_name, input_data)
        
        return RulebookError(
            error_type=error_type,
            message=message,
            node_name=node_name,
            rule_name=rule_name,
            input_data=input_data,
            context=context or {},
            stack_trace=stack_trace,
            suggestions=suggestions,
        )
    
    def _generate_suggestions(
        self,
        error: Exception,
        node_name: Optional[str],
        rule_name: Optional[str],
        input_data: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Generate helpful suggestions based on error type and context."""
        suggestions = []
        error_type = type(error).__name__
        
        if error_type == "KeyError":
            key = str(error).strip("'")
            suggestions.append(f"Missing required field: '{key}'")
            if input_data:
                suggestions.append(f"Available fields: {list(input_data.keys())}")
        
        elif error_type == "TypeError":
            message = str(error)
            if "expected" in message.lower() and "got" in message.lower():
                suggestions.append("Check that input types match function signature")
                suggestions.append("Use type hints on rule functions for better error messages")
        
        elif error_type == "ValueError":
            suggestions.append("Check that input values are within expected ranges")
            if rule_name:
                suggestions.append(f"Review rule '{rule_name}' logic")
        
        elif error_type == "AttributeError":
            attr = str(error).strip("'")
            suggestions.append(f"Missing attribute or method: '{attr}'")
            if node_name:
                suggestions.append(f"Check node '{node_name}' configuration")
        
        elif error_type == "NameError":
            name = str(error).strip("'")
            suggestions.append(f"Undefined variable or function: '{name}'")
            suggestions.append("Check that all dependencies are imported")
        
        elif "Connection" in error_type or "Network" in error_type:
            suggestions.append("Check network connectivity")
            suggestions.append("Verify external service endpoints")
        
        elif "Model" in error_type or "MLflow" in error_type:
            suggestions.append("Check MLflow model URI is correct")
            suggestions.append("Verify model is accessible")
        
        # General suggestions
        if node_name:
            suggestions.append(f"Review node '{node_name}' configuration and inputs")
        
        if rule_name:
            suggestions.append(f"Check rule '{rule_name}' implementation")
        
        return suggestions


class ValidationError(RulebookError):
    """Validation error with field-level details."""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field_errors: Dictionary mapping field names to error messages
            **kwargs: Additional RulebookError arguments
        """
        super().__init__(error_type="ValidationError", message=message, **kwargs)
        self.field_errors = field_errors or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result["field_errors"] = self.field_errors
        return result


class MappingError(RulebookError):
    """Error in field mapping between nodes."""
    
    def __init__(
        self,
        message: str,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        missing_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize mapping error.
        
        Args:
            message: Error message
            source_node: Source node name
            target_node: Target node name
            missing_fields: List of missing field names
            **kwargs: Additional RulebookError arguments
        """
        super().__init__(
            error_type="MappingError",
            message=message,
            node_name=target_node,
            **kwargs,
        )
        self.source_node = source_node
        self.target_node = target_node
        self.missing_fields = missing_fields or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        result["source_node"] = self.source_node
        result["target_node"] = self.target_node
        result["missing_fields"] = self.missing_fields
        return result


# Global error handler
error_handler = RulebookErrorHandler()

