"""Developer Experience modules (single-file rulebooks, strict typing, error ergonomics, auto-mapping)."""

from rulesmith.dx.auto_mapping import (
    AutoMapper,
    auto_mapper,
)
from rulesmith.dx.errors import (
    MappingError,
    RulebookError,
    RulebookErrorHandler,
    ValidationError,
    error_handler,
)
from rulesmith.dx.single_file import (
    RulebookDSL,
    load_rsm_file,
    save_rsm_file,
)
from rulesmith.dx.typing import (
    TypeInfo,
    TypeValidator,
    create_input_schema,
    create_output_schema,
    type_validator,
)

__all__ = [
    # Single-file rulebooks
    "RulebookDSL",
    "load_rsm_file",
    "save_rsm_file",
    # Strict typing
    "TypeValidator",
    "TypeInfo",
    "create_input_schema",
    "create_output_schema",
    "type_validator",
    # Error ergonomics
    "RulebookError",
    "RulebookErrorHandler",
    "ValidationError",
    "MappingError",
    "error_handler",
    # Auto-mapping
    "AutoMapper",
    "auto_mapper",
]

