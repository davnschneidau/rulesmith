"""Centralized logging utilities for Rulesmith."""

import logging
from typing import Any, Dict, Optional

# Configure default logger
_logger = logging.getLogger("rulesmith")
_logger.setLevel(logging.INFO)

# Create child loggers for different components
_loggers = {
    "alerts": logging.getLogger("rulesmith.alerts"),
    "execution": logging.getLogger("rulesmith.execution"),
    "lifecycle": logging.getLogger("rulesmith.lifecycle"),
    "monitoring": logging.getLogger("rulesmith.monitoring"),
    "explainability": logging.getLogger("rulesmith.explainability"),
    "operations": logging.getLogger("rulesmith.operations"),
    "compliance": logging.getLogger("rulesmith.compliance"),
    "change": logging.getLogger("rulesmith.change"),
}


def get_logger(component: Optional[str] = None) -> logging.Logger:
    """
    Get logger for a component.
    
    Args:
        component: Component name (e.g., "alerts", "execution")
                   If None, returns main rulesmith logger
    
    Returns:
        Logger instance
    """
    if component:
        return _loggers.get(component, _logger)
    return _logger


def log_error(
    logger: logging.Logger,
    message: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log an error with context.
    
    Args:
        logger: Logger instance
        message: Error message
        error: Exception object
        context: Optional context dictionary
    """
    error_msg = f"{message}: {type(error).__name__}: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    logger.error(error_msg, exc_info=True)


def log_warning(
    logger: logging.Logger,
    message: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log a warning with context.
    
    Args:
        logger: Logger instance
        message: Warning message
        context: Optional context dictionary
    """
    if context:
        message += f" | Context: {context}"
    logger.warning(message)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure logging level for all Rulesmith loggers.
    
    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    _logger.setLevel(level)
    for logger in _loggers.values():
        logger.setLevel(level)

