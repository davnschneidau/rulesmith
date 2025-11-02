"""Base model interface for BYOM (Bring Your Own Model)."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelInterface(ABC):
    """Base interface for all model types."""

    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model prediction.

        Args:
            inputs: Input dictionary

        Returns:
            Output dictionary
        """
        pass

