"""Runtime context for execution."""

from typing import Any, Dict, Optional
from uuid import uuid4


class RunContext:
    """Execution context for a rulebook run."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        identity: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.run_id = run_id or str(uuid4())
        self.identity = identity
        self.tags = tags or {}
        self.seed = seed
        self.params = params or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "run_id": self.run_id,
            "identity": self.identity,
            "tags": self.tags,
            "seed": self.seed,
            "params": self.params,
        }

