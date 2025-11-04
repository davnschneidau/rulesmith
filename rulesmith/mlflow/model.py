"""MLflow pyfunc model for Rulebooks."""

import json
import os
from typing import Any, Dict, Optional

import mlflow.pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.pyfunc import PythonModel, PythonModelContext

from rulesmith.dag.graph import Rulebook
from rulesmith.io.ser import RulebookSpec


def log_rulebook_model(
    rulebook: Rulebook,
    artifact_path: str = "model",
    registered_model_name: Optional[str] = None,
    extra_pip_requirements: Optional[list[str]] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Model:
    """
    Log a rulebook as an MLflow model.

    Args:
        rulebook: Rulebook instance
        artifact_path: Path within the run's artifact directory
        registered_model_name: Optional registered model name
        extra_pip_requirements: Additional pip requirements
        signature: Optional model signature
        input_example: Optional input example
        metadata: Optional metadata dictionary

    Returns:
        MLflow Model object
    """
    import mlflow

    # Serialize rulebook to spec
    spec = rulebook.to_spec()

    # Log spec as dict (artifact)
    mlflow.log_dict(spec.model_dump(), "rulebook_spec.json")

    # Log model
    model = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=RulebookPyfunc(spec=spec),
        registered_model_name=registered_model_name,
        extra_pip_requirements=extra_pip_requirements,
        signature=signature,
        input_example=input_example,
        metadata=metadata or {},
    )

    return model


class RulebookPyfunc(PythonModel):
    """MLflow pyfunc model wrapper for Rulebooks."""

    def __init__(self, spec: Optional[RulebookSpec] = None):
        self.spec = spec
        self._rulebook: Optional[Rulebook] = None

    def load_context(self, context: PythonModelContext) -> None:
        """Load rulebook spec from artifacts."""
        # Get the artifact path
        artifact_path = context.artifacts.get("rulebook_spec.json")
        if artifact_path:
            spec_path = artifact_path
        else:
            # Try to find it in the artifact directory
            artifact_dir = getattr(context, "artifact_path", "model")
            spec_path = os.path.join(artifact_dir, "rulebook_spec.json")

        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Rulebook spec not found at {spec_path}")

        with open(spec_path, "r") as f:
            spec_dict = json.load(f)
        self.spec = RulebookSpec(**spec_dict)

    def _get_rulebook(self) -> Rulebook:
        """Lazy load rulebook from spec."""
        if self._rulebook is None:
            # Reconstruct rulebook from spec
            # This is a simplified version - in practice, we'd need to
            # reconstruct all node types from their specs
            self._rulebook = Rulebook(
                name=self.spec.name,
                version=self.spec.version,
                metadata=self.spec.metadata,
            )
            # TODO: Reconstruct nodes and edges from spec
            # This requires access to rule registry and model URIs
        return self._rulebook

    def predict(
        self,
        context: PythonModelContext,
        model_input: Any,
    ) -> Any:
        """
        Execute rulebook prediction.

        Args:
            context: MLflow model context
            model_input: Input payload (dict or pandas DataFrame)

        Returns:
            Prediction result
        """
        # Convert input to dict if DataFrame
        if hasattr(model_input, "to_dict"):
            payload = model_input.iloc[0].to_dict()
        elif isinstance(model_input, dict):
            payload = model_input
        elif isinstance(model_input, (list, tuple)) and len(model_input) > 0:
            payload = model_input[0] if isinstance(model_input[0], dict) else {"input": model_input[0]}
        else:
            payload = {"input": model_input}

        # Execute rulebook
        rulebook = self._get_rulebook()
        result = rulebook.run(payload)

        # Convert result to format expected by MLflow
        return result

