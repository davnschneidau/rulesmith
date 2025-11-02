"""MLflow BYOM adapter for loading and executing MLflow models."""

from typing import Any, Dict

import mlflow.pyfunc

from rulesmith.models.base import ModelInterface


class BYOMRef(ModelInterface):
    """Reference to an MLflow model for BYOM nodes."""

    def __init__(self, model_uri: str):
        """
        Initialize BYOM reference.

        Args:
            model_uri: MLflow model URI (e.g., "models:/my_model/1" or "runs:/run_id/model")
        """
        self.model_uri = model_uri
        self._model = None

    def load(self):
        """Lazy load the MLflow model."""
        if self._model is None:
            self._model = mlflow.pyfunc.load_model(self.model_uri)
        return self._model

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model prediction."""
        model = self.load()

        # Convert inputs to DataFrame if needed
        try:
            import pandas as pd

            if isinstance(inputs, dict):
                input_df = pd.DataFrame([inputs])
            else:
                input_df = inputs

            result = model.predict(input_df)

            # Convert result to dict
            if hasattr(result, "iloc"):
                return result.iloc[0].to_dict()
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                return {"prediction": result[0]}
            else:
                return {"result": result}
        except ImportError:
            # Fallback if pandas not available
            result = model.predict(inputs)
            if isinstance(result, dict):
                return result
            return {"result": result}

