"""MLflow BYOM adapter for loading and executing MLflow models and LangChain models."""

from typing import Any, Dict, Optional

import mlflow.pyfunc

from rulesmith.models.base import ModelInterface


class BYOMRef(ModelInterface):
    """
    Reference to an MLflow model or LangChain model for ModelNode.
    
    Supports:
    - MLflow models (sklearn, PyTorch, XGBoost, custom pyfunc)
    - LangChain chains/models loaded from MLflow
    - Direct LangChain model initialization
    """

    def __init__(self, model_uri: Optional[str] = None, langchain_model: Optional[Any] = None):
        """
        Initialize model reference.

        Args:
            model_uri: MLflow model URI (e.g., "models:/my_model/1" or "runs:/run_id/model")
            langchain_model: Optional direct LangChain model/chain instance
        """
        if not model_uri and not langchain_model:
            raise ValueError("Either model_uri or langchain_model must be provided")
        
        self.model_uri = model_uri
        self.langchain_model = langchain_model
        self._model = None
        self._is_langchain = langchain_model is not None

    def load(self):
        """Lazy load the model."""
        if self._model is None:
            if self.langchain_model:
                # Direct LangChain model provided
                self._model = self.langchain_model
                self._is_langchain = True
            elif self.model_uri:
                # Try to load from MLflow
                try:
                    self._model = mlflow.pyfunc.load_model(self.model_uri)
                    # Check if it's a LangChain model
                    if hasattr(self._model, "invoke") or hasattr(self._model, "predict"):
                        # Likely a LangChain chain/model
                        self._is_langchain = True
                    else:
                        self._is_langchain = False
                except Exception as e:
                    raise ValueError(f"Could not load model from URI: {self.model_uri}: {str(e)}")
        return self._model

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model prediction."""
        model = self.load()

        # Handle LangChain models
        if self._is_langchain or (hasattr(model, "invoke") or hasattr(model, "predict")):
            try:
                # Try LangChain invoke interface
                if hasattr(model, "invoke"):
                    result = model.invoke(inputs)
                elif hasattr(model, "predict"):
                    result = model.predict(inputs)
                else:
                    # Fallback to callable
                    result = model(inputs)
                
                # Normalize result
                if isinstance(result, dict):
                    return result
                elif isinstance(result, str):
                    return {"output": result}
                else:
                    return {"result": result}
            except Exception as e:
                # If LangChain fails, try MLflow predict
                pass

        # Handle MLflow models (traditional ML models)
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

