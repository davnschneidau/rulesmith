"""Provider-agnostic GenAI/LLM wrapper."""

from typing import Any, Dict, Optional


class GenAIWrapper:
    """Provider-agnostic LLM interface."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        model_uri: Optional[str] = None,
        gateway_uri: Optional[str] = None,
    ):
        """
        Initialize GenAI wrapper.

        Args:
            provider: Provider name (openai, anthropic, etc.)
            model_name: Model name
            model_uri: Optional MLflow model URI
            gateway_uri: Optional MLflow AI Gateway URI
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.model_uri = model_uri
        self.gateway_uri = gateway_uri
        self._client = None
        self._chain = None

    def _get_client(self):
        """Lazy load the provider client."""
        if self._client is None:
            if self.gateway_uri:
                # Use MLflow AI Gateway
                try:
                    from mlflow.gateway import query, set_gateway_uri

                    set_gateway_uri(self.gateway_uri)
                    self._client = "gateway"
                except ImportError:
                    raise ImportError("MLflow Gateway requires mlflow>=2.9.0")

            elif self.provider == "openai":
                try:
                    import openai

                    self._client = openai
                except ImportError:
                    raise ImportError("OpenAI provider requires 'openai' package")

            elif self.provider == "anthropic":
                try:
                    import anthropic

                    self._client = anthropic
                except ImportError:
                    raise ImportError("Anthropic provider requires 'anthropic' package")

            elif self.model_uri:
                # Try to load as MLflow LangChain model
                try:
                    import mlflow.pyfunc

                    self._chain = mlflow.pyfunc.load_model(self.model_uri)
                    self._client = "mlflow"
                except Exception:
                    raise ValueError(f"Could not load model from URI: {self.model_uri}")

            else:
                raise ValueError(
                    f"Provider '{self.provider}' not supported. "
                    "Use 'openai', 'anthropic', provide model_uri, or gateway_uri."
                )

        return self._client

    def _load_chain(self):
        """Load LangChain chain from MLflow if available."""
        if self._chain is None and self.model_uri:
            try:
                import mlflow.pyfunc

                self._chain = mlflow.pyfunc.load_model(self.model_uri)
            except Exception:
                pass
        return self._chain

    def invoke(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Invoke LLM with prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific arguments

        Returns:
            Response dictionary with 'output', 'tokens', 'cost', etc.
        """
        # Try LangChain chain first
        chain = self._load_chain()
        if chain:
            try:
                result = chain.invoke({"input": prompt, **kwargs})
                if isinstance(result, dict):
                    return {"output": result.get("output", result), **result}
                return {"output": result}
            except Exception:
                pass  # Fall through to direct client

        # Get client
        client = self._get_client()

        # Use MLflow Gateway
        if client == "gateway":
            try:
                from mlflow.gateway import query

                route = kwargs.get("route", f"completions/{self.model_name or 'gpt-4'}")
                response = query(
                    route=route,
                    data={"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens, **kwargs},
                )

                return {
                    "output": response.get("candidates", [{}])[0].get("text", ""),
                    "tokens": {
                        "input_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                        "output_tokens": response.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": response.get("usage", {}).get("total_tokens", 0),
                    },
                    "cost": response.get("usage", {}).get("total_cost", 0.0),
                }
            except Exception as e:
                return {"output": f"Gateway error: {str(e)}", "error": str(e)}

        # Use OpenAI
        elif client and hasattr(client, "chat") and self.provider == "openai":
            try:
                model = self.model_name or kwargs.get("model", "gpt-4")
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **{k: v for k, v in kwargs.items() if k != "model"},
                )

                return {
                    "output": response.choices[0].message.content,
                    "tokens": {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                }
            except Exception as e:
                return {"output": f"OpenAI error: {str(e)}", "error": str(e)}

        # Use Anthropic
        elif client and hasattr(client, "messages") and self.provider == "anthropic":
            try:
                model = self.model_name or kwargs.get("model", "claude-3-5-sonnet-20241022")
                message = client.messages.create(
                    model=model,
                    max_tokens=max_tokens or 1024,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    **{k: v for k, v in kwargs.items() if k != "model"},
                )

                return {
                    "output": message.content[0].text,
                    "tokens": {
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens,
                        "total_tokens": message.usage.input_tokens + message.usage.output_tokens,
                    },
                }
            except Exception as e:
                return {"output": f"Anthropic error: {str(e)}", "error": str(e)}

        # Use MLflow LangChain model
        elif client == "mlflow" and self._chain:
            result = self._chain.invoke({"input": prompt, **kwargs})
            if isinstance(result, dict):
                return {"output": result.get("output", result), **result}
            return {"output": result}

        else:
            return {"output": f"Provider '{self.provider}' not yet fully implemented", "provider": self.provider}

