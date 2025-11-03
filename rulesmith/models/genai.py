"""Provider-agnostic GenAI/LLM wrapper with LangChain ChatModel integration."""

from typing import Any, Dict, Optional

from rulesmith.models.providers import (
    create_langchain_chat_model,
    detect_provider_from_model,
    get_provider_config,
    list_supported_providers,
)


class GenAIWrapper:
    """
    Provider-agnostic LLM interface with multi-model support via LangChain.
    
    This wrapper uses LangChain's ChatModel interface as the primary abstraction,
    supporting all LangChain-compatible providers (OpenAI, Anthropic, Google, etc.).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        model_uri: Optional[str] = None,
        gateway_uri: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize GenAI wrapper.

        Args:
            provider: Provider name (openai, anthropic, google, etc.). If None, auto-detects from model_name.
            model_name: Model name (e.g., "gpt-4", "claude-3-5-sonnet"). Provider auto-detected if not specified.
            model_uri: Optional MLflow model URI
            gateway_uri: Optional MLflow AI Gateway URI
            **kwargs: Provider-specific configuration (API keys, endpoints, temperature, etc.)
        """
        self.model_uri = model_uri
        self.gateway_uri = gateway_uri
        self._provider_config = kwargs.copy()
        
        # Auto-detect provider from model name if not provided
        if provider:
            self.provider = provider.lower()
        elif model_name:
            detected = detect_provider_from_model(model_name)
            if detected:
                self.provider = detected
            else:
                # Default to openai if can't detect
                self.provider = "openai"
        else:
            self.provider = "openai"
        
        self.model_name = model_name
        self._langchain_model = None
        self._client = None
        self._chain = None

    def _get_langchain_model(self):
        """Lazy load LangChain ChatModel for the provider."""
        if self._langchain_model is None:
            # Try to use LangChain ChatModel first (preferred)
            try:
                provider_config = get_provider_config(self.provider)
                if provider_config:
                    # Extract model name and provider-specific config
                    model = self.model_name or provider_config.get("default_model")
                    if not model:
                        raise ValueError(f"Model name required for provider '{self.provider}'")
                    
                    # Create LangChain ChatModel
                    self._langchain_model = create_langchain_chat_model(
                        provider=self.provider,
                        model_name=model,
                        **self._provider_config,
                    )
                    return self._langchain_model
            except (ImportError, ValueError) as e:
                # Fall back to direct client if LangChain not available
                pass
        
        return self._langchain_model
    
    def _get_client(self):
        """Lazy load the provider client (fallback for backward compatibility)."""
        if self._client is None:
            if self.gateway_uri:
                # Use MLflow AI Gateway
                try:
                    from mlflow.gateway import query, set_gateway_uri

                    set_gateway_uri(self.gateway_uri)
                    self._client = "gateway"
                except ImportError:
                    raise ImportError("MLflow Gateway requires mlflow>=2.9.0")

            elif self.model_uri:
                # Try to load as MLflow LangChain model
                try:
                    import mlflow.pyfunc

                    self._chain = mlflow.pyfunc.load_model(self.model_uri)
                    self._client = "mlflow"
                except Exception:
                    raise ValueError(f"Could not load model from URI: {self.model_uri}")

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

            else:
                # For other providers, try LangChain first
                langchain_model = self._get_langchain_model()
                if langchain_model:
                    self._client = "langchain"
                else:
                    raise ValueError(
                        f"Provider '{self.provider}' not supported. "
                        f"Supported providers: {', '.join(list_supported_providers())}"
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
        # Try LangChain ChatModel first (preferred method)
        langchain_model = self._get_langchain_model()
        if langchain_model:
            try:
                # Use LangChain ChatModel interface
                from langchain_core.messages import HumanMessage
                
                messages = [HumanMessage(content=prompt)]
                response = langchain_model.invoke(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                
                # Extract output and metadata
                output = response.content if hasattr(response, "content") else str(response)
                
                # Try to get usage/token info
                tokens = {}
                if hasattr(response, "response_metadata"):
                    metadata = response.response_metadata or {}
                    usage = metadata.get("usage", {})
                    if usage:
                        tokens = {
                            "input_tokens": usage.get("prompt_tokens", 0),
                            "output_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }
                
                return {
                    "output": output,
                    "tokens": tokens,
                    "provider": self.provider,
                }
            except Exception as e:
                # Fall through to other methods if LangChain fails
                pass
        
        # Try LangChain chain from MLflow
        chain = self._load_chain()
        if chain:
            try:
                result = chain.invoke({"input": prompt, **kwargs})
                if isinstance(result, dict):
                    return {"output": result.get("output", result), **result}
                return {"output": result}
            except Exception:
                pass  # Fall through to direct client

        # Get client (fallback for backward compatibility)
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

        elif client == "langchain":
            # Should have been handled above, but fallback
            return {"output": "LangChain model error", "provider": self.provider}
        else:
            # Try to get LangChain model as last resort
            langchain_model = self._get_langchain_model()
            if langchain_model:
                try:
                    from langchain_core.messages import HumanMessage
                    messages = [HumanMessage(content=prompt)]
                    response = langchain_model.invoke(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
                    output = response.content if hasattr(response, "content") else str(response)
                    return {"output": output, "provider": self.provider}
                except Exception as e:
                    return {"output": f"Provider '{self.provider}' error: {str(e)}", "provider": self.provider}
            
            return {
                "output": f"Provider '{self.provider}' not yet fully implemented. "
                         f"Supported providers: {', '.join(list_supported_providers())}",
                "provider": self.provider,
            }

