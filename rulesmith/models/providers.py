"""Provider registry for multi-model support via LangChain/LangGraph.

This module provides a registry of all supported LLM providers and their configurations.
It integrates with LangChain's ChatModel interface to support all LangChain-compatible providers.
"""

import re
from typing import Any, Dict, Optional, Tuple

# Provider registry mapping
PROVIDER_REGISTRY: Dict[str, Dict[str, Any]] = {
    # OpenAI
    "openai": {
        "langchain_class": "ChatOpenAI",
        "package": "langchain-openai",
        "default_model": "gpt-4",
        "model_patterns": [r"^gpt-", r"^o1-", r"^text-"],
    },
    # Anthropic
    "anthropic": {
        "langchain_class": "ChatAnthropic",
        "package": "langchain-anthropic",
        "default_model": "claude-3-5-sonnet-20241022",
        "model_patterns": [r"^claude-"],
    },
    # Google / Gemini
    "google": {
        "langchain_class": "ChatGoogleGenerativeAI",
        "package": "langchain-google-genai",
        "default_model": "gemini-pro",
        "model_patterns": [r"^gemini-"],
    },
    # Cohere
    "cohere": {
        "langchain_class": "ChatCohere",
        "package": "langchain-cohere",
        "default_model": "command",
        "model_patterns": [r"^command", r"^command-r"],
    },
    # HuggingFace
    "huggingface": {
        "langchain_class": "ChatHuggingFace",
        "package": "langchain-huggingface",
        "default_model": None,  # Requires explicit model
        "model_patterns": [],
    },
    # Mistral
    "mistral": {
        "langchain_class": "ChatMistralAI",
        "package": "langchain-mistralai",
        "default_model": "mistral-large",
        "model_patterns": [r"^mistral-"],
    },
    # Azure OpenAI
    "azure-openai": {
        "langchain_class": "AzureChatOpenAI",
        "package": "langchain-openai",
        "default_model": "gpt-4",
        "model_patterns": [r"^gpt-", r"^o1-"],
    },
    # Fireworks AI
    "fireworks": {
        "langchain_class": "ChatFireworks",
        "package": "langchain-fireworks",
        "default_model": "accounts/fireworks/models/llama-v3-70b-instruct",
        "model_patterns": [r"^accounts/fireworks/"],
    },
    # Together AI
    "together": {
        "langchain_class": "ChatTogether",
        "package": "langchain-together",
        "default_model": "meta-llama/Llama-3-8b-chat-hf",
        "model_patterns": [],
    },
    # Bedrock (AWS)
    "bedrock": {
        "langchain_class": "ChatBedrock",
        "package": "langchain-aws",
        "default_model": "anthropic.claude-v2",
        "model_patterns": [],
    },
    # Vertex AI (Google Cloud)
    "vertexai": {
        "langchain_class": "ChatVertexAI",
        "package": "langchain-google-vertexai",
        "default_model": "gemini-pro",
        "model_patterns": [r"^gemini-"],
    },
    # Groq
    "groq": {
        "langchain_class": "ChatGroq",
        "package": "langchain-groq",
        "default_model": "llama-3-8b-8192",
        "model_patterns": [r"^llama-", r"^mixtral-"],
    },
    # Ollama (local)
    "ollama": {
        "langchain_class": "ChatOllama",
        "package": "langchain-ollama",
        "default_model": "llama2",
        "model_patterns": [],
    },
    # AI21
    "ai21": {
        "langchain_class": "ChatAI21",
        "package": "langchain-ai21",
        "default_model": "j2-ultra",
        "model_patterns": [r"^j2-"],
    },
    # DeepInfra
    "deepinfra": {
        "langchain_class": "ChatDeepInfra",
        "package": "langchain-deepinfra",
        "default_model": "meta-llama/Llama-2-70b-chat-hf",
        "model_patterns": [],
    },
}


def detect_provider_from_model(model_name: str) -> Optional[str]:
    """
    Auto-detect provider from model name.
    
    Args:
        model_name: Model name (e.g., "gpt-4", "claude-3-5-sonnet")
    
    Returns:
        Provider name or None if not detected
    """
    model_name_lower = model_name.lower()
    
    # Check each provider's model patterns
    for provider, config in PROVIDER_REGISTRY.items():
        patterns = config.get("model_patterns", [])
        for pattern in patterns:
            if re.match(pattern, model_name_lower):
                return provider
    
    # Check for explicit provider prefixes
    if model_name_lower.startswith("openai:"):
        return "openai"
    elif model_name_lower.startswith("anthropic:"):
        return "anthropic"
    elif model_name_lower.startswith("google:"):
        return "google"
    elif model_name_lower.startswith("azure:"):
        return "azure-openai"
    
    return None


def get_provider_config(provider: str) -> Dict[str, Any]:
    """
    Get provider configuration.
    
    Args:
        provider: Provider name
    
    Returns:
        Provider configuration dictionary
    """
    return PROVIDER_REGISTRY.get(provider.lower(), {})


def create_langchain_chat_model(
    provider: str,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a LangChain ChatModel instance for the given provider.
    
    Args:
        provider: Provider name
        model_name: Optional model name (uses default if not provided)
        **kwargs: Provider-specific configuration (API keys, endpoints, etc.)
    
    Returns:
        LangChain ChatModel instance
    
    Raises:
        ImportError: If required package is not installed
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()
    config = get_provider_config(provider_lower)
    
    if not config:
        raise ValueError(
            f"Provider '{provider}' not supported. "
            f"Supported providers: {', '.join(PROVIDER_REGISTRY.keys())}"
        )
    
    langchain_class_name = config["langchain_class"]
    package = config["package"]
    default_model = config.get("default_model")
    
    # Get model name
    model = model_name or kwargs.get("model") or default_model
    if not model:
        raise ValueError(f"Model name required for provider '{provider}'")
    
    # Try to import and create the LangChain class
    try:
        # Try importing from langchain package structure
        # Most providers are in langchain-{provider} packages
        module_name = package.replace("langchain-", "").replace("-", "_")
        
        # Try different import patterns
        chat_model_class = None
        try:
            # Pattern 1: langchain_{provider} (most common)
            from importlib import import_module
            module = import_module(f"langchain_{module_name}")
            chat_model_class = getattr(module, langchain_class_name)
        except (ImportError, AttributeError):
            try:
                # Pattern 2: langchain.chat_models (older versions)
                from langchain import chat_models
                chat_model_class = getattr(chat_models, langchain_class_name)
            except (ImportError, AttributeError):
                try:
                    # Pattern 3: langchain_community
                    from langchain_community import chat_models as community_models
                    chat_model_class = getattr(community_models, langchain_class_name)
                except (ImportError, AttributeError):
                    # Pattern 4: Direct from package (e.g., langchain_openai)
                    try:
                        direct_module = import_module(package)
                        chat_model_class = getattr(direct_module, langchain_class_name)
                    except (ImportError, AttributeError):
                        pass
        
        if chat_model_class is None:
            raise ImportError(
                f"Could not import {langchain_class_name} from {package}. "
                f"Install it with: pip install {package}"
            )
    
        # Create instance
        return chat_model_class(model=model, **kwargs)
    
    except ImportError as e:
        raise ImportError(
            f"Provider '{provider}' requires {package}. "
            f"Install it with: pip install {package}"
        ) from e


def list_supported_providers() -> list[str]:
    """List all supported providers."""
    return list(PROVIDER_REGISTRY.keys())


def get_provider_info(provider: str) -> Dict[str, Any]:
    """
    Get information about a provider.
    
    Args:
        provider: Provider name
    
    Returns:
        Dictionary with provider information
    """
    config = get_provider_config(provider)
    if not config:
        return {}
    
    return {
        "provider": provider,
        "langchain_class": config["langchain_class"],
        "package": config["package"],
        "default_model": config.get("default_model"),
        "model_patterns": config.get("model_patterns", []),
    }

