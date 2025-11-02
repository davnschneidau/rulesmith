"""Tests for BYOM and GenAI integration."""

import pytest

from rulesmith import Rulebook
from rulesmith.dag.registry import rule_registry
from rulesmith.models.genai import GenAIWrapper
from rulesmith.models.mlflow_byom import BYOMRef


class TestBYOM:
    """Test Bring Your Own Model integration."""

    def test_byom_ref_creation(self):
        """Test creating BYOM reference."""
        # Note: This would require an actual MLflow model URI in practice
        # For testing, we'll just verify the structure
        model_uri = "models:/test_model/1"
        byom_ref = BYOMRef(model_uri)
        assert byom_ref.model_uri == model_uri
        assert byom_ref._model is None  # Not loaded yet

    def test_byom_node_creation(self):
        """Test creating BYOM node."""
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_byom("test_model", "models:/test_model/1")

        spec = rb.to_spec()
        assert len(spec.nodes) == 1
        assert spec.nodes[0].kind == "byom"
        assert spec.nodes[0].model_uri == "models:/test_model/1"


class TestGenAIWrapper:
    """Test GenAI wrapper."""

    def test_genai_wrapper_creation(self):
        """Test creating GenAI wrapper."""
        wrapper = GenAIWrapper(provider="openai", model_name="gpt-4")
        assert wrapper.provider == "openai"
        assert wrapper.model_name == "gpt-4"

    def test_genai_wrapper_with_gateway(self):
        """Test GenAI wrapper with gateway."""
        wrapper = GenAIWrapper(
            provider="openai",
            gateway_uri="http://localhost:8000",
            model_name="gpt-4",
        )
        assert wrapper.gateway_uri == "http://localhost:8000"

    def test_genai_wrapper_with_model_uri(self):
        """Test GenAI wrapper with MLflow model URI."""
        wrapper = GenAIWrapper(
            provider="openai",
            model_uri="models:/llm_chain/1",
        )
        assert wrapper.model_uri == "models:/llm_chain/1"


class TestGenAINode:
    """Test GenAI node integration."""

    def test_genai_node_creation(self):
        """Test creating GenAI node."""
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_genai("llm_node", provider="openai", model_name="gpt-4")

        spec = rb.to_spec()
        assert len(spec.nodes) == 1
        assert spec.nodes[0].kind == "llm"
        assert spec.nodes[0].params.get("provider") == "openai"
        assert spec.nodes[0].params.get("model_name") == "gpt-4"

    def test_genai_node_with_gateway(self):
        """Test GenAI node with gateway."""
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_genai(
            "llm_node",
            provider="openai",
            gateway_uri="http://localhost:8000",
            model_name="gpt-4",
        )

        spec = rb.to_spec()
        assert spec.nodes[0].params.get("gateway_uri") == "http://localhost:8000"


class TestLangChainNode:
    """Test LangChain node integration."""

    def test_langchain_node_creation(self):
        """Test creating LangChain node."""
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_langchain("chain_node", "models:/langchain_chain/1")

        spec = rb.to_spec()
        assert len(spec.nodes) == 1
        assert spec.nodes[0].kind == "langchain"
        assert spec.nodes[0].model_uri == "models:/langchain_chain/1"


class TestLangGraphNode:
    """Test LangGraph node integration."""

    def test_langgraph_node_creation(self):
        """Test creating LangGraph node."""
        rb = Rulebook(name="test", version="1.0.0")
        rb.add_langgraph("graph_node", "models:/langgraph_graph/1")

        spec = rb.to_spec()
        assert len(spec.nodes) == 1
        assert spec.nodes[0].kind == "langgraph"
        assert spec.nodes[0].model_uri == "models:/langgraph_graph/1"

