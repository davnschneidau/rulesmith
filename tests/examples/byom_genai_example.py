"""Example demonstrating BYOM and GenAI integration."""

from rulesmith import rule, Rulebook


@rule(name="prepare_input", inputs=["text"], outputs=["input_data"])
def prepare_input(text: str) -> dict:
    """Prepare input for model."""
    return {"input_data": {"text": text}}


@rule(name="process_output", inputs=["prediction"], outputs=["result"])
def process_output(prediction: dict) -> dict:
    """Process model output."""
    return {"result": prediction.get("result", prediction)}


def example_byom_rulebook():
    """Example rulebook with BYOM node."""
    print("=" * 60)
    print("Example: BYOM (Bring Your Own Model) Integration")
    print("=" * 60)

    rb = Rulebook(name="byom_example", version="1.0.0")

    # Add rule to prepare input
    rb.add_rule(prepare_input, as_name="prepare")

    # Add BYOM node (requires actual MLflow model URI)
    # rb.add_byom("classifier", "models:/sentiment_classifier/1")

    # Add rule to process output
    rb.add_rule(process_output, as_name="process")

    # Connect nodes
    # rb.connect("prepare", "classifier")
    # rb.connect("classifier", "process")

    print("BYOM rulebook structure:")
    print(f"  - Name: {rb.name}")
    print(f"  - Version: {rb.version}")
    print("  - Nodes: prepare -> [BYOM] -> process")
    print()
    print("To use:")
    print("  1. Log your MLflow model")
    print("  2. Use model URI in add_byom()")
    print("  3. Execute rulebook with input data")
    print()


def example_genai_rulebook():
    """Example rulebook with GenAI node."""
    print("=" * 60)
    print("Example: GenAI/LLM Integration")
    print("=" * 60)

    rb = Rulebook(name="genai_example", version="1.0.0")

    # Add GenAI node with OpenAI provider
    rb.add_genai(
        "llm_node",
        provider="openai",
        model_name="gpt-4",
        params={"temperature": 0.7, "max_tokens": 100},
    )

    print("GenAI rulebook structure:")
    print(f"  - Name: {rb.name}")
    print(f"  - Version: {rb.version}")
    print("  - Nodes: llm_node (OpenAI GPT-4)")
    print()
    print("Supported providers:")
    print("  - openai: OpenAI models (requires 'openai' package)")
    print("  - anthropic: Anthropic Claude (requires 'anthropic' package)")
    print("  - MLflow Gateway: Via gateway_uri parameter")
    print("  - MLflow LangChain: Via model_uri parameter")
    print()


def example_langchain_rulebook():
    """Example rulebook with LangChain node."""
    print("=" * 60)
    print("Example: LangChain Integration")
    print("=" * 60)

    rb = Rulebook(name="langchain_example", version="1.0.0")

    # Add LangChain node (requires MLflow model with LangChain chain)
    # rb.add_langchain("chain_node", "models:/my_langchain_chain/1")

    print("LangChain rulebook structure:")
    print(f"  - Name: {rb.name}")
    print(f"  - Version: {rb.version}")
    print("  - Nodes: chain_node (LangChain)")
    print()
    print("To use:")
    print("  1. Log LangChain chain as MLflow model")
    print("  2. Use model URI in add_langchain()")
    print("  3. Execute rulebook")
    print()


def example_langgraph_rulebook():
    """Example rulebook with LangGraph node."""
    print("=" * 60)
    print("Example: LangGraph Integration")
    print("=" * 60)

    rb = Rulebook(name="langgraph_example", version="1.0.0")

    # Add LangGraph node (requires MLflow model with LangGraph graph)
    # rb.add_langgraph("graph_node", "models:/my_langgraph_graph/1")

    print("LangGraph rulebook structure:")
    print(f"  - Name: {rb.name}")
    print(f"  - Version: {rb.version}")
    print("  - Nodes: graph_node (LangGraph)")
    print()
    print("To use:")
    print("  1. Log LangGraph graph as MLflow model")
    print("  2. Use model URI in add_langgraph()")
    print("  3. Execute rulebook")
    print()


def example_hybrid_rulebook():
    """Example rulebook combining rules, BYOM, and GenAI."""
    print("=" * 60)
    print("Example: Hybrid Rulebook (Rules + BYOM + GenAI)")
    print("=" * 60)

    rb = Rulebook(name="hybrid_example", version="1.0.0")

    # Add rules
    rb.add_rule(prepare_input, as_name="prepare")

    # Add BYOM node
    # rb.add_byom("classifier", "models:/classifier/1")

    # Add GenAI node
    rb.add_genai("llm_fallback", provider="openai", model_name="gpt-3.5-turbo")

    # Add output processing
    rb.add_rule(process_output, as_name="process")

    # Connect: prepare -> classifier -> llm_fallback -> process
    # rb.connect("prepare", "classifier")
    # rb.connect("classifier", "llm_fallback")
    rb.connect("prepare", "llm_fallback")
    rb.connect("llm_fallback", "process")

    print("Hybrid rulebook structure:")
    print(f"  - Name: {rb.name}")
    print(f"  - Version: {rb.version}")
    print("  - Flow: Rules -> BYOM -> GenAI -> Rules")
    print()
    print("This demonstrates combining:")
    print("  - Business logic rules")
    print("  - ML models (BYOM)")
    print("  - LLM inference (GenAI)")
    print()


if __name__ == "__main__":
    example_byom_rulebook()
    example_genai_rulebook()
    example_langchain_rulebook()
    example_langgraph_rulebook()
    example_hybrid_rulebook()

