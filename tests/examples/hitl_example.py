"""Examples demonstrating Human-in-the-Loop (HITL) system."""

from rulesmith import rule, Rulebook
from rulesmith.hitl.adapters import InMemoryQueue, LocalFileQueue
from rulesmith.hitl.base import ReviewDecision


@rule(name="generate_proposal", inputs=["topic"], outputs=["proposal"])
def generate_proposal(topic: str) -> dict:
    """Generate a proposal."""
    return {"proposal": f"Proposal for {topic}"}


@rule(name="process_output", inputs=["proposal"], outputs=["final_output"])
def process_output(proposal: dict) -> dict:
    """Process the final output."""
    return {"final_output": proposal.get("proposal", "")}


def example_basic_hitl():
    """Example with basic HITL node."""
    print("=" * 60)
    print("Example: Basic HITL Integration")
    print("=" * 60)

    # Create in-memory queue
    queue = InMemoryQueue()

    rb = Rulebook(name="hitl_example", version="1.0.0")
    rb.add_rule(generate_proposal, as_name="generate")
    rb.add_hitl("review", queue, timeout=5.0)
    rb.add_rule(process_output, as_name="process")

    rb.connect("generate", "review")
    rb.connect("review", "process")

    print("HITL Rulebook:")
    print("  - Flow: generate -> review -> process")
    print("  - Review node blocks execution until decision")
    print()
    print("To test:")
    print("  1. Run rulebook (will block at review)")
    print("  2. Add decision manually: queue.add_decision(request_id, decision)")
    print("  3. Execution continues")
    print()


def example_async_hitl():
    """Example with async HITL node."""
    print("=" * 60)
    print("Example: Async HITL (Non-Blocking)")
    print("=" * 60)

    queue = InMemoryQueue()

    rb = Rulebook(name="async_hitl_example", version="1.0.0")
    rb.add_rule(generate_proposal, as_name="generate")
    rb.add_hitl("review", queue, async_mode=True)  # Non-blocking
    rb.add_rule(process_output, as_name="process")

    print("Async HITL:")
    print("  - async_mode=True: Execution continues immediately")
    print("  - Returns pending state with request_id")
    print("  - Process decision later via queue")
    print()
    print("Use case: Background review, continue processing")
    print()


def example_active_learning():
    """Example with active learning threshold."""
    print("=" * 60)
    print("Example: Active Learning with Confidence Threshold")
    print("=" * 60)

    queue = InMemoryQueue()

    rb = Rulebook(name="active_learning_example", version="1.0.0")
    rb.add_rule(generate_proposal, as_name="generate")
    rb.add_hitl("review", queue, active_learning_threshold=0.7)
    rb.add_rule(process_output, as_name="process")

    print("Active Learning:")
    print("  - active_learning_threshold=0.7")
    print("  - If confidence >= 0.7: Skip review, use model output")
    print("  - If confidence < 0.7: Request human review")
    print()
    print("Benefits:")
    print("  - Only review uncertain cases")
    print("  - Reduce review load")
    print("  - Improve over time")
    print()


def example_local_file_queue():
    """Example with local file queue."""
    print("=" * 60)
    print("Example: Local File Queue")
    print("=" * 60)

    queue = LocalFileQueue(queue_dir="./hitl_queue")

    rb = Rulebook(name="file_queue_example", version="1.0.0")
    rb.add_rule(generate_proposal, as_name="generate")
    rb.add_hitl("review", queue)

    print("Local File Queue:")
    print("  - Stores requests in: ./hitl_queue/requests/")
    print("  - Stores decisions in: ./hitl_queue/decisions/")
    print("  - Persistent across restarts")
    print()
    print("Usage:")
    print("  1. Review requests appear as JSON files")
    print("  2. Create decision files with same ID")
    print("  3. Rulebook execution continues")
    print()


def example_postgres_queue():
    """Example with PostgreSQL queue."""
    print("=" * 60)
    print("Example: PostgreSQL Queue")
    print("=" * 60)

    # Note: Requires postgres connection
    # from rulesmith.hitl.adapters import PostgresQueue
    # queue = PostgresQueue("postgresql://user:pass@localhost/db")

    rb = Rulebook(name="postgres_queue_example", version="1.0.0")
    # rb.add_hitl("review", queue)

    print("PostgreSQL Queue:")
    print("  - Production-ready persistent storage")
    print("  - Supports multiple workers")
    print("  - Transactional guarantees")
    print()
    print("Setup:")
    print("  queue = PostgresQueue('postgresql://user:pass@host/db')")
    print()


def example_redis_queue():
    """Example with Redis queue."""
    print("=" * 60)
    print("Example: Redis Queue")
    print("=" * 60)

    # Note: Requires redis
    # from rulesmith.hitl.adapters import RedisQueue
    # queue = RedisQueue("redis://localhost:6379")

    rb = Rulebook(name="redis_queue_example", version="1.0.0")
    # rb.add_hitl("review", queue)

    print("Redis Queue:")
    print("  - Fast, in-memory storage")
    print("  - Good for high-throughput scenarios")
    print("  - Can be configured with persistence")
    print()
    print("Setup:")
    print("  queue = RedisQueue('redis://localhost:6379')")
    print()


def example_slack_queue():
    """Example with Slack notifications."""
    print("=" * 60)
    print("Example: Slack Queue (Notifications)")
    print("=" * 60)

    # Note: Requires Slack webhook
    # from rulesmith.hitl.adapters import SlackQueue
    # queue = SlackQueue("https://hooks.slack.com/services/...")

    rb = Rulebook(name="slack_queue_example", version="1.0.0")
    # rb.add_hitl("review", queue)

    print("Slack Queue:")
    print("  - Sends notifications to Slack channel")
    print("  - Decisions via external API/webhook")
    print("  - Good for team notifications")
    print()
    print("Setup:")
    print("  queue = SlackQueue('https://hooks.slack.com/...')")
    print()


def example_hitl_workflow():
    """Example complete HITL workflow."""
    print("=" * 60)
    print("Example: Complete HITL Workflow")
    print("=" * 60)

    queue = InMemoryQueue()

    rb = Rulebook(name="hitl_workflow", version="1.0.0")
    rb.add_rule(generate_proposal, as_name="generate")
    rb.add_hitl("review", queue, active_learning_threshold=0.7)
    rb.add_rule(process_output, as_name="process")

    rb.connect("generate", "review")
    rb.connect("review", "process")

    print("Complete Workflow:")
    print("  1. Generate proposal (rule)")
    print("  2. Check confidence:")
    print("     - If high: Auto-approve, continue")
    print("     - If low: Request human review")
    print("  3. Human reviews and approves/rejects")
    print("  4. Process final output")
    print()
    print("Benefits:")
    print("  - Quality control")
    print("  - Active learning")
    print("  - Human oversight")
    print("  - Audit trail")
    print()


if __name__ == "__main__":
    example_basic_hitl()
    example_async_hitl()
    example_active_learning()
    example_local_file_queue()
    example_postgres_queue()
    example_redis_queue()
    example_slack_queue()
    example_hitl_workflow()

