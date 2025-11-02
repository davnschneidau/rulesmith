"""HITL queue adapters (LocalFile, Postgres, Redis, Slack)."""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from rulesmith.hitl.base import HITLQueue, ReviewDecision, ReviewRequest


class LocalFileQueue(HITLQueue):
    """Local file-based HITL queue (for development/testing)."""

    def __init__(self, queue_dir: str = ".hitl_queue"):
        """
        Initialize local file queue.

        Args:
            queue_dir: Directory to store queue files
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.requests_dir = self.queue_dir / "requests"
        self.decisions_dir = self.queue_dir / "decisions"
        self.requests_dir.mkdir(exist_ok=True)
        self.decisions_dir.mkdir(exist_ok=True)

    def submit(self, request: ReviewRequest) -> str:
        """Submit a review request."""
        request_file = self.requests_dir / f"{request.id}.json"
        with open(request_file, "w") as f:
            json.dump(
                {
                    "id": request.id,
                    "node": request.node,
                    "payload": request.payload,
                    "suggestions": request.suggestions,
                    "expires_at": request.expires_at.isoformat() if request.expires_at else None,
                    "submitted_at": datetime.utcnow().isoformat(),
                },
                f,
                indent=2,
            )
        return request.id

    def get_decision(self, request_id: str, timeout: Optional[float] = None) -> Optional[ReviewDecision]:
        """Get decision for a request (blocking)."""
        decision_file = self.decisions_dir / f"{request_id}.json"

        # Poll for decision
        start_time = time.time()
        while True:
            if decision_file.exists():
                with open(decision_file, "r") as f:
                    decision_data = json.load(f)

                return ReviewDecision(
                    id=decision_data["id"],
                    approved=decision_data["approved"],
                    edited_output=decision_data.get("edited_output"),
                    comment=decision_data.get("comment"),
                    reviewer=decision_data.get("reviewer"),
                )

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None

            # Poll interval
            time.sleep(0.5)

    def list_pending(self) -> list[ReviewRequest]:
        """List all pending requests."""
        requests = []
        for request_file in self.requests_dir.glob("*.json"):
            with open(request_file, "r") as f:
                data = json.load(f)

            # Check if decision exists
            decision_file = self.decisions_dir / f"{data['id']}.json"
            if not decision_file.exists():
                request = ReviewRequest(
                    id=data["id"],
                    node=data["node"],
                    payload=data["payload"],
                    suggestions=data.get("suggestions", {}),
                    expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                )
                requests.append(request)

        return requests


class InMemoryQueue(HITLQueue):
    """In-memory HITL queue (for testing)."""

    def __init__(self):
        self._requests: Dict[str, ReviewRequest] = {}
        self._decisions: Dict[str, ReviewDecision] = {}

    def submit(self, request: ReviewRequest) -> str:
        """Submit a review request."""
        self._requests[request.id] = request
        return request.id

    def get_decision(self, request_id: str, timeout: Optional[float] = None) -> Optional[ReviewDecision]:
        """Get decision for a request (blocking)."""
        start_time = time.time()
        while True:
            if request_id in self._decisions:
                return self._decisions[request_id]

            if timeout and (time.time() - start_time) > timeout:
                return None

            time.sleep(0.1)

    def add_decision(self, request_id: str, decision: ReviewDecision) -> None:
        """Manually add a decision (for testing)."""
        self._decisions[request_id] = decision


class PostgresQueue(HITLQueue):
    """PostgreSQL-based HITL queue."""

    def __init__(self, connection_string: str, table_prefix: str = "hitl_"):
        """
        Initialize PostgreSQL queue.

        Args:
            connection_string: PostgreSQL connection string
            table_prefix: Table name prefix
        """
        self.connection_string = connection_string
        self.table_prefix = table_prefix
        self._connection = None

    def _get_connection(self):
        """Lazy load PostgreSQL connection."""
        if self._connection is None:
            try:
                import psycopg2

                self._connection = psycopg2.connect(self.connection_string)
                self._create_tables()
            except ImportError:
                raise ImportError("PostgreSQL support requires psycopg2-binary. Install with: pip install psycopg2-binary")
        return self._connection

    def _create_tables(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_prefix}requests (
                id VARCHAR(255) PRIMARY KEY,
                node VARCHAR(255),
                payload JSONB,
                suggestions JSONB,
                expires_at TIMESTAMP,
                submitted_at TIMESTAMP DEFAULT NOW()
            )
            """
        )

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_prefix}decisions (
                request_id VARCHAR(255) PRIMARY KEY,
                approved BOOLEAN,
                edited_output JSONB,
                comment TEXT,
                reviewer VARCHAR(255),
                decided_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (request_id) REFERENCES {self.table_prefix}requests(id)
            )
            """
        )

        conn.commit()

    def submit(self, request: ReviewRequest) -> str:
        """Submit a review request."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute(
            f"""
            INSERT INTO {self.table_prefix}requests (id, node, payload, suggestions, expires_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                request.id,
                request.node,
                json.dumps(request.payload),
                json.dumps(request.suggestions),
                request.expires_at,
            ),
        )

        conn.commit()
        return request.id

    def get_decision(self, request_id: str, timeout: Optional[float] = None) -> Optional[ReviewDecision]:
        """Get decision for a request (blocking)."""
        conn = self._get_connection()
        cur = conn.cursor()

        start_time = time.time()
        while True:
            cur.execute(
                f"""
                SELECT approved, edited_output, comment, reviewer
                FROM {self.table_prefix}decisions
                WHERE request_id = %s
                """,
                (request_id,),
            )

            row = cur.fetchone()
            if row:
                return ReviewDecision(
                    id=request_id,
                    approved=row[0],
                    edited_output=json.loads(row[1]) if row[1] else {},
                    comment=row[2],
                    reviewer=row[3],
                )

            if timeout and (time.time() - start_time) > timeout:
                return None

            time.sleep(0.5)


class RedisQueue(HITLQueue):
    """Redis-based HITL queue."""

    def __init__(self, redis_url: str = "redis://localhost:6379", key_prefix: str = "hitl:"):
        """
        Initialize Redis queue.

        Args:
            redis_url: Redis connection URL
            key_prefix: Key prefix for Redis keys
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None

    def _get_redis(self):
        """Lazy load Redis connection."""
        if self._redis is None:
            try:
                import redis

                self._redis = redis.from_url(self.redis_url)
            except ImportError:
                raise ImportError("Redis support requires redis. Install with: pip install redis")
        return self._redis

    def submit(self, request: ReviewRequest) -> str:
        """Submit a review request."""
        redis_client = self._get_redis()
        key = f"{self.key_prefix}request:{request.id}"

        data = {
            "id": request.id,
            "node": request.node,
            "payload": json.dumps(request.payload),
            "suggestions": json.dumps(request.suggestions),
            "expires_at": request.expires_at.isoformat() if request.expires_at else None,
            "submitted_at": datetime.utcnow().isoformat(),
        }

        redis_client.hset(key, mapping=data)

        # Add to pending set
        redis_client.sadd(f"{self.key_prefix}pending", request.id)

        return request.id

    def get_decision(self, request_id: str, timeout: Optional[float] = None) -> Optional[ReviewDecision]:
        """Get decision for a request (blocking)."""
        redis_client = self._get_redis()
        decision_key = f"{self.key_prefix}decision:{request_id}"

        start_time = time.time()
        while True:
            decision_data = redis_client.hgetall(decision_key)
            if decision_data:
                return ReviewDecision(
                    id=request_id,
                    approved=decision_data.get(b"approved", b"false").decode() == "true",
                    edited_output=json.loads(decision_data[b"edited_output"].decode())
                    if decision_data.get(b"edited_output")
                    else {},
                    comment=decision_data.get(b"comment", b"").decode() if decision_data.get(b"comment") else None,
                    reviewer=decision_data.get(b"reviewer", b"").decode() if decision_data.get(b"reviewer") else None,
                )

            if timeout and (time.time() - start_time) > timeout:
                return None

            time.sleep(0.5)


class SlackQueue(HITLQueue):
    """Slack-based HITL queue (notifications only, decisions via API/webhook)."""

    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        """
        Initialize Slack queue.

        Args:
            webhook_url: Slack webhook URL
            channel: Optional Slack channel override
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self._pending_decisions: Dict[str, ReviewDecision] = {}  # In-memory for now

    def submit(self, request: ReviewRequest) -> str:
        """Submit a review request and send Slack notification."""
        try:
            import requests

            # Send Slack notification
            message = {
                "text": f"Human Review Request: {request.node}",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Review Request*\n*Node:* {request.node}\n*Request ID:* `{request.id}`",
                        },
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Payload:*\n```{json.dumps(request.payload, indent=2)[:500]}```",
                            }
                        ],
                    },
                ],
            }

            if self.channel:
                message["channel"] = self.channel

            requests.post(self.webhook_url, json=message)

            return request.id
        except ImportError:
            raise ImportError("Slack support requires requests. Install with: pip install requests")

    def get_decision(self, request_id: str, timeout: Optional[float] = None) -> Optional[ReviewDecision]:
        """Get decision (requires external decision API/webhook)."""
        # For now, check in-memory cache
        # In production, this would poll a decision API or webhook endpoint
        start_time = time.time()
        while True:
            if request_id in self._pending_decisions:
                return self._pending_decisions[request_id]

            if timeout and (time.time() - start_time) > timeout:
                return None

            time.sleep(1.0)

    def record_decision(self, request_id: str, decision: ReviewDecision) -> None:
        """Record a decision (called by webhook or external system)."""
        self._pending_decisions[request_id] = decision
