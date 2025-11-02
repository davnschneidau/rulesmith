"""Alert system for rule monitoring."""

import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from rulesmith.governance.audit import audit_logger


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertRule:
    """Defines an alert condition."""

    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        description: Optional[str] = None,
        cooldown_minutes: int = 60,
    ):
        """
        Initialize an alert rule.

        Args:
            name: Alert rule name
            condition: Function that evaluates to True if alert should fire
                      Takes metrics dictionary as input
            severity: Alert severity level
            description: Optional description
            cooldown_minutes: Minutes between repeated alerts (prevents spam)
        """
        self.name = name
        self.condition = condition
        self.severity = severity
        self.description = description
        self.cooldown_minutes = cooldown_minutes
        self.last_triggered: Optional[datetime] = None

    def should_fire(self, metrics: Dict[str, Any]) -> bool:
        """Check if alert should fire."""
        try:
            should_fire = self.condition(metrics)

            # Check cooldown
            if should_fire and self.last_triggered:
                elapsed = (datetime.utcnow() - self.last_triggered).total_seconds() / 60
                if elapsed < self.cooldown_minutes:
                    return False  # Still in cooldown

            if should_fire:
                self.last_triggered = datetime.utcnow()

            return should_fire
        except Exception:
            # If condition evaluation fails, don't fire alert
            return False


class Alert:
    """Represents an active alert."""

    def __init__(
        self,
        alert_rule_name: str,
        severity: AlertSeverity,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.alert_id = f"{alert_rule_name}:{datetime.utcnow().isoformat()}"
        self.alert_rule_name = alert_rule_name
        self.severity = severity
        self.message = message
        self.metrics = metrics or {}
        self.metadata = metadata or {}
        self.status = AlertStatus.ACTIVE
        self.created_at = datetime.utcnow()
        self.acknowledged_at: Optional[datetime] = None
        self.resolved_at: Optional[datetime] = None

    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()

    def resolve(self) -> None:
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_rule_name": self.alert_rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class AlertChannel:
    """Base class for alert channels."""

    def send(self, alert: Alert) -> bool:
        """
        Send an alert.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        raise NotImplementedError


class LoggingChannel(AlertChannel):
    """Alert channel that logs to standard logging."""

    def send(self, alert: Alert) -> bool:
        """Send alert via logging."""
        import logging

        logger = logging.getLogger("rulesmith.alerts")
        level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)

        logger.log(level, f"Alert [{alert.severity.value}]: {alert.message}")
        return True


class WebhookChannel(AlertChannel):
    """Alert channel that sends to a webhook URL."""

    def __init__(self, webhook_url: str, timeout: int = 10):
        """
        Initialize webhook channel.

        Args:
            webhook_url: Webhook URL
            timeout: Request timeout in seconds
        """
        self.webhook_url = webhook_url
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import requests

            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                timeout=self.timeout,
            )
            return response.status_code < 400
        except Exception:
            return False


class EmailChannel(AlertChannel):
    """Alert channel that sends email via SMTP."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        from_email: str,
        to_emails: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize email channel.

        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            from_email: From email address
            to_emails: List of recipient email addresses
            username: Optional SMTP username
            password: Optional SMTP password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.from_email = from_email
        self.to_emails = to_emails
        self.username = username
        self.password = password

    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            msg = MIMEText(f"{alert.message}\n\nSeverity: {alert.severity.value}\n\n{alert.to_dict()}")
            msg["Subject"] = f"Rulesmith Alert [{alert.severity.value}]: {alert.alert_rule_name}"
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.username and self.password:
                server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            return True
        except Exception:
            return False


class AlertManager:
    """
    Manages alert rules and sends notifications.
    
    Evaluates alert conditions against metrics and sends
    alerts via configured channels.
    """

    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.channels: List[AlertChannel] = [LoggingChannel()]  # Default to logging
        self.active_alerts: Dict[str, Alert] = {}

    def register_alert_rule(self, alert_rule: AlertRule) -> None:
        """Register an alert rule."""
        self.alert_rules[alert_rule.name] = alert_rule

    def add_channel(self, channel: AlertChannel) -> None:
        """Add an alert channel."""
        self.channels.append(channel)

    def evaluate(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all alert rules against metrics.

        Args:
            metrics: Current metrics dictionary

        Returns:
            List of alerts that fired
        """
        fired_alerts = []

        for rule_name, rule in self.alert_rules.items():
            if rule.should_fire(metrics):
                alert = Alert(
                    alert_rule_name=rule_name,
                    severity=rule.severity,
                    message=rule.description or f"Alert {rule_name} triggered",
                    metrics=metrics,
                )

                # Send alert via all channels
                for channel in self.channels:
                    try:
                        channel.send(alert)
                    except Exception:
                        pass  # Continue with other channels if one fails

                # Store active alert
                self.active_alerts[alert.alert_id] = alert
                fired_alerts.append(alert)

                # Log alert
                audit_logger.log(
                    action="alert_fired",
                    entity_type="alert",
                    entity_id=alert.alert_id,
                    metadata={
                        "alert_rule": rule_name,
                        "severity": rule.severity.value,
                    },
                )

        return fired_alerts

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = [
            alert for alert in self.active_alerts.values() if alert.status == AlertStatus.ACTIVE
        ]

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        alert = self.active_alerts.get(alert_id)
        if alert:
            alert.acknowledge()
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        alert = self.active_alerts.get(alert_id)
        if alert:
            alert.resolve()
            return True
        return False


# Global alert manager instance
alert_manager = AlertManager()

