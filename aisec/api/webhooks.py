"""
AISec webhook delivery system.

Sends real-time security alerts to configured HTTP endpoints
when dangerous AI agent actions are detected.

Security:
    - Every webhook payload is HMAC-SHA256 signed.
    - Receivers can verify the signature to confirm authenticity.
    - Retry with exponential backoff on delivery failure.
    - Sensitive payload data is never included in webhook bodies.
    - Delivery failures are logged but never crash AISec.

Payload format:
    {
        "event_id":    "uuid",
        "agent_id":    "trading_bot_v1",
        "action_type": "manipulate_news_feed",
        "decision":    "BLOCK",
        "risk_score":  0.94,
        "rule_hits":   ["TRADING-002"],
        "scenario":    "trading_ai",
        "timestamp":   "2025-05-03T22:14:05+00:00",
        "aisec_version": "1.2.0"
    }

Signature header:
    X-AISec-Signature: sha256=<hmac_hex>

Verification (receiver side):
    import hmac, hashlib
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    received = request.headers["X-AISec-Signature"].removeprefix("sha256=")
    assert hmac.compare_digest(expected, received)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from aisec.utils.logger import get_logger
from aisec.utils.time import now_utc

log = get_logger("aisec.api.webhooks")

_VERSION = "1.2.0"


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class WebhookConfig:
    """
    Configuration for a single webhook endpoint.

    Attributes:
        url:             The HTTPS endpoint to deliver events to.
        secret:          HMAC-SHA256 signing secret.
                         Minimum 32 characters. Never log this.
        events:          Event types to deliver.
                         Empty list = deliver all event types.
        max_retries:     Maximum delivery attempts before giving up.
        timeout_seconds: HTTP request timeout.
        retry_delay_s:   Initial retry delay (doubles each attempt).
        enabled:         Set to False to temporarily disable.
    """

    url: str
    secret: str
    events: list[str] = field(default_factory=list)
    max_retries: int = 3
    timeout_seconds: float = 5.0
    retry_delay_s: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.url.startswith(("https://", "http://")):
            raise ValueError(
                f"Webhook URL must start with https:// or http://, "
                f"got: {self.url[:30]}"
            )
        if len(self.secret) < 32:
            raise ValueError("Webhook secret must be at least 32 characters long.")

    def should_deliver(self, event_type: str) -> bool:
        """Return True if this webhook should receive this event type."""
        if not self.enabled:
            return False
        if not self.events:
            return True  # Empty list = deliver all
        return event_type in self.events


# ── Webhook payload ───────────────────────────────────────────────────────────


@dataclass
class WebhookPayload:
    """
    A single webhook delivery payload.

    Contains only non-sensitive fields from the analysis result.
    Raw payloads and detailed rule configurations are never included.
    """

    event_id: str
    agent_id: str
    action_type: str
    decision: str
    risk_score: float
    rule_hits: list[str]
    scenario: str
    explanation: str
    blocked: bool
    requires_review: bool
    temporal_alerts: list[dict[str, Any]]
    timestamp: str = field(default_factory=now_utc)
    aisec_version: str = _VERSION
    event_type: str = "analysis_complete"

    def to_json(self) -> bytes:
        """Serialise to JSON bytes for HTTP delivery."""
        data = {
            "event_id": self.event_id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "decision": self.decision,
            "risk_score": round(self.risk_score, 4),
            "rule_hits": self.rule_hits,
            "scenario": self.scenario,
            "explanation": self.explanation[:500],
            "blocked": self.blocked,
            "requires_review": self.requires_review,
            "temporal_alerts": self.temporal_alerts,
            "timestamp": self.timestamp,
            "aisec_version": self.aisec_version,
            "event_type": self.event_type,
        }
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    def sign(self, secret: str) -> str:
        """
        Compute HMAC-SHA256 signature over the JSON payload.

        Returns:
            Hex signature string for the X-AISec-Signature header.
        """
        body = self.to_json()
        return hmac.new(
            secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()


# ── Webhook dispatcher ────────────────────────────────────────────────────────


class WebhookDispatcher:
    """
    Delivers webhook payloads to configured endpoints.

    Deliveries are made asynchronously in background threads
    so they never block the analysis pipeline.

    Thread safety:
        Each delivery runs in its own thread.
        The dispatcher itself is stateless between deliveries.

    Failure handling:
        Retries with exponential backoff up to max_retries.
        Failures are logged but never crash AISec.
        Metrics are updated on each delivery attempt.
    """

    def __init__(
        self,
        configs: list[WebhookConfig],
        metrics: Any = None,  # AISeCMetrics — optional
    ) -> None:
        """
        Args:
            configs: List of webhook endpoint configurations.
            metrics: Optional AISeCMetrics for delivery tracking.
        """
        self._configs = [c for c in configs if c.enabled]
        self._metrics = metrics

        log.info(
            "webhook_dispatcher_initialized",
            endpoint_count=len(self._configs),
            endpoints=[c.url for c in self._configs],
        )

    def dispatch(self, payload: WebhookPayload) -> None:
        """
        Dispatch a payload to all configured endpoints asynchronously.

        Each delivery runs in a background daemon thread.
        Returns immediately — delivery is fire-and-forget.

        Args:
            payload: The WebhookPayload to deliver.
        """
        for config in self._configs:
            if not config.should_deliver(payload.event_type):
                continue

            thread = threading.Thread(
                target=self._deliver_with_retry,
                args=(config, payload),
                daemon=True,  # Dies with the main process
                name=f"aisec-webhook-{payload.event_id[:8]}",
            )
            thread.start()

    def dispatch_blocked(self, payload: WebhookPayload) -> None:
        """
        Dispatch only to endpoints configured for block events.
        Synchronous — waits for first delivery attempt.
        Used for critical block events that must not be delayed.
        """
        payload.event_type = "action_blocked"
        for config in self._configs:
            if config.should_deliver("action_blocked"):
                self._deliver_with_retry(config, payload)

    # ── Private delivery logic ────────────────────────────────────────────────

    def _deliver_with_retry(
        self,
        config: WebhookConfig,
        payload: WebhookPayload,
    ) -> None:
        """
        Attempt delivery with exponential backoff retry.

        Args:
            config:  Webhook endpoint configuration.
            payload: Payload to deliver.
        """
        body = payload.to_json()
        signature = payload.sign(config.secret)
        delay = config.retry_delay_s

        for attempt in range(1, config.max_retries + 1):
            success = self._attempt_delivery(
                url=config.url,
                body=body,
                signature=signature,
                timeout=config.timeout_seconds,
                attempt=attempt,
            )

            if success:
                if self._metrics:
                    self._metrics.record_webhook_delivery("success")
                return

            if attempt < config.max_retries:
                log.warning(
                    "webhook_delivery_retry",
                    url=config.url,
                    attempt=attempt,
                    next_delay_s=delay,
                    event_id=payload.event_id,
                )
                time.sleep(delay)
                delay *= 2.0  # Exponential backoff

        # All retries exhausted
        log.error(
            "webhook_delivery_failed",
            url=config.url,
            max_retries=config.max_retries,
            event_id=payload.event_id,
        )
        if self._metrics:
            self._metrics.record_webhook_delivery("failure")

    def _attempt_delivery(
        self,
        url: str,
        body: bytes,
        signature: str,
        timeout: float,
        attempt: int,
    ) -> bool:
        """
        Make a single HTTP POST delivery attempt.

        Args:
            url:       Target URL.
            body:      JSON payload bytes.
            signature: HMAC signature hex string.
            timeout:   Request timeout in seconds.
            attempt:   Attempt number for logging.

        Returns:
            True if delivery succeeded (2xx response), False otherwise.
        """
        try:
            request = Request(
                url=url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": f"AISec/{_VERSION}",
                    "X-AISec-Signature": f"sha256={signature}",
                    "X-AISec-Version": _VERSION,
                    "X-AISec-Event": "security_alert",
                },
            )

            with urlopen(request, timeout=timeout) as response:
                status = response.status
                if 200 <= status < 300:
                    log.info(
                        "webhook_delivered",
                        url=url,
                        status=status,
                        attempt=attempt,
                    )
                    return True
                else:
                    log.warning(
                        "webhook_non_2xx",
                        url=url,
                        status=status,
                        attempt=attempt,
                    )
                    return False

        except HTTPError as exc:
            log.warning(
                "webhook_http_error",
                url=url,
                status=exc.code,
                attempt=attempt,
            )
            return False
        except URLError as exc:
            log.warning(
                "webhook_connection_error",
                url=url,
                reason=str(exc.reason)[:100],
                attempt=attempt,
            )
            return False
        except Exception as exc:
            log.error(
                "webhook_unexpected_error",
                url=url,
                exc_type=type(exc).__name__,
                detail=str(exc)[:100],
                attempt=attempt,
            )
            return False
