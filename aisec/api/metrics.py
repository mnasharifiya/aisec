"""
AISec Prometheus metrics.

Exposes real-time security telemetry in Prometheus format
for ingestion by Grafana, PagerDuty, Datadog, and any
OpenMetrics-compatible monitoring system.

Metrics exposed:

    aisec_events_total
        Counter — total events analysed, labelled by decision,
        scenario, and agent_id. Primary operational metric.

    aisec_risk_score_histogram
        Histogram — distribution of risk scores across all events.
        Used to detect risk score drift and calibration issues.

    aisec_temporal_alerts_total
        Counter — temporal threat alerts fired, labelled by threat type
        and severity. Critical for detecting attack campaigns.

    aisec_audit_chain_status
        Gauge — 1.0 if chain intact, 0.0 if broken.
        Alert on this immediately — a broken chain is a security incident.

    aisec_blocked_events_total
        Counter — events blocked, labelled by rule_id.
        Shows which rules are firing most frequently.

    aisec_api_request_duration_seconds
        Histogram — API endpoint latency.
        Used to detect performance degradation under load.

    aisec_agents_active
        Gauge — number of distinct agent IDs seen in last 5 minutes.
        Spikes indicate new agent deployments or impersonation attempts.

Usage in Grafana:
    rate(aisec_events_total[5m])                 — events per second
    aisec_audit_chain_status == 0                — chain broken alert
    rate(aisec_temporal_alerts_total[1m]) > 0    — active attack alert
    histogram_quantile(0.95, aisec_risk_score_histogram) — p95 risk score
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from typing import Any

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

from aisec.utils.logger import get_logger

log = get_logger("aisec.api.metrics")


# ── Metric definitions ────────────────────────────────────────────────────────


def _create_registry() -> Any:
    """Create a fresh Prometheus registry for AISec metrics."""
    if not _PROMETHEUS_AVAILABLE:
        return None
    return CollectorRegistry()


class AISeCMetrics:
    """
    AISec Prometheus metrics collector.

    Thread-safe. All metric updates are atomic operations
    on Prometheus Counter/Gauge/Histogram objects which are
    internally thread-safe.

    Usage:
        metrics = AISeCMetrics()
        metrics.record_event(
            decision="BLOCK",
            scenario="trading_ai",
            agent_id="trading_bot_v1",
            risk_score=0.94,
            rule_hits=["TRADING-001"],
        )
        # Expose at /metrics endpoint
        output = metrics.generate_output()
    """

    def __init__(self) -> None:
        if not _PROMETHEUS_AVAILABLE:
            log.warning(
                "prometheus_client_not_installed",
                detail="Metrics endpoint will return empty. "
                "Install: pip install prometheus-client",
            )
            self._available = False
            return

        self._available = True
        self._registry = _create_registry()

        # ── Counters ──────────────────────────────────────────────────────────

        self.events_total = Counter(
            "aisec_events_total",
            "Total AI agent actions analysed by AISec",
            labelnames=["decision", "scenario", "agent_id"],
            registry=self._registry,
        )

        self.temporal_alerts_total = Counter(
            "aisec_temporal_alerts_total",
            "Temporal threat alerts fired by the anomaly detector",
            labelnames=["threat", "severity"],
            registry=self._registry,
        )

        self.blocked_by_rule_total = Counter(
            "aisec_blocked_by_rule_total",
            "Events blocked by specific rule IDs",
            labelnames=["rule_id", "scenario"],
            registry=self._registry,
        )

        self.api_requests_total = Counter(
            "aisec_api_requests_total",
            "Total API requests received",
            labelnames=["endpoint", "status_code"],
            registry=self._registry,
        )

        self.webhook_deliveries_total = Counter(
            "aisec_webhook_deliveries_total",
            "Webhook delivery attempts",
            labelnames=["outcome"],  # success, failure, retry
            registry=self._registry,
        )

        # ── Histograms ────────────────────────────────────────────────────────

        self.risk_score_histogram = Histogram(
            "aisec_risk_score",
            "Distribution of risk scores across all analysed events",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self._registry,
        )

        self.api_latency_histogram = Histogram(
            "aisec_api_request_duration_seconds",
            "API endpoint request latency in seconds",
            labelnames=["endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self._registry,
        )

        self.analysis_latency_histogram = Histogram(
            "aisec_analysis_duration_seconds",
            "End-to-end analysis pipeline latency in seconds",
            buckets=[0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
            registry=self._registry,
        )

        # ── Gauges ────────────────────────────────────────────────────────────

        self.audit_chain_status = Gauge(
            "aisec_audit_chain_status",
            "Audit log hash chain integrity: 1=intact, 0=broken",
            registry=self._registry,
        )
        # Start as intact — updated on each verify
        self.audit_chain_status.set(1.0)

        self.audit_entries_total = Gauge(
            "aisec_audit_entries_total",
            "Total number of entries in the audit log",
            registry=self._registry,
        )

        self.queue_pending_gauge = Gauge(
            "aisec_soc_queue_pending",
            "Number of events currently pending SOC analyst review",
            registry=self._registry,
        )

        self.agents_seen_gauge = Gauge(
            "aisec_agents_seen_total",
            "Total distinct agent IDs seen since startup",
            registry=self._registry,
        )

        # Internal tracking
        self._agents_seen: set[str] = set()
        self._lock: threading.Lock = threading.Lock()

        log.info("prometheus_metrics_initialized")

    # ── Recording methods ─────────────────────────────────────────────────────

    def record_event(
        self,
        decision: str,
        scenario: str,
        agent_id: str,
        risk_score: float,
        rule_hits: list[str],
        latency_s: float = 0.0,
    ) -> None:
        """
        Record metrics for a single analysed event.

        Args:
            decision:   The enforcement decision (ALLOW/BLOCK/etc).
            scenario:   The threat scenario (trading_ai/urban_ai).
            agent_id:   The agent ID that submitted the event.
            risk_score: The computed risk score in [0.0, 1.0].
            rule_hits:  List of rule IDs that fired.
            latency_s:  Analysis pipeline latency in seconds.
        """
        if not self._available:
            return

        # Sanitise agent_id for label safety — Prometheus labels
        # must not contain characters that break the exposition format
        safe_agent = agent_id[:32].replace('"', "").replace("\n", "")

        self.events_total.labels(
            decision=decision,
            scenario=scenario,
            agent_id=safe_agent,
        ).inc()

        self.risk_score_histogram.observe(risk_score)

        for rule_id in rule_hits:
            self.blocked_by_rule_total.labels(
                rule_id=rule_id,
                scenario=scenario,
            ).inc()

        if latency_s > 0:
            self.analysis_latency_histogram.observe(latency_s)

        with self._lock:
            self._agents_seen.add(agent_id)
            self.agents_seen_gauge.set(len(self._agents_seen))

    def record_temporal_alert(
        self,
        threat: str,
        severity: str,
    ) -> None:
        """Record a temporal anomaly alert."""
        if not self._available:
            return
        self.temporal_alerts_total.labels(
            threat=threat,
            severity=severity,
        ).inc()

    def record_api_request(
        self,
        endpoint: str,
        status_code: int,
        latency_s: float,
    ) -> None:
        """Record an API request with latency."""
        if not self._available:
            return
        self.api_requests_total.labels(
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        self.api_latency_histogram.labels(endpoint=endpoint).observe(latency_s)

    def record_webhook_delivery(self, outcome: str) -> None:
        """Record a webhook delivery attempt outcome."""
        if not self._available:
            return
        self.webhook_deliveries_total.labels(outcome=outcome).inc()

    def update_audit_status(
        self,
        chain_intact: bool,
        entry_count: int,
    ) -> None:
        """Update audit chain status gauges."""
        if not self._available:
            return
        self.audit_chain_status.set(1.0 if chain_intact else 0.0)
        self.audit_entries_total.set(entry_count)

    def update_queue_size(self, pending: int) -> None:
        """Update the SOC queue pending count gauge."""
        if not self._available:
            return
        self.queue_pending_gauge.set(pending)

    def generate_output(self) -> tuple[bytes, str]:
        """
        Generate Prometheus exposition format output.

        Returns:
            (content_bytes, content_type) tuple for HTTP response.
        """
        if not self._available:
            return (
                b"# prometheus_client not installed\n",
                "text/plain; version=0.0.4",
            )
        return (
            generate_latest(self._registry),
            CONTENT_TYPE_LATEST,
        )

    @property
    def available(self) -> bool:
        """True if prometheus_client is installed."""
        return self._available
