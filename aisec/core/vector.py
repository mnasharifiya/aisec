"""
AISec feature vector builder.

Transforms a raw Event into a normalised 8-dimensional
FeatureVector suitable for the risk scorer.

Each dimension is computed independently and independently
validated to be in [0.0, 1.0] before the vector is assembled.

Paper reference:
    Section 8 — Analysis Model.
    x = [a, k, f, p, d, n, s, u]
"""

from __future__ import annotations

# ── Trading AI keyword risk vocabulary ────────────────────────────────────────
#
# Words that appear in dangerous trading actions.
# Scored by severity: critical=1.0, high=0.8, medium=0.6, low=0.3

TRADING_RISK_KEYWORDS: dict[str, float] = {
    # Critical — immediate threat
    "manipulate":      1.0,
    "override":        1.0,
    "inject":          1.0,
    "launder":         1.0,
    "fake":            1.0,
    # High risk
    "large_trade":     0.8,
    "bulk":            0.8,
    "mass":            0.8,
    "unrestricted":    0.8,
    # Medium risk
    "after_hours":     0.6,
    "bypass":          0.6,
    "unlimited":       0.6,
    # Lower risk — worth noting
    "execute":         0.3,
    "sell":            0.2,
    "buy":             0.1,
}

# ── Urban AI keyword risk vocabulary ──────────────────────────────────────────

URBAN_RISK_KEYWORDS: dict[str, float] = {
    # Critical
    "curfew":          1.0,
    "lockdown":        1.0,
    "shutdown":        1.0,
    "cut_power":       1.0,
    "blackout":        1.0,
    # High risk
    "restrict":        0.8,
    "override":        0.8,
    "disable":         0.8,
    "mass":            0.8,
    # Medium risk
    "redirect":        0.6,
    "reroute":         0.6,
    "emergency":       0.5,
    # Lower risk
    "adjust":          0.2,
    "read":            0.0,
    "monitor":         0.0,
}

# ── Action type encoding ──────────────────────────────────────────────────────
#
# Maps action type strings to a normalised risk value in [0.0, 1.0].
# Actions not listed default to 0.1 (unknown = slightly elevated).

ACTION_TYPE_ENCODING: dict[str, float] = {
    # Safe reads
    "read_sensor":          0.0,
    "read_market_data":     0.0,
    "log_event":            0.0,
    "get_status":           0.0,
    # Low risk writes
    "minor_trade":          0.1,
    "adjust_traffic_flow":  0.2,
    "update_config":        0.3,
    # Medium risk
    "execute_trade":        0.5,
    "mass_traffic_redirect":0.6,
    "override_traffic_system": 0.7,
    # High risk
    "execute_large_trade":  0.8,
    "shutdown_power_grid":  0.9,
    "set_curfew":           0.9,
    # Critical
    "manipulate_news_feed": 1.0,
    "override_risk_limit":  1.0,
    "cut_power":            1.0,
    "lockdown_zone":        1.0,
}

# ── Sensitive targets ─────────────────────────────────────────────────────────

SENSITIVE_TARGETS: set[str] = {
    "emergency_dispatch",
    "police_network",
    "fire_service",
    "ambulance_routing",
    "hospital_power",
    "power_grid",
    "water_treatment",
    "market_feed",
    "risk_management_system",
    "audit_log",
    "aisec_config",
}

# ── Network-using actions ─────────────────────────────────────────────────────

NETWORK_ACTIONS: set[str] = {
    "execute_trade",
    "execute_large_trade",
    "manipulate_news_feed",
    "inject_market_data",
    "publish_fake_news",
    "mass_traffic_redirect",
    "override_traffic_system",
    "shutdown_power_grid",
    "set_curfew",
}

# ── File-touching actions ─────────────────────────────────────────────────────

FILE_ACTIONS: set[str] = {
    "update_config",
    "modify_risk_parameters",
    "alter_market_feed",
    "write_audit_log",
}

# ── Privileged actions ────────────────────────────────────────────────────────

PRIVILEGED_ACTIONS: set[str] = {
    "override_risk_limit",
    "modify_risk_parameters",
    "shutdown_power_grid",
    "cut_power",
    "lockdown_zone",
    "set_curfew",
    "restrict_movement",
    "disable_zone_power",
}


from aisec.storage.models import Event, FeatureVector, Scenario


class FeatureVectorBuilder:
    """
    Transforms a raw Event into a normalised FeatureVector.

    Each dimension is computed from the event fields and payload,
    then clamped to [0.0, 1.0] before assembly.

    Usage:
        builder = FeatureVectorBuilder()
        fv = builder.build(event)
        # fv.vector is ready for RiskScorer.score()
    """

    def build(self, event: Event) -> FeatureVector:
        """
        Build and return a FeatureVector from the given Event.

        Args:
            event: A normalised Event from the interceptor.

        Returns:
            FeatureVector with all 8 dimensions populated.
        """
        vector = [
            self._action_type_encoding(event),
            self._keyword_risk_score(event),
            self._frequency_score(event),
            self._api_call_flag(event),
            self._file_access_flag(event),
            self._network_access_flag(event),
            self._sensitive_path_flag(event),
            self._privilege_flag(event),
        ]

        # Clamp all values to [0.0, 1.0] as a safety guarantee
        vector = [max(0.0, min(1.0, v)) for v in vector]

        return FeatureVector(event_id=event.event_id, vector=vector)

    # ── Dimension extractors ──────────────────────────────────────────────────

    def _action_type_encoding(self, event: Event) -> float:
        """Map the action type to its pre-defined risk encoding."""
        return ACTION_TYPE_ENCODING.get(event.action_type, 0.1)

    def _keyword_risk_score(self, event: Event) -> float:
        """
        Compute a keyword risk score from the action type and target.

        Scans both the action type and target strings for risk keywords.
        Returns the maximum score found (not cumulative — one bad keyword
        is enough to produce a high score).
        """
        keywords = (
            TRADING_RISK_KEYWORDS
            if event.scenario == Scenario.TRADING_AI
            else URBAN_RISK_KEYWORDS
        )

        text = f"{event.action_type} {event.target}".lower().replace("_", " ")
        score = 0.0
        for keyword, weight in keywords.items():
            if keyword.replace("_", " ") in text:
                score = max(score, weight)
        return score

    def _frequency_score(self, event: Event) -> float:
        """
        Return a burst/frequency risk score from the event payload.

        The interceptor or agent wrapper sets 'burst_rate' in the payload
        when it detects rapid repeated actions. Normalised to [0.0, 1.0]
        where 1.0 = 100 or more actions per second.
        """
        burst = event.raw_payload.get("burst_rate", 0.0)
        try:
            burst = float(burst)
        except (TypeError, ValueError):
            burst = 0.0
        return min(burst / 100.0, 1.0)

    def _api_call_flag(self, event: Event) -> float:
        """Return 1.0 if this action calls an external API."""
        return 1.0 if event.action_type in NETWORK_ACTIONS else 0.0

    def _file_access_flag(self, event: Event) -> float:
        """Return 1.0 if this action touches the filesystem."""
        return 1.0 if event.action_type in FILE_ACTIONS else 0.0

    def _network_access_flag(self, event: Event) -> float:
        """
        Return 1.0 if this action uses network access.

        Slightly broader than _api_call_flag — includes any action
        with network_access=True in its payload even if not listed
        in NETWORK_ACTIONS.
        """
        if event.action_type in NETWORK_ACTIONS:
            return 1.0
        return 1.0 if event.raw_payload.get("network_access", False) else 0.0

    def _sensitive_path_flag(self, event: Event) -> float:
        """Return 1.0 if the target is a protected sensitive resource."""
        return 1.0 if event.target in SENSITIVE_TARGETS else 0.0

    def _privilege_flag(self, event: Event) -> float:
        """Return 1.0 if this action requires elevated privileges."""
        return 1.0 if event.action_type in PRIVILEGED_ACTIONS else 0.0