"""
AISec Deployment Study — Data Schemas.

Schema-first definitions for the real-agent deployment study.

Design principle:
    If a field is not in the schema, it does not exist.

This module defines:
    - experimental groups
    - ground-truth labels
    - threat labels
    - baseline modes
    - task definitions
    - study event records
    - study metrics
    - reproducibility metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

SCHEMA_VERSION = "1.0"


# =============================================================================
# Helpers
# =============================================================================


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_probability(name: str, value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number or None")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _validate_non_negative_number(name: str, value: int | float) -> None:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_iso_timestamp(name: str, value: str) -> None:
    _require_non_empty(name, value)

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp") from exc

    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include timezone information")


def _enum_value(value: Enum | str) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


# =============================================================================
# Experimental groups and labels
# =============================================================================


class ExperimentGroup(str, Enum):
    """
    Deployment-study groups.

    A — benign tasks: false-positive measurement
    B — prompt injection: injection detection measurement
    C — risky tool use: enforcement measurement
    D — contextual or multi-action risk measurement
    N — no-tool-call, malformed, or ambiguity-control measurement
    """

    A_BENIGN = "A"
    B_INJECTION = "B"
    C_RISKY = "C"
    D_MULTIAGENT = "D"
    N_OPERATIONAL = "N"


class GroundTruth(str, Enum):
    """Task-level ground truth."""

    SAFE = "safe"
    MALICIOUS = "malicious"
    AMBIGUOUS = "ambiguous"


class ThreatLabel(str, Enum):
    """
    Detailed threat label.

    GroundTruth gives the task-level safety class.
    ThreatLabel explains what kind of safety/security condition exists.
    """

    SAFE = "safe"
    PROMPT_INJECTION = "prompt_injection"
    RISKY_TOOL_USE = "risky_tool_use"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"
    POLICY_VIOLATION = "policy_violation"
    BENIGN_HIGH_RISK = "benign_high_risk"
    CONTEXTUAL_RISK = "contextual_risk"
    MALFORMED_OR_NO_TOOL = "malformed_or_no_tool"
    UNKNOWN = "unknown"


class BaselineMode(str, Enum):
    """
    Baseline comparison modes.

    baseline_none:
        No AISec monitoring. Raw agent execution only.

    baseline_static_rules:
        Static rule logic only.

    baseline_prompt_only:
        Prompt-injection detector only.

    baseline_llm_judge:
        LLM-based safety judge or action classifier.

    aisec_full:
        Full AISec pipeline.
    """

    NONE = "baseline_none"
    STATIC_RULES = "baseline_static_rules"
    PROMPT_ONLY = "baseline_prompt_only"
    LLM_JUDGE = "baseline_llm_judge"
    AISEC_FULL = "aisec_full"


class Framework(str, Enum):
    """AI agent framework used in the deployment study."""

    LANGCHAIN = "langchain"
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    OPENAI = "openai"
    CUSTOM = "custom"
    SIMULATED = "simulated"


class ModelProvider(str, Enum):
    """Model provider used in the deployment study."""

    GROQ = "groq"
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    SIMULATED = "simulated"


class StudyDecision(str, Enum):
    """
    Normalized study decision.

    NOT_EVALUATED is important for baseline_none, where AISec does not run.
    """

    NOT_EVALUATED = "NOT_EVALUATED"
    ALLOW = "ALLOW"
    PENDING_REVIEW = "PENDING_REVIEW"
    BLOCK = "BLOCK"
    ESCALATE = "ESCALATE"
    ERROR = "ERROR"


# =============================================================================
# Reproducibility metadata
# =============================================================================


@dataclass(frozen=True)
class StudyRunMetadata:
    """
    Reproducibility metadata for one study run.

    This makes the experiment traceable and paper-grade.
    """

    study_run_id: str
    aisec_version: str
    git_commit: str
    schema_version: str = SCHEMA_VERSION
    seed: int | None = None
    framework_version: str = "unknown"
    model_name: str = "unknown"
    started_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        _require_non_empty("study_run_id", self.study_run_id)
        _require_non_empty("aisec_version", self.aisec_version)
        _require_non_empty("git_commit", self.git_commit)
        _require_non_empty("schema_version", self.schema_version)
        _validate_iso_timestamp("started_at", self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_run_id": self.study_run_id,
            "aisec_version": self.aisec_version,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "framework_version": self.framework_version,
            "model_name": self.model_name,
            "started_at": self.started_at,
        }


# =============================================================================
# Task definition
# =============================================================================


@dataclass
class TaskDefinition:
    """
    A labeled task in the deployment study.

    Task-level labels are used to avoid manually labeling thousands of events.
    """

    task_id: str
    group: ExperimentGroup
    ground_truth: GroundTruth
    threat_label: ThreatLabel
    description: str
    expected_behavior: str
    actions: list[dict[str, Any]]

    framework: Framework = Framework.SIMULATED
    model_provider: ModelProvider = ModelProvider.SIMULATED

    task_author: str = "self"
    attack_source: str = "self_designed"
    is_external_attack: bool = False
    adversary_id: str = ""

    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.group = ExperimentGroup(self.group)
        self.ground_truth = GroundTruth(self.ground_truth)
        self.threat_label = ThreatLabel(self.threat_label)
        self.framework = Framework(self.framework)
        self.model_provider = ModelProvider(self.model_provider)

        _require_non_empty("task_id", self.task_id)
        _require_non_empty("description", self.description)
        _require_non_empty("expected_behavior", self.expected_behavior)

        if not isinstance(self.actions, list) or not self.actions:
            raise ValueError("actions must be a non-empty list")

        for index, action in enumerate(self.actions):
            if not isinstance(action, dict):
                raise TypeError(f"actions[{index}] must be a dictionary")
            if "action_type" not in action:
                raise ValueError(f"actions[{index}] is missing action_type")

        if self.ground_truth == GroundTruth.SAFE and self.threat_label not in {
            ThreatLabel.SAFE,
            ThreatLabel.BENIGN_HIGH_RISK,
        }:
            raise ValueError(
                "SAFE ground_truth must use SAFE or BENIGN_HIGH_RISK threat_label"
            )

        if self.ground_truth == GroundTruth.MALICIOUS and self.threat_label in {
            ThreatLabel.SAFE,
            ThreatLabel.BENIGN_HIGH_RISK,
            ThreatLabel.MALFORMED_OR_NO_TOOL,
        }:
            raise ValueError("MALICIOUS ground_truth must use a malicious threat_label")

        if self.ground_truth == GroundTruth.AMBIGUOUS and self.threat_label not in {
            ThreatLabel.UNKNOWN,
            ThreatLabel.MALFORMED_OR_NO_TOOL,
            ThreatLabel.CONTEXTUAL_RISK,
        }:
            raise ValueError(
                "AMBIGUOUS ground_truth must use UNKNOWN, MALFORMED_OR_NO_TOOL, "
                "or CONTEXTUAL_RISK threat_label"
            )

        if not isinstance(self.tags, list):
            raise TypeError("tags must be a list")

    def is_malicious(self) -> bool:
        return self.ground_truth == GroundTruth.MALICIOUS

    def is_benign(self) -> bool:
        return self.ground_truth == GroundTruth.SAFE

    def is_ambiguous(self) -> bool:
        return self.ground_truth == GroundTruth.AMBIGUOUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "group": self.group.value,
            "ground_truth": self.ground_truth.value,
            "threat_label": self.threat_label.value,
            "description": self.description,
            "expected_behavior": self.expected_behavior,
            "actions": self.actions,
            "framework": self.framework.value,
            "model_provider": self.model_provider.value,
            "task_author": self.task_author,
            "attack_source": self.attack_source,
            "is_external_attack": self.is_external_attack,
            "adversary_id": self.adversary_id,
            "notes": self.notes,
            "tags": self.tags,
        }


# =============================================================================
# Study event
# =============================================================================


@dataclass
class StudyEvent:
    """
    Atomic event record for the deployment study.

    One tool call or attempted action should produce one StudyEvent.
    """

    event_id: str
    study_run_id: str
    task_run_id: str
    task_id: str

    group: ExperimentGroup
    ground_truth: GroundTruth
    threat_label: ThreatLabel
    baseline_mode: BaselineMode

    agent_id: str
    framework: Framework
    model_provider: ModelProvider
    model_name: str

    action_type: str
    target: str
    payload_summary: str

    decision: StudyDecision = StudyDecision.NOT_EVALUATED
    risk_score: float | None = None
    rule_hits: list[str] = field(default_factory=list)

    was_blocked: bool = False
    was_intercepted: bool = False
    was_reviewed: bool = False

    injection_detected: bool = False
    injection_confidence: float | None = None

    correlation_alerts: int = 0
    temporal_alerts: int = 0
    safe_state_active: bool = False

    latency_ms: float = 0.0
    audit_entry_id: str | None = None

    schema_version: str = SCHEMA_VERSION
    aisec_version: str = "unknown"
    git_commit: str = "unknown"
    seed: int | None = None
    framework_version: str = "unknown"

    timestamp: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        self.group = ExperimentGroup(self.group)
        self.ground_truth = GroundTruth(self.ground_truth)
        self.threat_label = ThreatLabel(self.threat_label)
        self.baseline_mode = BaselineMode(self.baseline_mode)
        self.framework = Framework(self.framework)
        self.model_provider = ModelProvider(self.model_provider)
        self.decision = StudyDecision(self.decision)

        _require_non_empty("event_id", self.event_id)
        _require_non_empty("study_run_id", self.study_run_id)
        _require_non_empty("task_run_id", self.task_run_id)
        _require_non_empty("task_id", self.task_id)
        _require_non_empty("agent_id", self.agent_id)
        _require_non_empty("model_name", self.model_name)
        _require_non_empty("action_type", self.action_type)
        _require_non_empty("target", self.target)
        _require_non_empty("payload_summary", self.payload_summary)
        _require_non_empty("schema_version", self.schema_version)
        _validate_iso_timestamp("timestamp", self.timestamp)

        _validate_probability("risk_score", self.risk_score)
        _validate_probability("injection_confidence", self.injection_confidence)

        _validate_non_negative_int("correlation_alerts", self.correlation_alerts)
        _validate_non_negative_int("temporal_alerts", self.temporal_alerts)
        _validate_non_negative_number("latency_ms", self.latency_ms)

        if not isinstance(self.rule_hits, list):
            raise TypeError("rule_hits must be a list")

        if self.baseline_mode == BaselineMode.NONE:
            if self.decision != StudyDecision.NOT_EVALUATED:
                raise ValueError("baseline_none events must use decision=NOT_EVALUATED")
            if self.risk_score is not None:
                raise ValueError("baseline_none events must use risk_score=None")

        if self.decision in {StudyDecision.BLOCK, StudyDecision.ESCALATE}:
            self.was_intercepted = True

        if self.decision == StudyDecision.PENDING_REVIEW:
            self.was_reviewed = True

        if self.was_blocked and self.decision not in {
            StudyDecision.BLOCK,
            StudyDecision.ESCALATE,
        }:
            raise ValueError("was_blocked=True requires decision BLOCK or ESCALATE")

    def hard_intervention(self) -> bool:
        """Strict intervention: BLOCK or ESCALATE."""
        return self.decision in {StudyDecision.BLOCK, StudyDecision.ESCALATE}

    def soft_intervention(self) -> bool:
        """Review intervention: BLOCK, ESCALATE, or PENDING_REVIEW."""
        return self.decision in {
            StudyDecision.BLOCK,
            StudyDecision.ESCALATE,
            StudyDecision.PENDING_REVIEW,
        }

    def is_true_positive(self) -> bool:
        return self.ground_truth == GroundTruth.MALICIOUS and self.soft_intervention()

    def is_true_negative(self) -> bool:
        return self.ground_truth == GroundTruth.SAFE and not self.soft_intervention()

    def is_false_positive(self) -> bool:
        return self.ground_truth == GroundTruth.SAFE and self.soft_intervention()

    def is_false_negative(self) -> bool:
        return (
            self.ground_truth == GroundTruth.MALICIOUS and not self.soft_intervention()
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Structured JSONL-friendly representation."""
        return {
            "event_id": self.event_id,
            "study_run_id": self.study_run_id,
            "task_run_id": self.task_run_id,
            "task_id": self.task_id,
            "group": self.group.value,
            "ground_truth": self.ground_truth.value,
            "threat_label": self.threat_label.value,
            "baseline_mode": self.baseline_mode.value,
            "agent_id": self.agent_id,
            "framework": self.framework.value,
            "model_provider": self.model_provider.value,
            "model_name": self.model_name,
            "action_type": self.action_type,
            "target": self.target,
            "payload_summary": self.payload_summary,
            "decision": self.decision.value,
            "risk_score": (
                None if self.risk_score is None else round(self.risk_score, 4)
            ),
            "rule_hits": list(self.rule_hits),
            "was_blocked": self.was_blocked,
            "was_intercepted": self.was_intercepted,
            "was_reviewed": self.was_reviewed,
            "injection_detected": self.injection_detected,
            "injection_confidence": (
                None
                if self.injection_confidence is None
                else round(self.injection_confidence, 4)
            ),
            "correlation_alerts": self.correlation_alerts,
            "temporal_alerts": self.temporal_alerts,
            "safe_state_active": self.safe_state_active,
            "latency_ms": round(self.latency_ms, 3),
            "audit_entry_id": self.audit_entry_id,
            "schema_version": self.schema_version,
            "aisec_version": self.aisec_version,
            "git_commit": self.git_commit,
            "seed": self.seed,
            "framework_version": self.framework_version,
            "timestamp": self.timestamp,
        }

    def to_flat_dict(self) -> dict[str, Any]:
        """Flat CSV-friendly representation."""
        data = self.to_json_dict()
        data["rule_hits"] = ";".join(self.rule_hits)
        return data

    def to_dict(self) -> dict[str, Any]:
        """Backward-compatible alias for JSON representation."""
        return self.to_json_dict()


# =============================================================================
# Study metrics
# =============================================================================


@dataclass
class StudyMetrics:
    """
    Complete evaluation metrics for one baseline/study run.
    """

    baseline_mode: str
    total_events: int
    total_tasks: int

    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    not_evaluated_count: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    accuracy: float = 0.0

    group_a_fpr: float = 0.0
    group_b_detection_rate: float = 0.0
    group_c_enforcement_rate: float = 0.0
    group_d_correlation_rate: float = 0.0
    group_n_operational_rate: float = 0.0

    hard_block_rate: float = 0.0
    human_review_rate: float = 0.0
    intervention_rate: float = 0.0

    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    safe_state_activation_count: int = 0
    correlation_alert_count: int = 0
    prompt_injection_alert_count: int = 0
    audit_chain_intact: bool = False

    schema_version: str = SCHEMA_VERSION
    study_run_id: str = "unknown"
    aisec_version: str = "unknown"
    git_commit: str = "unknown"

    def __post_init__(self) -> None:
        _require_non_empty("baseline_mode", self.baseline_mode)
        _require_non_empty("schema_version", self.schema_version)

        count_fields = {
            "total_events": self.total_events,
            "total_tasks": self.total_tasks,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "not_evaluated_count": self.not_evaluated_count,
            "safe_state_activation_count": self.safe_state_activation_count,
            "correlation_alert_count": self.correlation_alert_count,
            "prompt_injection_alert_count": self.prompt_injection_alert_count,
        }

        for name, value in count_fields.items():
            _validate_non_negative_int(name, value)

        confusion_total = (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
            + self.not_evaluated_count
        )

        if confusion_total > self.total_events:
            raise ValueError("confusion matrix counts cannot exceed total_events")

        probability_fields = {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "accuracy": self.accuracy,
            "group_a_fpr": self.group_a_fpr,
            "group_b_detection_rate": self.group_b_detection_rate,
            "group_c_enforcement_rate": self.group_c_enforcement_rate,
            "group_d_correlation_rate": self.group_d_correlation_rate,
            "group_n_operational_rate": self.group_n_operational_rate,
            "hard_block_rate": self.hard_block_rate,
            "human_review_rate": self.human_review_rate,
            "intervention_rate": self.intervention_rate,
        }

        for name, value in probability_fields.items():
            _validate_probability(name, value)

        latency_fields = {
            "latency_mean_ms": self.latency_mean_ms,
            "latency_median_ms": self.latency_median_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
        }

        for name, value in latency_fields.items():
            _validate_non_negative_number(name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_run_id": self.study_run_id,
            "aisec_version": self.aisec_version,
            "git_commit": self.git_commit,
            "baseline_mode": self.baseline_mode,
            "total_events": self.total_events,
            "total_tasks": self.total_tasks,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "not_evaluated": self.not_evaluated_count,
            },
            "metrics": {
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1_score": round(self.f1_score, 4),
                "false_positive_rate": round(self.false_positive_rate, 4),
                "false_negative_rate": round(self.false_negative_rate, 4),
                "accuracy": round(self.accuracy, 4),
            },
            "per_group": {
                "group_a_false_positive_rate": round(self.group_a_fpr, 4),
                "group_b_injection_detection": round(self.group_b_detection_rate, 4),
                "group_c_enforcement_rate": round(self.group_c_enforcement_rate, 4),
                "group_d_correlation_rate": round(self.group_d_correlation_rate, 4),
                "group_n_operational_rate": round(self.group_n_operational_rate, 4),
            },
            "intervention": {
                "hard_block_rate": round(self.hard_block_rate, 4),
                "human_review_rate": round(self.human_review_rate, 4),
                "intervention_rate": round(self.intervention_rate, 4),
            },
            "latency_ms": {
                "mean": round(self.latency_mean_ms, 3),
                "median": round(self.latency_median_ms, 3),
                "p95": round(self.latency_p95_ms, 3),
                "p99": round(self.latency_p99_ms, 3),
            },
            "security": {
                "safe_state_activations": self.safe_state_activation_count,
                "correlation_alerts": self.correlation_alert_count,
                "prompt_injection_alerts": self.prompt_injection_alert_count,
                "audit_chain_intact": self.audit_chain_intact,
            },
        }