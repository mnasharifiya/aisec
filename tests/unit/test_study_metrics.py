"""
Unit tests for AISec deployment-study metrics calculator.

These tests protect the scientific scoring logic used by the
deployment-study framework.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from experiments.deployment_study.metrics import compare_baselines, compute_metrics
from experiments.deployment_study.schemas import (
    BaselineMode,
    ExperimentGroup,
    Framework,
    GroundTruth,
    ModelProvider,
    StudyDecision,
    StudyEvent,
    StudyMetrics,
    ThreatLabel,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_event(
    *,
    event_id: str = "evt-001",
    study_run_id: str = "study-001",
    task_run_id: str | None = None,
    task_id: str = "task-001",
    group: ExperimentGroup = ExperimentGroup.A_BENIGN,
    ground_truth: GroundTruth = GroundTruth.SAFE,
    threat_label: ThreatLabel = ThreatLabel.SAFE,
    baseline_mode: BaselineMode = BaselineMode.AISEC_FULL,
    decision: StudyDecision = StudyDecision.ALLOW,
    risk_score: float | None = 0.1,
    was_blocked: bool = False,
    was_intercepted: bool = False,
    was_reviewed: bool = False,
    injection_detected: bool = False,
    injection_confidence: float | None = None,
    correlation_alerts: int = 0,
    temporal_alerts: int = 0,
    safe_state_active: bool = False,
    latency_ms: float = 1.0,
    agent_id: str = "bot",
    action_type: str = "execute_trade",
    target: str = "NYSE",
    audit_entry_id: str | None = "audit-001",
) -> StudyEvent:
    if task_run_id is None:
        task_run_id = f"{study_run_id}:{baseline_mode.value}:{task_id}"

    return StudyEvent(
        event_id=event_id,
        study_run_id=study_run_id,
        task_run_id=task_run_id,
        task_id=task_id,
        group=group,
        ground_truth=ground_truth,
        threat_label=threat_label,
        baseline_mode=baseline_mode,
        agent_id=agent_id,
        framework=Framework.SIMULATED,
        model_provider=ModelProvider.SIMULATED,
        model_name="simulated",
        action_type=action_type,
        target=target,
        payload_summary='{"keys": ["symbol"]}',
        decision=decision,
        risk_score=risk_score,
        rule_hits=[],
        was_blocked=was_blocked,
        was_intercepted=was_intercepted,
        was_reviewed=was_reviewed,
        injection_detected=injection_detected,
        injection_confidence=injection_confidence,
        correlation_alerts=correlation_alerts,
        temporal_alerts=temporal_alerts,
        safe_state_active=safe_state_active,
        latency_ms=latency_ms,
        audit_entry_id=audit_entry_id,
        schema_version="1.0",
        aisec_version="test",
        git_commit="test",
        seed=42,
        framework_version="test",
        timestamp=_timestamp(),
    )


def _make_blocked_malicious_event(
    *,
    event_id: str,
    task_id: str,
    group: ExperimentGroup = ExperimentGroup.C_RISKY,
    threat_label: ThreatLabel = ThreatLabel.RISKY_TOOL_USE,
) -> StudyEvent:
    return _make_event(
        event_id=event_id,
        task_id=task_id,
        group=group,
        ground_truth=GroundTruth.MALICIOUS,
        threat_label=threat_label,
        decision=StudyDecision.BLOCK,
        risk_score=0.9,
        was_blocked=True,
        was_intercepted=True,
    )


def _make_reviewed_malicious_event(
    *,
    event_id: str,
    task_id: str,
    group: ExperimentGroup = ExperimentGroup.B_INJECTION,
    threat_label: ThreatLabel = ThreatLabel.PROMPT_INJECTION,
) -> StudyEvent:
    return _make_event(
        event_id=event_id,
        task_id=task_id,
        group=group,
        ground_truth=GroundTruth.MALICIOUS,
        threat_label=threat_label,
        decision=StudyDecision.PENDING_REVIEW,
        risk_score=0.7,
        was_reviewed=True,
        injection_detected=True,
        injection_confidence=0.95,
    )


def _make_safe_allow_event(
    *,
    event_id: str,
    task_id: str,
    group: ExperimentGroup = ExperimentGroup.A_BENIGN,
) -> StudyEvent:
    return _make_event(
        event_id=event_id,
        task_id=task_id,
        group=group,
        ground_truth=GroundTruth.SAFE,
        threat_label=ThreatLabel.SAFE,
        decision=StudyDecision.ALLOW,
        risk_score=0.1,
        was_blocked=False,
        was_intercepted=False,
        was_reviewed=False,
    )


def _make_safe_false_positive_event(
    *,
    event_id: str,
    task_id: str,
    group: ExperimentGroup = ExperimentGroup.A_BENIGN,
) -> StudyEvent:
    return _make_event(
        event_id=event_id,
        task_id=task_id,
        group=group,
        ground_truth=GroundTruth.SAFE,
        threat_label=ThreatLabel.SAFE,
        decision=StudyDecision.BLOCK,
        risk_score=0.8,
        was_blocked=True,
        was_intercepted=True,
    )


def _make_baseline_none_event(
    *,
    event_id: str,
    task_id: str,
    ground_truth: GroundTruth = GroundTruth.SAFE,
    threat_label: ThreatLabel = ThreatLabel.SAFE,
    group: ExperimentGroup = ExperimentGroup.A_BENIGN,
) -> StudyEvent:
    return _make_event(
        event_id=event_id,
        task_id=task_id,
        group=group,
        ground_truth=ground_truth,
        threat_label=threat_label,
        baseline_mode=BaselineMode.NONE,
        decision=StudyDecision.NOT_EVALUATED,
        risk_score=None,
        was_blocked=False,
        was_intercepted=False,
        was_reviewed=False,
        injection_detected=False,
        injection_confidence=None,
        audit_entry_id=None,
        latency_ms=0.0,
    )


class TestComputeMetrics:
    def test_empty_events_returns_zero_metrics(self) -> None:
        metrics = compute_metrics([], BaselineMode.AISEC_FULL)

        assert metrics.total_events == 0
        assert metrics.total_tasks == 0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_perfect_detection(self) -> None:
        events = [
            _make_blocked_malicious_event(event_id="evt-1", task_id="mal-1"),
            _make_blocked_malicious_event(event_id="evt-2", task_id="mal-2"),
            _make_safe_allow_event(event_id="evt-3", task_id="safe-1"),
            _make_safe_allow_event(event_id="evt-4", task_id="safe-2"),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.true_positives == 2
        assert metrics.true_negatives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.not_evaluated_count == 0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.accuracy == 1.0

    def test_no_detection_on_malicious_events(self) -> None:
        events = [
            _make_event(
                event_id="evt-1",
                task_id="mal-1",
                group=ExperimentGroup.C_RISKY,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.RISKY_TOOL_USE,
                decision=StudyDecision.ALLOW,
                risk_score=0.2,
            ),
            _make_event(
                event_id="evt-2",
                task_id="mal-2",
                group=ExperimentGroup.C_RISKY,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.RISKY_TOOL_USE,
                decision=StudyDecision.ALLOW,
                risk_score=0.2,
            ),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.true_positives == 0
        assert metrics.false_negatives == 2
        assert metrics.recall == 0.0
        assert metrics.false_negative_rate == 1.0

    def test_pending_review_counts_as_soft_intervention(self) -> None:
        events = [
            _make_reviewed_malicious_event(event_id="evt-1", task_id="inj-1"),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.true_positives == 1
        assert metrics.false_negatives == 0
        assert metrics.recall == 1.0
        assert metrics.human_review_rate == 1.0
        assert metrics.intervention_rate == 1.0
        assert metrics.hard_block_rate == 0.0

    def test_safe_blocked_event_counts_as_false_positive(self) -> None:
        events = [
            _make_safe_false_positive_event(event_id="evt-1", task_id="safe-1"),
            _make_safe_allow_event(event_id="evt-2", task_id="safe-2"),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.false_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_positive_rate == 0.5

    def test_baseline_none_events_are_not_in_confusion_matrix(self) -> None:
        events = [
            _make_baseline_none_event(
                event_id="evt-1",
                task_id="safe-1",
                ground_truth=GroundTruth.SAFE,
                threat_label=ThreatLabel.SAFE,
                group=ExperimentGroup.A_BENIGN,
            ),
            _make_baseline_none_event(
                event_id="evt-2",
                task_id="mal-1",
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.PROMPT_INJECTION,
                group=ExperimentGroup.B_INJECTION,
            ),
        ]

        metrics = compute_metrics(events, BaselineMode.NONE)

        assert metrics.total_events == 2
        assert metrics.not_evaluated_count == 2
        assert metrics.true_positives == 0
        assert metrics.true_negatives == 0
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.accuracy == 0.0

    def test_rejects_mixed_baseline_modes(self) -> None:
        events = [
            _make_safe_allow_event(event_id="evt-1", task_id="safe-1"),
            _make_baseline_none_event(event_id="evt-2", task_id="safe-2"),
        ]

        with pytest.raises(ValueError):
            compute_metrics(events, BaselineMode.AISEC_FULL)

    def test_group_a_false_positive_rate_is_task_level(self) -> None:
        events = [
            _make_safe_false_positive_event(event_id="evt-1", task_id="safe-1"),
            _make_safe_allow_event(event_id="evt-2", task_id="safe-2"),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.group_a_fpr == 0.5

    def test_group_b_injection_detection_rate_is_task_level(self) -> None:
        events = [
            _make_event(
                event_id="evt-1",
                task_id="inj-1",
                group=ExperimentGroup.B_INJECTION,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.PROMPT_INJECTION,
                decision=StudyDecision.PENDING_REVIEW,
                risk_score=0.7,
                was_reviewed=True,
                injection_detected=True,
                injection_confidence=0.95,
            ),
            _make_event(
                event_id="evt-2",
                task_id="inj-2",
                group=ExperimentGroup.B_INJECTION,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.PROMPT_INJECTION,
                decision=StudyDecision.ALLOW,
                risk_score=0.2,
                injection_detected=False,
                injection_confidence=None,
            ),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.group_b_detection_rate == 0.5

    def test_group_b_multiple_events_same_task_detected_once(self) -> None:
        events = [
            _make_event(
                event_id="evt-1",
                task_id="inj-1",
                task_run_id="study-001:aisec_full:inj-1",
                group=ExperimentGroup.B_INJECTION,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.PROMPT_INJECTION,
                decision=StudyDecision.ALLOW,
                risk_score=0.2,
                injection_detected=False,
            ),
            _make_event(
                event_id="evt-2",
                task_id="inj-1",
                task_run_id="study-001:aisec_full:inj-1",
                group=ExperimentGroup.B_INJECTION,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.PROMPT_INJECTION,
                decision=StudyDecision.PENDING_REVIEW,
                risk_score=0.7,
                was_reviewed=True,
                injection_detected=True,
                injection_confidence=0.95,
            ),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.total_events == 2
        assert metrics.total_tasks == 1
        assert metrics.group_b_detection_rate == 1.0

    def test_group_c_enforcement_rate_is_task_level(self) -> None:
        events = [
            _make_blocked_malicious_event(
                event_id="evt-1",
                task_id="risk-1",
                group=ExperimentGroup.C_RISKY,
                threat_label=ThreatLabel.RISKY_TOOL_USE,
            ),
            _make_event(
                event_id="evt-2",
                task_id="risk-2",
                group=ExperimentGroup.C_RISKY,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.RISKY_TOOL_USE,
                decision=StudyDecision.ALLOW,
                risk_score=0.2,
            ),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.group_c_enforcement_rate == 0.5

    def test_group_d_correlation_rate_is_task_level(self) -> None:
        events = [
            _make_event(
                event_id="evt-1",
                task_id="multi-1",
                group=ExperimentGroup.D_MULTIAGENT,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.MULTI_AGENT_COORDINATION,
                decision=StudyDecision.ESCALATE,
                risk_score=0.9,
                was_intercepted=True,
                correlation_alerts=2,
            ),
            _make_event(
                event_id="evt-2",
                task_id="multi-2",
                group=ExperimentGroup.D_MULTIAGENT,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.MULTI_AGENT_COORDINATION,
                decision=StudyDecision.ALLOW,
                risk_score=0.2,
                correlation_alerts=0,
            ),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.group_d_correlation_rate == 0.5

    def test_latency_metrics_computed(self) -> None:
        events = [
            _make_safe_allow_event(event_id="evt-1", task_id="safe-1"),
            _make_safe_allow_event(event_id="evt-2", task_id="safe-2"),
            _make_safe_allow_event(event_id="evt-3", task_id="safe-3"),
            _make_safe_allow_event(event_id="evt-4", task_id="safe-4"),
        ]

        events[0].latency_ms = 1.0
        events[1].latency_ms = 2.0
        events[2].latency_ms = 3.0
        events[3].latency_ms = 4.0

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.latency_mean_ms == 2.5
        assert metrics.latency_median_ms == 2.5
        assert metrics.latency_p95_ms == 4.0
        assert metrics.latency_p99_ms == 4.0

    def test_security_counts_computed(self) -> None:
        events = [
            _make_event(
                event_id="evt-1",
                task_id="inj-1",
                group=ExperimentGroup.B_INJECTION,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.PROMPT_INJECTION,
                decision=StudyDecision.PENDING_REVIEW,
                risk_score=0.7,
                was_reviewed=True,
                injection_detected=True,
                injection_confidence=0.95,
                safe_state_active=True,
            ),
            _make_event(
                event_id="evt-2",
                task_id="multi-1",
                group=ExperimentGroup.D_MULTIAGENT,
                ground_truth=GroundTruth.MALICIOUS,
                threat_label=ThreatLabel.MULTI_AGENT_COORDINATION,
                decision=StudyDecision.ESCALATE,
                risk_score=0.9,
                was_intercepted=True,
                correlation_alerts=3,
            ),
        ]

        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)

        assert metrics.safe_state_activation_count == 1
        assert metrics.correlation_alert_count == 3
        assert metrics.prompt_injection_alert_count == 1

    def test_metrics_to_dict_has_all_sections(self) -> None:
        events = [_make_safe_allow_event(event_id="evt-1", task_id="safe-1")]
        metrics = compute_metrics(events, BaselineMode.AISEC_FULL)
        data = metrics.to_dict()

        assert "confusion_matrix" in data
        assert "metrics" in data
        assert "per_group" in data
        assert "intervention" in data
        assert "latency_ms" in data
        assert "security" in data


class TestCompareBaselines:
    def test_compare_baselines_returns_empty_without_aisec_full(self) -> None:
        baseline = StudyMetrics(
            baseline_mode=BaselineMode.NONE.value,
            total_events=0,
            total_tasks=0,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0,
            not_evaluated_count=0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            accuracy=0.0,
            group_a_fpr=0.0,
            group_b_detection_rate=0.0,
            group_c_enforcement_rate=0.0,
            group_d_correlation_rate=0.0,
            hard_block_rate=0.0,
            human_review_rate=0.0,
            intervention_rate=0.0,
            latency_mean_ms=0.0,
            latency_median_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            safe_state_activation_count=0,
            correlation_alert_count=0,
            prompt_injection_alert_count=0,
            audit_chain_intact=True,
        )

        assert compare_baselines({BaselineMode.NONE.value: baseline}) == {}

    def test_compare_baselines_reports_improvement(self) -> None:
        baseline = StudyMetrics(
            baseline_mode=BaselineMode.STATIC_RULES.value,
            total_events=10,
            total_tasks=5,
            true_positives=2,
            true_negatives=5,
            false_positives=0,
            false_negatives=3,
            not_evaluated_count=0,
            precision=1.0,
            recall=0.4,
            f1_score=0.5714,
            false_positive_rate=0.0,
            false_negative_rate=0.6,
            accuracy=0.7,
            group_a_fpr=0.0,
            group_b_detection_rate=0.0,
            group_c_enforcement_rate=0.5,
            group_d_correlation_rate=0.0,
            hard_block_rate=0.2,
            human_review_rate=0.0,
            intervention_rate=0.2,
            latency_mean_ms=1.0,
            latency_median_ms=1.0,
            latency_p95_ms=1.0,
            latency_p99_ms=1.0,
            safe_state_activation_count=0,
            correlation_alert_count=0,
            prompt_injection_alert_count=0,
            audit_chain_intact=True,
        )

        aisec = StudyMetrics(
            baseline_mode=BaselineMode.AISEC_FULL.value,
            total_events=10,
            total_tasks=5,
            true_positives=4,
            true_negatives=5,
            false_positives=0,
            false_negatives=1,
            not_evaluated_count=0,
            precision=1.0,
            recall=0.8,
            f1_score=0.8889,
            false_positive_rate=0.0,
            false_negative_rate=0.2,
            accuracy=0.9,
            group_a_fpr=0.0,
            group_b_detection_rate=0.9,
            group_c_enforcement_rate=0.8,
            group_d_correlation_rate=0.7,
            hard_block_rate=0.4,
            human_review_rate=0.2,
            intervention_rate=0.6,
            latency_mean_ms=2.0,
            latency_median_ms=2.0,
            latency_p95_ms=2.0,
            latency_p99_ms=2.0,
            safe_state_activation_count=1,
            correlation_alert_count=2,
            prompt_injection_alert_count=3,
            audit_chain_intact=True,
        )

        comparison = compare_baselines(
            {
                BaselineMode.STATIC_RULES.value: baseline,
                BaselineMode.AISEC_FULL.value: aisec,
            }
        )

        assert "aisec_full" in comparison
        assert "comparisons" in comparison
        assert BaselineMode.STATIC_RULES.value in comparison["comparisons"]

        improvement = comparison["comparisons"][BaselineMode.STATIC_RULES.value][
            "aisec_improvement"
        ]

        assert improvement["recall_gain"] == 0.4
        assert improvement["f1_gain"] == 0.3175
        assert improvement["fnr_reduction"] == 0.4
