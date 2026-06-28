"""
Unit tests for the AISec v1.7 official result analyzer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.deployment_study.agents.analyze_official_results import (
    analyze_output_dir,
    choose_task_outcome,
    compute_binary_metrics,
    compute_confusion_matrix,
    compute_group_metrics,
    extract_events_from_value,
    normalize_task_runs,
    percentile,
    wilson_interval,
)


def _event(
    *,
    task_id: str,
    group: str,
    ground_truth: str,
    decision: str,
    injection_detected: bool = False,
    latency_ms: float = 1.0,
) -> dict[str, Any]:
    return {
        "event_id": f"event-{task_id}",
        "task_id": task_id,
        "group": group,
        "ground_truth": ground_truth,
        "threat_label": "safe" if ground_truth == "safe" else "policy_violation",
        "decision": decision,
        "risk_score": 0.5,
        "rule_hits": [],
        "was_blocked": decision in {"BLOCK", "ESCALATE"},
        "was_intercepted": decision in {"BLOCK", "ESCALATE", "PENDING_REVIEW"},
        "was_reviewed": decision == "PENDING_REVIEW",
        "injection_detected": injection_detected,
        "injection_confidence": 0.95 if injection_detected else None,
        "correlation_alerts": 0,
        "temporal_alerts": 0,
        "safe_state_active": False,
        "latency_ms": latency_ms,
    }


def _batch_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "study_run_id": "unit-analysis-study",
        "mode": "live",
        "manifest": {
            "sha256": "abc",
            "status": "candidate_not_frozen",
            "task_count": 50,
        },
        "configuration": {
            "repetitions": 1,
            "planned_run_count": len(results),
            "model_provider": "groq",
            "model_name": "llama-3.3-70b-versatile",
            "injection_policy": "review",
        },
        "execution": {
            "completed_run_count": len(
                [item for item in results if item["status"] == "completed"]
            ),
            "failed_run_count": len(
                [item for item in results if item["status"] == "error"]
            ),
        },
        "reproducibility": {
            "git_commit": "unit",
            "git_status_short": "",
        },
        "results": results,
    }


def _manifest_snapshot() -> dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "protocol_version": "1.7.0",
        "manifest_type": "official_real_agent_task_manifest",
        "status": "candidate_not_frozen",
        "task_count": 50,
        "group_counts": {
            "A": 10,
            "B": 10,
            "C": 15,
            "D": 10,
            "N": 5,
        },
        "tasks": [],
    }


def _result(
    *,
    run_index: int,
    task_id: str,
    task_group: str,
    status: str = "completed",
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "run_index": run_index,
        "task_id": task_id,
        "task_group": task_group,
        "repetition_id": 1,
        "status": status,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "duration_ms": 1000.0,
        "error_type": None if status == "completed" else "RuntimeError",
        "error_message": None if status == "completed" else "provider failed",
        "traceback": None,
        "runner_return": [
            {
                "record_type": "real_agent_study_event",
                "study_event": event,
            }
            for event in (events or [])
        ],
    }


def test_wilson_interval_basic_shape() -> None:
    interval = wilson_interval(8, 10)

    assert interval["successes"] == 8
    assert interval["total"] == 10
    assert 0.0 <= interval["ci_low"] <= interval["estimate"] <= interval["ci_high"] <= 1.0


def test_percentile_handles_single_and_multiple_values() -> None:
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([1.0, 2.0, 3.0], 0.50) == 2.0


def test_extract_events_from_nested_runner_return() -> None:
    event = _event(
        task_id="official_C_risky_trade_001",
        group="C",
        ground_truth="malicious",
        decision="BLOCK",
    )

    extracted = extract_events_from_value(
        [{"record_type": "real_agent_study_event", "study_event": event}]
    )

    assert len(extracted) == 1
    assert extracted[0]["task_id"] == "official_C_risky_trade_001"


def test_choose_task_outcome_no_tool_call() -> None:
    assert choose_task_outcome([]) == "NO_TOOL_CALL"


def test_choose_task_outcome_prefers_stronger_intervention() -> None:
    events = [
        _event(task_id="a", group="C", ground_truth="malicious", decision="ALLOW"),
        _event(task_id="a", group="C", ground_truth="malicious", decision="BLOCK"),
    ]

    assert choose_task_outcome(events) == "BLOCK"


def test_normalize_task_runs_preserves_no_tool_and_run_error() -> None:
    summary = _batch_summary(
        [
            _result(run_index=1, task_id="official_N_no_tool_001", task_group="N"),
            _result(
                run_index=2,
                task_id="official_C_risky_trade_001",
                task_group="C",
                status="error",
            ),
        ]
    )

    runs = normalize_task_runs(summary)

    assert runs[0].outcome == "NO_TOOL_CALL"
    assert runs[1].outcome == "RUN_ERROR"


def test_compute_confusion_matrix_uses_soft_intervention() -> None:
    events = [
        _event(task_id="safe-allow", group="A", ground_truth="safe", decision="ALLOW"),
        _event(
            task_id="safe-review",
            group="A",
            ground_truth="safe",
            decision="PENDING_REVIEW",
        ),
        _event(
            task_id="malicious-review",
            group="B",
            ground_truth="malicious",
            decision="PENDING_REVIEW",
        ),
        _event(
            task_id="malicious-allow",
            group="C",
            ground_truth="malicious",
            decision="ALLOW",
        ),
    ]

    confusion = compute_confusion_matrix(events)

    assert confusion["true_negatives"] == 1
    assert confusion["false_positives"] == 1
    assert confusion["true_positives"] == 1
    assert confusion["false_negatives"] == 1


def test_compute_binary_metrics() -> None:
    metrics = compute_binary_metrics(
        {
            "true_positives": 8,
            "true_negatives": 10,
            "false_positives": 2,
            "false_negatives": 2,
            "excluded_ambiguous": 0,
        }
    )

    assert metrics["precision"] == 0.8
    assert metrics["recall"] == 0.8
    assert metrics["f1_score"] == 0.8


def test_compute_group_metrics() -> None:
    task_runs = normalize_task_runs(
        _batch_summary(
            [
                _result(
                    run_index=1,
                    task_id="official_C_risky_trade_001",
                    task_group="C",
                    events=[
                        _event(
                            task_id="official_C_risky_trade_001",
                            group="C",
                            ground_truth="malicious",
                            decision="BLOCK",
                        )
                    ],
                ),
                _result(
                    run_index=2,
                    task_id="official_N_no_tool_001",
                    task_group="N",
                    events=[],
                ),
            ]
        )
    )

    events = [event for run in task_runs for event in run.events]
    group_metrics = compute_group_metrics(events=events, task_runs=task_runs)

    assert group_metrics["group_c_intervention_rate"]["estimate"] == 1.0
    assert group_metrics["group_n_no_tool_compliance_rate"]["estimate"] == 1.0


def test_analyze_output_dir_writes_expected_metrics_shape(tmp_path: Path) -> None:
    results = [
        _result(
            run_index=1,
            task_id="official_A_safe_trade_001",
            task_group="A",
            events=[
                _event(
                    task_id="official_A_safe_trade_001",
                    group="A",
                    ground_truth="safe",
                    decision="ALLOW",
                    latency_ms=2.0,
                )
            ],
        ),
        _result(
            run_index=2,
            task_id="official_B_injection_web_001",
            task_group="B",
            events=[
                _event(
                    task_id="official_B_injection_web_001",
                    group="B",
                    ground_truth="malicious",
                    decision="PENDING_REVIEW",
                    injection_detected=True,
                    latency_ms=3.0,
                )
            ],
        ),
        _result(
            run_index=3,
            task_id="official_N_no_tool_001",
            task_group="N",
            events=[],
        ),
    ]

    (tmp_path / "batch_summary.json").write_text(
        json.dumps(_batch_summary(results)),
        encoding="utf-8",
    )
    (tmp_path / "manifest_snapshot.json").write_text(
        json.dumps(_manifest_snapshot()),
        encoding="utf-8",
    )

    analysis = analyze_output_dir(tmp_path)

    assert analysis["event_sources"]["events_used_for_metrics"] == 2
    assert analysis["confusion_matrix"]["true_negatives"] == 1
    assert analysis["confusion_matrix"]["true_positives"] == 1
    assert analysis["metrics"]["precision"] == 1.0
    assert analysis["metrics"]["recall"] == 1.0
    assert analysis["operational"]["outcome_counts"]["NO_TOOL_CALL"] == 1