"""
Schema alignment tests for AISec v1.7 official real-agent evaluation.

These tests ensure that the deployment-study schema supports the official
task-design document before the 50-task manifest is created.
"""

from __future__ import annotations

import pytest

from experiments.deployment_study.schemas import (
    BaselineMode,
    ExperimentGroup,
    GroundTruth,
    TaskDefinition,
    ThreatLabel,
)


def test_experiment_group_supports_official_task_groups() -> None:
    assert ExperimentGroup.A_BENIGN.value == "A"
    assert ExperimentGroup.B_INJECTION.value == "B"
    assert ExperimentGroup.C_RISKY.value == "C"
    assert ExperimentGroup.D_MULTIAGENT.value == "D"
    assert ExperimentGroup.N_OPERATIONAL.value == "N"


def test_ground_truth_supports_ambiguous_operational_cases() -> None:
    assert GroundTruth.SAFE.value == "safe"
    assert GroundTruth.MALICIOUS.value == "malicious"
    assert GroundTruth.AMBIGUOUS.value == "ambiguous"


def test_threat_label_supports_official_manifest_labels() -> None:
    assert ThreatLabel.SAFE.value == "safe"
    assert ThreatLabel.PROMPT_INJECTION.value == "prompt_injection"
    assert ThreatLabel.RISKY_TOOL_USE.value == "risky_tool_use"
    assert ThreatLabel.MULTI_AGENT_COORDINATION.value == "multi_agent_coordination"
    assert ThreatLabel.POLICY_VIOLATION.value == "policy_violation"
    assert ThreatLabel.BENIGN_HIGH_RISK.value == "benign_high_risk"
    assert ThreatLabel.CONTEXTUAL_RISK.value == "contextual_risk"
    assert ThreatLabel.MALFORMED_OR_NO_TOOL.value == "malformed_or_no_tool"
    assert ThreatLabel.UNKNOWN.value == "unknown"


def test_baseline_mode_supports_llm_judge_baseline() -> None:
    assert BaselineMode.NONE.value == "baseline_none"
    assert BaselineMode.STATIC_RULES.value == "baseline_static_rules"
    assert BaselineMode.PROMPT_ONLY.value == "baseline_prompt_only"
    assert BaselineMode.LLM_JUDGE.value == "baseline_llm_judge"
    assert BaselineMode.AISEC_FULL.value == "aisec_full"


def test_task_definition_accepts_operational_group_n_task() -> None:
    task = TaskDefinition(
        task_id="official_operational_no_tool_001",
        group=ExperimentGroup.N_OPERATIONAL,
        ground_truth=GroundTruth.AMBIGUOUS,
        threat_label=ThreatLabel.MALFORMED_OR_NO_TOOL,
        description="Ambiguous operational task where no tool call may be valid.",
        expected_behavior="Record the outcome without forcing a detection label.",
        actions=[
            {
                "action_type": "NO_TOOL_CALL",
                "expected_outcome": "NO_TOOL_CALL",
            }
        ],
    )

    assert task.group == ExperimentGroup.N_OPERATIONAL
    assert task.ground_truth == GroundTruth.AMBIGUOUS
    assert task.threat_label == ThreatLabel.MALFORMED_OR_NO_TOOL


def test_task_definition_rejects_safe_task_with_malicious_label() -> None:
    with pytest.raises(ValueError, match="SAFE ground_truth"):
        TaskDefinition(
            task_id="invalid_safe_task",
            group=ExperimentGroup.A_BENIGN,
            ground_truth=GroundTruth.SAFE,
            threat_label=ThreatLabel.RISKY_TOOL_USE,
            description="Invalid safe task.",
            expected_behavior="Should fail validation.",
            actions=[
                {
                    "action_type": "execute_trade",
                }
            ],
        )


def test_task_definition_rejects_malicious_task_with_safe_label() -> None:
    with pytest.raises(ValueError, match="MALICIOUS ground_truth"):
        TaskDefinition(
            task_id="invalid_malicious_task",
            group=ExperimentGroup.C_RISKY,
            ground_truth=GroundTruth.MALICIOUS,
            threat_label=ThreatLabel.SAFE,
            description="Invalid malicious task.",
            expected_behavior="Should fail validation.",
            actions=[
                {
                    "action_type": "execute_large_trade",
                }
            ],
        )


def test_task_definition_rejects_ambiguous_task_with_safe_label() -> None:
    with pytest.raises(ValueError, match="AMBIGUOUS ground_truth"):
        TaskDefinition(
            task_id="invalid_ambiguous_task",
            group=ExperimentGroup.N_OPERATIONAL,
            ground_truth=GroundTruth.AMBIGUOUS,
            threat_label=ThreatLabel.SAFE,
            description="Invalid ambiguous task.",
            expected_behavior="Should fail validation.",
            actions=[
                {
                    "action_type": "NO_TOOL_CALL",
                }
            ],
        )
        