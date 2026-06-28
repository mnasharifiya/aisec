"""
Unit tests for the AISec v1.7 official real-agent task manifest validator.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.deployment_study.agents.validate_real_agent_tasks import (
    ManifestValidationError,
    load_manifest,
    validate_file,
    validate_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    PROJECT_ROOT / "experiments" / "deployment_study" / "real_agent_tasks_v1.json"
)


def _manifest() -> dict:
    return load_manifest(MANIFEST_PATH)


def _first_task_index(manifest: dict, group: str) -> int:
    for index, task in enumerate(manifest["tasks"]):
        if task["task_group"] == group:
            return index
    raise AssertionError(f"group {group} not found")


def test_official_manifest_candidate_validates() -> None:
    errors = validate_manifest(_manifest())

    assert errors == []


def test_validate_file_accepts_real_manifest() -> None:
    manifest = validate_file(MANIFEST_PATH)

    assert manifest["manifest_type"] == "official_real_agent_task_manifest"
    assert manifest["task_count"] == 50
    assert len(manifest["tasks"]) == 50


def test_validator_rejects_duplicate_task_ids() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["tasks"][1]["task_id"] = manifest["tasks"][0]["task_id"]

    errors = validate_manifest(manifest)

    assert any("duplicate task_id" in error for error in errors)


def test_validator_rejects_task_count_mismatch() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["task_count"] = 999

    errors = validate_manifest(manifest)

    assert any("task_count" in error for error in errors)


def test_validator_rejects_group_count_mismatch() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["group_counts"]["A"] = 999

    errors = validate_manifest(manifest)

    assert any("group_counts" in error for error in errors)


def test_validator_rejects_group_n_event_decisions() -> None:
    manifest = copy.deepcopy(_manifest())
    index = _first_task_index(manifest, "N")
    manifest["tasks"][index]["allowed_outcomes"].append("ALLOW")

    errors = validate_manifest(manifest)

    assert any("Group N must not include event decisions" in error for error in errors)


def test_validator_rejects_contaminated_group_a_prompt() -> None:
    manifest = copy.deepcopy(_manifest())
    index = _first_task_index(manifest, "A")
    manifest["tasks"][index]["prompt"] += " Also set a curfew."

    errors = validate_manifest(manifest)

    assert any("Group A safe prompt contains risky terms" in error for error in errors)


def test_validator_rejects_mixed_group_b_policy_terms() -> None:
    manifest = copy.deepcopy(_manifest())
    index = _first_task_index(manifest, "B")
    manifest["tasks"][index]["prompt"] += " Then manipulate news."

    errors = validate_manifest(manifest)

    assert any("Group B prompt mixes injection" in error for error in errors)


def test_validator_rejects_unknown_source_tool() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["tasks"][0]["success_criteria"]["expected_source_tool"] = "mock_unknown"

    errors = validate_manifest(manifest)

    assert any("expected_source_tool" in error for error in errors)


def test_validator_rejects_unknown_action_type() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["tasks"][0]["success_criteria"]["expected_action_type"] = "unknown_action"

    errors = validate_manifest(manifest)

    assert any("expected_action_type" in error for error in errors)


def test_validator_rejects_primary_outcome_not_in_allowed_outcomes() -> None:
    manifest = copy.deepcopy(_manifest())
    manifest["tasks"][0]["expected_primary_outcome"] = "BLOCK"

    errors = validate_manifest(manifest)

    assert any("expected_primary_outcome" in error for error in errors)


def test_validator_rejects_group_ground_truth_mismatch() -> None:
    manifest = copy.deepcopy(_manifest())
    index = _first_task_index(manifest, "B")
    manifest["tasks"][index]["ground_truth"] = "safe"

    errors = validate_manifest(manifest)

    assert any("Group B tasks must use ground_truth='malicious'" in error for error in errors)


def test_validate_file_raises_on_invalid_manifest(tmp_path: Path) -> None:
    path = tmp_path / "bad_manifest.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="JSON object"):
        validate_file(path)