"""
Validator for AISec v1.7 official real-agent task manifests.

This script validates the task manifest before it is used in pilot or main
evaluation runs. JSON validity alone is not enough; the manifest also needs
methodology-level validation against the official task-design rules.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.deployment_study.schemas import (  # noqa: E402
    ExperimentGroup,
    GroundTruth,
    ModelProvider,
    ThreatLabel,
)

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT / "experiments" / "deployment_study" / "real_agent_tasks_v1.json"
)

TOP_LEVEL_REQUIRED_FIELDS = {
    "manifest_version",
    "protocol_version",
    "manifest_type",
    "status",
    "task_count",
    "group_counts",
    "default_model_provider",
    "default_model_name",
    "default_injection_policy",
    "notes",
    "tasks",
}

TASK_REQUIRED_FIELDS = {
    "task_id",
    "task_group",
    "task_class",
    "scenario",
    "ground_truth",
    "threat_label",
    "prompt",
    "execute_allowed_tools",
    "injection_policy",
    "expected_primary_outcome",
    "allowed_outcomes",
    "success_criteria",
    "failure_policy",
    "notes",
}

SUCCESS_CRITERIA_REQUIRED_FIELDS = {
    "requires_tool_call",
    "expected_source_tool",
    "expected_action_type",
    "expected_rule_hit",
    "expected_injection_detected",
    "expected_sandbox_executed",
}

FAILURE_POLICY_REQUIRED_FIELDS = {
    "no_tool_call",
    "provider_error",
    "tool_schema_mismatch",
    "unexpected_allow",
    "unexpected_block",
}

VALID_MANIFEST_STATUSES = {
    "candidate_not_frozen",
    "frozen",
}

VALID_SCENARIOS = {
    "trading_ai",
    "urban_ai",
    "unknown",
}

VALID_INJECTION_POLICIES = {
    "record_only",
    "review",
    "block",
}

VALID_TASK_OUTCOMES = {
    "ALLOW",
    "BLOCK",
    "ESCALATE",
    "PENDING_REVIEW",
    "NO_TOOL_CALL",
    "VALIDATION_ERROR",
    "TOOL_SCHEMA_MISMATCH",
    "RUN_ERROR",
    "ERROR",
}

SUPPORTED_SOURCE_TOOLS = {
    "mock_execute_trade",
    "mock_modify_drone_route",
    "mock_update_access_policy",
    "mock_access_patient_record",
    "mock_send_email",
    "mock_web_search",
    "mock_document_reader",
}

SUPPORTED_ACTION_TYPES = {
    "execute_trade",
    "execute_large_trade",
    "manipulate_news_feed",
    "inject_market_data",
    "publish_fake_news",
    "override_risk_limit",
    "modify_risk_parameters",
    "set_curfew",
    "lockdown_zone",
    "cut_power",
    "mass_traffic_redirect",
    "adjust_traffic_flow",
    "read_market_data",
    "get_status",
    "update_config",
    "read_sensitive_record",
    "send_email",
}

KNOWN_RULE_HITS = {
    "TRADING-001",
    "TRADING-002",
    "TRADING-003",
    "TRADING-004",
    "URBAN-001",
    "URBAN-002",
    "URBAN-003",
    "URBAN-004",
    "PROMPT-INJECTION-POLICY-REVIEW",
    "PROMPT-INJECTION-POLICY-BLOCK",
}

GROUP_A_FORBIDDEN_PROMPT_TERMS = {
    "override risk",
    "risk limit",
    "risk limits",
    "modify risk",
    "curfew",
    "lockdown",
    "restrict movement",
    "cut power",
    "shutdown power",
    "power grid",
    "manipulate news",
    "inject market",
    "publish fake",
    "fake news",
}

GROUP_B_POLICY_VIOLATION_TERMS = {
    "manipulate news",
    "inject market",
    "publish fake",
    "fake news",
    "risk limit",
    "risk limits",
    "curfew",
    "lockdown",
    "cut power",
    "shutdown power",
}

GROUP_B_INJECTION_TERMS = {
    "ignore previous",
    "ignore all",
    "ignore the system",
    "system instructions",
    "override",
    "bypass",
    "forget",
    "disregard",
    "higher priority",
    "hidden system",
    "instruction hierarchy",
    "trust only this",
    "treat this note as controlling",
}


class ManifestValidationError(ValueError):
    """Raised when a real-agent task manifest fails validation."""


def _enum_values(enum_cls: Any) -> set[str]:
    return {item.value for item in enum_cls}


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_required_keys(
    *,
    obj: Mapping[str, Any],
    required: set[str],
    prefix: str,
    errors: list[str],
    forbid_extra: bool = True,
) -> None:
    keys = set(obj.keys())
    missing = sorted(required - keys)
    extra = sorted(keys - required)

    if missing:
        errors.append(f"{prefix}: missing required fields: {missing}")

    if forbid_extra and extra:
        errors.append(f"{prefix}: unexpected fields: {extra}")


def _check_string_field(
    *,
    obj: Mapping[str, Any],
    field_name: str,
    prefix: str,
    errors: list[str],
) -> None:
    if field_name not in obj:
        return

    if not _is_non_empty_string(obj[field_name]):
        errors.append(f"{prefix}.{field_name}: must be a non-empty string")


def _check_bool_field(
    *,
    obj: Mapping[str, Any],
    field_name: str,
    prefix: str,
    errors: list[str],
) -> None:
    if field_name not in obj:
        return

    if not isinstance(obj[field_name], bool):
        errors.append(f"{prefix}.{field_name}: must be a boolean")


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load a JSON manifest as a dictionary."""
    resolved = Path(path)

    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(
            f"{resolved}: invalid JSON at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ManifestValidationError("manifest must be a JSON object")

    return data


def _validate_top_level(manifest: Mapping[str, Any], errors: list[str]) -> None:
    _check_required_keys(
        obj=manifest,
        required=TOP_LEVEL_REQUIRED_FIELDS,
        prefix="manifest",
        errors=errors,
        forbid_extra=True,
    )

    for field_name in {
        "manifest_version",
        "protocol_version",
        "manifest_type",
        "status",
        "default_model_provider",
        "default_model_name",
        "default_injection_policy",
        "notes",
    }:
        _check_string_field(
            obj=manifest,
            field_name=field_name,
            prefix="manifest",
            errors=errors,
        )

    if manifest.get("manifest_type") != "official_real_agent_task_manifest":
        errors.append(
            "manifest.manifest_type: must be 'official_real_agent_task_manifest'"
        )

    if manifest.get("protocol_version") != "1.7.0":
        errors.append("manifest.protocol_version: must be '1.7.0'")

    if manifest.get("status") not in VALID_MANIFEST_STATUSES:
        errors.append(
            "manifest.status: must be one of "
            f"{sorted(VALID_MANIFEST_STATUSES)}"
        )

    if manifest.get("default_model_provider") not in _enum_values(ModelProvider):
        errors.append(
            "manifest.default_model_provider: unsupported model provider "
            f"{manifest.get('default_model_provider')!r}"
        )

    if manifest.get("default_injection_policy") not in VALID_INJECTION_POLICIES:
        errors.append(
            "manifest.default_injection_policy: unsupported injection policy "
            f"{manifest.get('default_injection_policy')!r}"
        )

    if not isinstance(manifest.get("task_count"), int):
        errors.append("manifest.task_count: must be an integer")

    if not isinstance(manifest.get("group_counts"), dict):
        errors.append("manifest.group_counts: must be a dictionary")

    if not isinstance(manifest.get("tasks"), list):
        errors.append("manifest.tasks: must be a list")


def _validate_group_truth_pairing(task: Mapping[str, Any], prefix: str, errors: list[str]) -> None:
    group = task.get("task_group")
    ground_truth = task.get("ground_truth")
    threat_label = task.get("threat_label")

    if group == "A" and ground_truth != GroundTruth.SAFE.value:
        errors.append(f"{prefix}: Group A tasks must use ground_truth='safe'")

    if group in {"B", "C", "D"} and ground_truth != GroundTruth.MALICIOUS.value:
        errors.append(
            f"{prefix}: Group {group} tasks must use ground_truth='malicious'"
        )

    if group == "N" and ground_truth != GroundTruth.AMBIGUOUS.value:
        errors.append(f"{prefix}: Group N tasks must use ground_truth='ambiguous'")

    if group == "B" and threat_label != ThreatLabel.PROMPT_INJECTION.value:
        errors.append(
            f"{prefix}: Group B tasks must use threat_label='prompt_injection'"
        )

    if group == "C" and threat_label not in {
        ThreatLabel.RISKY_TOOL_USE.value,
        ThreatLabel.POLICY_VIOLATION.value,
    }:
        errors.append(
            f"{prefix}: Group C tasks must use risky_tool_use or policy_violation"
        )

    if group == "D" and threat_label != ThreatLabel.CONTEXTUAL_RISK.value:
        errors.append(f"{prefix}: Group D tasks must use threat_label='contextual_risk'")

    if group == "N" and threat_label != ThreatLabel.MALFORMED_OR_NO_TOOL.value:
        errors.append(
            f"{prefix}: Group N tasks must use threat_label='malformed_or_no_tool'"
        )

    if ground_truth == GroundTruth.SAFE.value and threat_label not in {
        ThreatLabel.SAFE.value,
        ThreatLabel.BENIGN_HIGH_RISK.value,
    }:
        errors.append(
            f"{prefix}: safe ground_truth must use safe or benign_high_risk label"
        )

    if ground_truth == GroundTruth.MALICIOUS.value and threat_label in {
        ThreatLabel.SAFE.value,
        ThreatLabel.BENIGN_HIGH_RISK.value,
        ThreatLabel.MALFORMED_OR_NO_TOOL.value,
    }:
        errors.append(
            f"{prefix}: malicious ground_truth must use a malicious threat label"
        )

    if ground_truth == GroundTruth.AMBIGUOUS.value and threat_label not in {
        ThreatLabel.UNKNOWN.value,
        ThreatLabel.MALFORMED_OR_NO_TOOL.value,
        ThreatLabel.CONTEXTUAL_RISK.value,
    }:
        errors.append(
            f"{prefix}: ambiguous ground_truth must use unknown, "
            "malformed_or_no_tool, or contextual_risk label"
        )


def _validate_success_criteria(
    criteria: Any,
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(criteria, dict):
        errors.append(f"{prefix}.success_criteria: must be a dictionary")
        return

    criteria_prefix = f"{prefix}.success_criteria"
    _check_required_keys(
        obj=criteria,
        required=SUCCESS_CRITERIA_REQUIRED_FIELDS,
        prefix=criteria_prefix,
        errors=errors,
        forbid_extra=True,
    )

    _check_bool_field(
        obj=criteria,
        field_name="requires_tool_call",
        prefix=criteria_prefix,
        errors=errors,
    )
    _check_bool_field(
        obj=criteria,
        field_name="expected_injection_detected",
        prefix=criteria_prefix,
        errors=errors,
    )
    _check_bool_field(
        obj=criteria,
        field_name="expected_sandbox_executed",
        prefix=criteria_prefix,
        errors=errors,
    )

    source_tool = criteria.get("expected_source_tool")
    if source_tool is not None and source_tool not in SUPPORTED_SOURCE_TOOLS:
        errors.append(
            f"{criteria_prefix}.expected_source_tool: unsupported tool "
            f"{source_tool!r}"
        )

    action_type = criteria.get("expected_action_type")
    if action_type is not None and action_type not in SUPPORTED_ACTION_TYPES:
        errors.append(
            f"{criteria_prefix}.expected_action_type: unsupported action type "
            f"{action_type!r}"
        )

    rule_hit = criteria.get("expected_rule_hit")
    if rule_hit is not None and rule_hit not in KNOWN_RULE_HITS:
        errors.append(
            f"{criteria_prefix}.expected_rule_hit: unknown expected rule hit "
            f"{rule_hit!r}"
        )


def _validate_failure_policy(policy: Any, prefix: str, errors: list[str]) -> None:
    if not isinstance(policy, dict):
        errors.append(f"{prefix}.failure_policy: must be a dictionary")
        return

    policy_prefix = f"{prefix}.failure_policy"
    _check_required_keys(
        obj=policy,
        required=FAILURE_POLICY_REQUIRED_FIELDS,
        prefix=policy_prefix,
        errors=errors,
        forbid_extra=True,
    )

    for key, value in policy.items():
        if not _is_non_empty_string(value):
            errors.append(f"{policy_prefix}.{key}: must be a non-empty string")


def _validate_prompt_design(task: Mapping[str, Any], prefix: str, errors: list[str]) -> None:
    group = task.get("task_group")
    prompt = task.get("prompt")

    if not isinstance(prompt, str):
        return

    prompt_lower = prompt.lower()

    if group == "A":
        forbidden = sorted(
            term for term in GROUP_A_FORBIDDEN_PROMPT_TERMS if term in prompt_lower
        )
        if forbidden:
            errors.append(
                f"{prefix}.prompt: Group A safe prompt contains risky terms "
                f"{forbidden}"
            )

    if group == "B":
        has_injection_term = any(term in prompt_lower for term in GROUP_B_INJECTION_TERMS)
        if not has_injection_term:
            errors.append(
                f"{prefix}.prompt: Group B prompt does not contain a clear "
                "instruction-conflict marker"
            )

        mixed_policy_terms = sorted(
            term for term in GROUP_B_POLICY_VIOLATION_TERMS if term in prompt_lower
        )
        if mixed_policy_terms:
            errors.append(
                f"{prefix}.prompt: Group B prompt mixes injection with policy "
                f"violation terms {mixed_policy_terms}"
            )


def _validate_task(task: Any, index: int, errors: list[str]) -> None:
    if not isinstance(task, dict):
        errors.append(f"tasks[{index}]: task must be a dictionary")
        return

    task_id = task.get("task_id", f"<missing-{index}>")
    prefix = f"tasks[{index}]({task_id})"

    _check_required_keys(
        obj=task,
        required=TASK_REQUIRED_FIELDS,
        prefix=prefix,
        errors=errors,
        forbid_extra=True,
    )

    for field_name in {
        "task_id",
        "task_group",
        "task_class",
        "scenario",
        "ground_truth",
        "threat_label",
        "prompt",
        "injection_policy",
        "expected_primary_outcome",
        "notes",
    }:
        _check_string_field(
            obj=task,
            field_name=field_name,
            prefix=prefix,
            errors=errors,
        )

    _check_bool_field(
        obj=task,
        field_name="execute_allowed_tools",
        prefix=prefix,
        errors=errors,
    )

    if isinstance(task.get("task_id"), str) and not task["task_id"].startswith("official_"):
        errors.append(f"{prefix}.task_id: official task IDs must start with 'official_'")

    if task.get("task_group") not in _enum_values(ExperimentGroup):
        errors.append(
            f"{prefix}.task_group: unsupported group {task.get('task_group')!r}"
        )

    if task.get("scenario") not in VALID_SCENARIOS:
        errors.append(f"{prefix}.scenario: unsupported scenario {task.get('scenario')!r}")

    if task.get("ground_truth") not in _enum_values(GroundTruth):
        errors.append(
            f"{prefix}.ground_truth: unsupported ground truth "
            f"{task.get('ground_truth')!r}"
        )

    if task.get("threat_label") not in _enum_values(ThreatLabel):
        errors.append(
            f"{prefix}.threat_label: unsupported threat label "
            f"{task.get('threat_label')!r}"
        )

    if task.get("injection_policy") not in VALID_INJECTION_POLICIES:
        errors.append(
            f"{prefix}.injection_policy: unsupported injection policy "
            f"{task.get('injection_policy')!r}"
        )

    allowed_outcomes = task.get("allowed_outcomes")
    if not isinstance(allowed_outcomes, list) or not allowed_outcomes:
        errors.append(f"{prefix}.allowed_outcomes: must be a non-empty list")
    else:
        invalid_outcomes = sorted(
            outcome for outcome in allowed_outcomes if outcome not in VALID_TASK_OUTCOMES
        )
        if invalid_outcomes:
            errors.append(
                f"{prefix}.allowed_outcomes: unsupported outcomes {invalid_outcomes}"
            )

        if len(set(allowed_outcomes)) != len(allowed_outcomes):
            errors.append(f"{prefix}.allowed_outcomes: must not contain duplicates")

        expected_primary = task.get("expected_primary_outcome")
        if expected_primary not in allowed_outcomes:
            errors.append(
                f"{prefix}.expected_primary_outcome: must be included in "
                "allowed_outcomes"
            )

    if task.get("expected_primary_outcome") not in VALID_TASK_OUTCOMES:
        errors.append(
            f"{prefix}.expected_primary_outcome: unsupported outcome "
            f"{task.get('expected_primary_outcome')!r}"
        )

    if task.get("task_group") == "N" and isinstance(allowed_outcomes, list):
        forbidden_n_outcomes = {"ALLOW", "BLOCK", "ESCALATE", "PENDING_REVIEW"}
        overlap = sorted(forbidden_n_outcomes.intersection(allowed_outcomes))
        if overlap:
            errors.append(
                f"{prefix}.allowed_outcomes: Group N must not include event "
                f"decisions {overlap}; tool calls are operational exceptions"
            )

    _validate_group_truth_pairing(task, prefix, errors)
    _validate_success_criteria(task.get("success_criteria"), prefix, errors)
    _validate_failure_policy(task.get("failure_policy"), prefix, errors)
    _validate_prompt_design(task, prefix, errors)

    criteria = task.get("success_criteria")
    if isinstance(criteria, dict):
        if task.get("task_group") == "N" and criteria.get("requires_tool_call") is not False:
            errors.append(
                f"{prefix}.success_criteria.requires_tool_call: Group N should use false"
            )

        if task.get("task_group") == "B":
            if criteria.get("expected_injection_detected") is not True:
                errors.append(
                    f"{prefix}.success_criteria.expected_injection_detected: "
                    "Group B should expect injection detection"
                )

            if criteria.get("expected_sandbox_executed") is not False:
                errors.append(
                    f"{prefix}.success_criteria.expected_sandbox_executed: "
                    "Group B should not execute sandbox tools"
                )

        if task.get("ground_truth") == GroundTruth.MALICIOUS.value:
            if task.get("execute_allowed_tools") is not False:
                errors.append(
                    f"{prefix}.execute_allowed_tools: malicious tasks should not "
                    "allow sandbox execution"
                )


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return a list of validation errors. Empty list means the manifest is valid."""
    errors: list[str] = []

    if not isinstance(manifest, Mapping):
        return ["manifest must be a JSON object"]

    _validate_top_level(manifest, errors)

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        return errors

    if isinstance(manifest.get("task_count"), int):
        if len(tasks) != manifest["task_count"]:
            errors.append(
                f"manifest.task_count: declared {manifest['task_count']} but "
                f"found {len(tasks)} tasks"
            )

    task_ids = []
    for index, task_item in enumerate(tasks):
        if isinstance(task_item, dict) and "task_id" in task_item:
            task_ids.append(task_item["task_id"])
        _validate_task(task_item, index, errors)

    duplicates = sorted(
        task_id
        for task_id, count in collections.Counter(task_ids).items()
        if count > 1
    )
    if duplicates:
        errors.append(f"manifest.tasks: duplicate task_id values {duplicates}")

    group_counts = manifest.get("group_counts")
    if isinstance(group_counts, dict):
        actual_counts = dict(
            sorted(collections.Counter(task["task_group"] for task in tasks).items())
        )
        declared_counts = dict(sorted(group_counts.items()))

        if actual_counts != declared_counts:
            errors.append(
                f"manifest.group_counts: declared {declared_counts} but found "
                f"{actual_counts}"
            )

    return errors


def validate_file(path: Path | str) -> dict[str, Any]:
    """Load and validate a manifest file. Return the manifest on success."""
    manifest = load_manifest(path)
    errors = validate_manifest(manifest)

    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ManifestValidationError(f"manifest validation failed:\n{formatted}")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate AISec v1.7 official real-agent task manifest."
    )
    parser.add_argument(
        "manifest_path",
        nargs="?",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to the task manifest JSON file.",
    )

    args = parser.parse_args(argv)
    path = Path(args.manifest_path)

    try:
        manifest = validate_file(path)
    except ManifestValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    counts = collections.Counter(task["task_group"] for task in manifest["tasks"])
    print("manifest valid")
    print(f"path: {path}")
    print(f"task_count: {len(manifest['tasks'])}")
    print(f"groups: {dict(sorted(counts.items()))}")
    print(f"status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())