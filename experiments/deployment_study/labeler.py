"""
AISec Deployment Study — Task Labeler.

Loads task definitions from tasks.yaml and provides ground-truth labels
for the evaluation framework.

Design principle:
    Labels are assigned at the TASK level, not the event level.
    Every event produced by a malicious task is labeled malicious.
    Every event produced by a safe task is labeled safe.

This module must fail loudly on malformed tasks. Silent fallback is dangerous
because it can corrupt evaluation results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from experiments.deployment_study.schemas import (
    ExperimentGroup,
    Framework,
    GroundTruth,
    ModelProvider,
    StudyDecision,
    TaskDefinition,
    ThreatLabel,
)

TASKS_FILE = Path(__file__).parent / "tasks.yaml"

ALLOWED_SCENARIOS = {
    "trading_ai",
    "urban_ai",
    "healthcare",
    "drone",
}

REQUIRED_TASK_FIELDS = {
    "task_id",
    "group",
    "ground_truth",
    "threat_label",
    "description",
    "expected_behavior",
    "actions",
}

REQUIRED_ACTION_FIELDS = {
    "action_type",
    "scenario",
    "agent_id",
    "target",
    "payload",
    "expected_decision",
    "expected_rule_hits",
    "expected_injection_detected",
    "expected_correlation_alerts",
}


def load_tasks(path: Path = TASKS_FILE) -> list[TaskDefinition]:
    """
    Load all task definitions from tasks.yaml.

    Args:
        path: Path to tasks.yaml.

    Returns:
        Validated list of TaskDefinition objects.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If YAML structure or task contents are invalid.
        TypeError: If fields have invalid types.
    """
    raw_tasks = load_raw_tasks(path)

    tasks: list[TaskDefinition] = []
    seen_task_ids: set[str] = set()

    for index, raw in enumerate(raw_tasks):
        task = _parse_task(raw, index=index)

        if task.task_id in seen_task_ids:
            raise ValueError(f"Duplicate task_id found: {task.task_id}")

        seen_task_ids.add(task.task_id)
        tasks.append(task)

    return tasks


def load_raw_tasks(path: Path = TASKS_FILE) -> list[dict[str, Any]]:
    """
    Load raw task dictionaries from YAML without converting to TaskDefinition.
    """
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError("tasks.yaml must contain a YAML mapping at the top level")

    if "tasks" not in data:
        raise ValueError("tasks.yaml must contain a 'tasks' key")

    tasks = data["tasks"]

    if not isinstance(tasks, list):
        raise TypeError("'tasks' must be a list")

    if not tasks:
        raise ValueError("'tasks' cannot be empty")

    for index, raw in enumerate(tasks):
        if not isinstance(raw, dict):
            raise TypeError(f"tasks[{index}] must be a dictionary")

    return tasks


def load_tasks_by_group(
    group: ExperimentGroup | str,
    path: Path = TASKS_FILE,
) -> list[TaskDefinition]:
    """Return only tasks in one experimental group."""
    normalized_group = _parse_group(group, task_id="<filter>")
    return [task for task in load_tasks(path) if task.group == normalized_group]


def load_tasks_by_ground_truth(
    ground_truth: GroundTruth | str,
    path: Path = TASKS_FILE,
) -> list[TaskDefinition]:
    """Return only tasks with one ground-truth label."""
    normalized_ground_truth = _parse_ground_truth(
        ground_truth,
        task_id="<filter>",
    )
    return [
        task
        for task in load_tasks(path)
        if task.ground_truth == normalized_ground_truth
    ]


def load_tasks_by_threat_label(
    threat_label: ThreatLabel | str,
    path: Path = TASKS_FILE,
) -> list[TaskDefinition]:
    """Return only tasks with one threat label."""
    normalized_threat_label = _parse_threat_label(
        threat_label,
        task_id="<filter>",
    )
    return [
        task
        for task in load_tasks(path)
        if task.threat_label == normalized_threat_label
    ]


def get_task_summary(path: Path = TASKS_FILE) -> dict[str, Any]:
    """
    Return a summary of the task distribution.
    """
    tasks = load_tasks(path)

    summary: dict[str, Any] = {
        "total": len(tasks),
        "total_actions": sum(len(task.actions) for task in tasks),
        "by_group": {},
        "by_ground_truth": {
            GroundTruth.SAFE.value: sum(1 for task in tasks if task.is_benign()),
            GroundTruth.MALICIOUS.value: sum(
                1 for task in tasks if task.is_malicious()
            ),
        },
        "by_threat_label": {},
        "by_framework": {},
        "by_model_provider": {},
        "external_attack_tasks": sum(1 for task in tasks if task.is_external_attack),
    }

    for group in ExperimentGroup:
        group_tasks = [task for task in tasks if task.group == group]
        summary["by_group"][group.value] = {
            "count": len(group_tasks),
            "safe": sum(1 for task in group_tasks if task.is_benign()),
            "malicious": sum(1 for task in group_tasks if task.is_malicious()),
            "actions": sum(len(task.actions) for task in group_tasks),
        }

    for threat_label in ThreatLabel:
        summary["by_threat_label"][threat_label.value] = sum(
            1 for task in tasks if task.threat_label == threat_label
        )

    for framework in Framework:
        summary["by_framework"][framework.value] = sum(
            1 for task in tasks if task.framework == framework
        )

    for provider in ModelProvider:
        summary["by_model_provider"][provider.value] = sum(
            1 for task in tasks if task.model_provider == provider
        )

    return summary


def validate_task_file(path: Path = TASKS_FILE) -> dict[str, Any]:
    """
    Validate tasks.yaml and return a summary.

    This is useful for CLI checks and tests.
    """
    tasks = load_tasks(path)

    if not tasks:
        raise ValueError("No tasks loaded")

    return get_task_summary(path)


# =============================================================================
# Private parsing helpers
# =============================================================================


def _parse_task(raw: dict[str, Any], index: int) -> TaskDefinition:
    """Parse a raw YAML task dictionary into a validated TaskDefinition."""
    task_id = str(raw.get("task_id", f"tasks[{index}]"))

    missing = sorted(field for field in REQUIRED_TASK_FIELDS if field not in raw)
    if missing:
        raise ValueError(f"Task '{task_id}' missing required fields: {missing}")

    group = _parse_group(raw["group"], task_id=task_id)
    ground_truth = _parse_ground_truth(raw["ground_truth"], task_id=task_id)
    threat_label = _parse_threat_label(raw["threat_label"], task_id=task_id)
    framework = _parse_framework(raw.get("framework", "simulated"), task_id=task_id)
    model_provider = _parse_model_provider(
        raw.get("model_provider", "simulated"),
        task_id=task_id,
    )

    actions = raw["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"Task '{task_id}' must contain a non-empty actions list")

    validated_actions: list[dict[str, Any]] = []
    for action_index, action in enumerate(actions):
        validated_actions.append(
            _validate_action(
                action,
                task_id=task_id,
                action_index=action_index,
            )
        )

    return TaskDefinition(
        task_id=task_id,
        group=group,
        ground_truth=ground_truth,
        threat_label=threat_label,
        description=str(raw["description"]),
        expected_behavior=str(raw["expected_behavior"]),
        actions=validated_actions,
        framework=framework,
        model_provider=model_provider,
        task_author=str(raw.get("task_author", "self")),
        attack_source=str(raw.get("attack_source", "self_designed")),
        is_external_attack=bool(raw.get("is_external_attack", False)),
        adversary_id=str(raw.get("adversary_id", "")),
        notes=str(raw.get("notes", "")),
        tags=_parse_tags(raw.get("tags", []), task_id=task_id),
    )


def _validate_action(
    action: dict[str, Any],
    task_id: str,
    action_index: int,
) -> dict[str, Any]:
    """Validate a single action dictionary."""
    if not isinstance(action, dict):
        raise TypeError(f"Task '{task_id}' action[{action_index}] must be a dict")

    missing = sorted(field for field in REQUIRED_ACTION_FIELDS if field not in action)
    if missing:
        raise ValueError(
            f"Task '{task_id}' action[{action_index}] missing fields: {missing}"
        )

    scenario = str(action["scenario"]).strip()
    if scenario not in ALLOWED_SCENARIOS:
        raise ValueError(
            f"Task '{task_id}' action[{action_index}] has invalid scenario "
            f"'{scenario}'. Valid: {sorted(ALLOWED_SCENARIOS)}"
        )

    _require_non_empty_action_field(action, "action_type", task_id, action_index)
    _require_non_empty_action_field(action, "agent_id", task_id, action_index)
    _require_non_empty_action_field(action, "target", task_id, action_index)

    if not isinstance(action["payload"], dict):
        raise TypeError(
            f"Task '{task_id}' action[{action_index}] payload must be a dict"
        )

    expected_decision = str(action["expected_decision"]).strip()
    try:
        StudyDecision(expected_decision)
    except ValueError as exc:
        valid = [decision.value for decision in StudyDecision]
        raise ValueError(
            f"Task '{task_id}' action[{action_index}] invalid "
            f"expected_decision '{expected_decision}'. Valid: {valid}"
        ) from exc

    if not isinstance(action["expected_rule_hits"], list):
        raise TypeError(
            f"Task '{task_id}' action[{action_index}] expected_rule_hits "
            "must be a list"
        )

    if not isinstance(action["expected_injection_detected"], bool):
        raise TypeError(
            f"Task '{task_id}' action[{action_index}] "
            "expected_injection_detected must be true or false"
        )

    if not isinstance(action["expected_correlation_alerts"], list):
        raise TypeError(
            f"Task '{task_id}' action[{action_index}] "
            "expected_correlation_alerts must be a list"
        )

    if "delay_after_ms" in action:
        delay = action["delay_after_ms"]
        if not isinstance(delay, int | float) or delay < 0:
            raise ValueError(
                f"Task '{task_id}' action[{action_index}] delay_after_ms "
                "must be a non-negative number"
            )

    if "concurrent" in action and not isinstance(action["concurrent"], bool):
        raise TypeError(
            f"Task '{task_id}' action[{action_index}] concurrent must be bool"
        )

    return dict(action)


def _parse_group(value: ExperimentGroup | str, task_id: str) -> ExperimentGroup:
    if isinstance(value, ExperimentGroup):
        return value

    normalized = str(value).strip().upper()
    group_map = {
        "A": ExperimentGroup.A_BENIGN,
        "B": ExperimentGroup.B_INJECTION,
        "C": ExperimentGroup.C_RISKY,
        "D": ExperimentGroup.D_MULTIAGENT,
    }

    if normalized not in group_map:
        raise ValueError(
            f"Task '{task_id}' has invalid group '{value}'. Valid: A, B, C, D"
        )

    return group_map[normalized]


def _parse_ground_truth(
    value: GroundTruth | str,
    task_id: str,
) -> GroundTruth:
    if isinstance(value, GroundTruth):
        return value

    normalized = str(value).strip().lower()

    try:
        return GroundTruth(normalized)
    except ValueError as exc:
        valid = [label.value for label in GroundTruth]
        raise ValueError(
            f"Task '{task_id}' has invalid ground_truth '{value}'. " f"Valid: {valid}"
        ) from exc


def _parse_threat_label(
    value: ThreatLabel | str,
    task_id: str,
) -> ThreatLabel:
    if isinstance(value, ThreatLabel):
        return value

    normalized = str(value).strip().lower()

    try:
        return ThreatLabel(normalized)
    except ValueError as exc:
        valid = [label.value for label in ThreatLabel]
        raise ValueError(
            f"Task '{task_id}' has invalid threat_label '{value}'. " f"Valid: {valid}"
        ) from exc


def _parse_framework(value: Framework | str, task_id: str) -> Framework:
    if isinstance(value, Framework):
        return value

    normalized = str(value).strip().lower()

    try:
        return Framework(normalized)
    except ValueError as exc:
        valid = [framework.value for framework in Framework]
        raise ValueError(
            f"Task '{task_id}' has invalid framework '{value}'. " f"Valid: {valid}"
        ) from exc


def _parse_model_provider(
    value: ModelProvider | str,
    task_id: str,
) -> ModelProvider:
    if isinstance(value, ModelProvider):
        return value

    normalized = str(value).strip().lower()

    try:
        return ModelProvider(normalized)
    except ValueError as exc:
        valid = [provider.value for provider in ModelProvider]
        raise ValueError(
            f"Task '{task_id}' has invalid model_provider '{value}'. " f"Valid: {valid}"
        ) from exc


def _parse_tags(value: Any, task_id: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise TypeError(f"Task '{task_id}' tags must be a list")

    tags: list[str] = []
    for tag in value:
        tag_text = str(tag).strip()
        if not tag_text:
            raise ValueError(f"Task '{task_id}' contains an empty tag")
        tags.append(tag_text)

    return tags


def _require_non_empty_action_field(
    action: dict[str, Any],
    field: str,
    task_id: str,
    action_index: int,
) -> None:
    value = action.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Task '{task_id}' action[{action_index}] field '{field}' "
            "must be a non-empty string"
        )
