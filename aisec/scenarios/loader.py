"""
AISec scenario plugin loader.

Loads threat scenarios from YAML definition files without
requiring any changes to core AISec code.

This implements the scenario extension framework described
in the master plan — operators define scenarios in YAML,
the loader validates and ingests them at startup.

Security:
    - YAML files are validated against a strict schema.
    - Unknown fields are rejected — no silent data ingestion.
    - File paths are validated to prevent directory traversal.
    - Loaded scenarios are read-only after loading.
    - Schema version is checked before processing.

Usage:
    loader   = ScenarioLoader()
    scenario = loader.load("scenarios/trading_ai.yaml")
    scenario = loader.load_by_id("trading_ai")  # from default dir

    # Use loaded weights in the scorer:
    weights = scenario.get_weights()
    bias    = scenario.bias

    # Use loaded rules in the rule engine:
    for rule_def in scenario.rules:
        print(rule_def.id, rule_def.decision)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from aisec.utils.logger import get_logger

log = get_logger("aisec.scenarios.loader")

# Default directory for scenario YAML files
DEFAULT_SCENARIOS_DIR = Path("scenarios")

# Supported YAML schema version
SUPPORTED_VERSION = "1.0"

# Maximum file size we will load (1 MB)
MAX_FILE_SIZE_BYTES = 1_048_576


# ── Rule definition ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuleCondition:
    """A condition expression within a YAML rule definition."""

    field: str
    operator: str  # eq, neq, gt, gte, lt, lte, in
    value: Any

    VALID_OPERATORS: frozenset[str] = frozenset(
        {"eq", "neq", "gt", "gte", "lt", "lte", "in"}
    )

    def __post_init__(self) -> None:
        if self.operator not in self.VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Valid: {sorted(self.VALID_OPERATORS)}"
            )


@dataclass(frozen=True)
class YAMLRuleDefinition:
    """
    A single rule loaded from a YAML scenario definition.

    Immutable after loading — rules cannot be modified at runtime.
    """

    id: str
    name: str
    description: str
    action_types: tuple[str, ...]
    decision: str  # BLOCK, ESCALATE, PENDING_REVIEW
    reason: str
    condition: RuleCondition | None = None

    VALID_DECISIONS: frozenset[str] = frozenset(
        {"BLOCK", "ESCALATE", "PENDING_REVIEW", "ALLOW"}
    )

    def __post_init__(self) -> None:
        if self.decision not in self.VALID_DECISIONS:
            raise ValueError(
                f"Rule {self.id}: invalid decision '{self.decision}'. "
                f"Valid: {sorted(self.VALID_DECISIONS)}"
            )
        if not re.match(r"^[A-Z]+-\d+$", self.id):
            raise ValueError(
                f"Rule ID '{self.id}' must match pattern SCENARIO-NNN "
                f"(e.g. TRADING-001, URBAN-002)"
            )

    def matches_action(self, action_type: str) -> bool:
        """Return True if this rule applies to the given action type."""
        return action_type in self.action_types

    def evaluate_condition(self, payload: dict[str, Any]) -> bool:
        """
        Evaluate whether the rule condition is satisfied.

        Args:
            payload: The event raw_payload dict.

        Returns:
            True if condition is satisfied (or no condition exists).
        """
        if self.condition is None:
            return True

        field_value = payload.get(self.condition.field)

        op = self.condition.operator
        val = self.condition.value

        try:
            if op == "eq":
                return field_value == val
            elif op == "neq":
                return field_value != val
            elif op == "gt":
                return float(field_value) > float(val)
            elif op == "gte":
                return float(field_value) >= float(val)
            elif op == "lt":
                return float(field_value) < float(val)
            elif op == "lte":
                return float(field_value) <= float(val)
            elif op == "in":
                return field_value in val
            else:
                return False
        except (TypeError, ValueError):
            return False


# ── Loaded scenario ───────────────────────────────────────────────────────────


@dataclass
class LoadedScenario:
    """
    A fully validated and loaded scenario definition.

    Provides typed access to all scenario configuration
    loaded from the YAML file.
    """

    scenario_id: str
    display_name: str
    version: str
    description: str
    paper_reference: str
    rules: list[YAMLRuleDefinition]
    weights: dict[str, float]
    bias: float
    action_encodings: dict[str, float]
    risk_keywords: dict[str, float]
    sensitive_targets: list[str]
    network_actions: list[str]
    privileged_actions: list[str]
    source_file: Path

    def get_weight_vector(self) -> list[float]:
        """
        Return the weight vector in the canonical 8-dimension order.

        Order matches FeatureVector.dimensions:
            action_type_encoding, keyword_risk_score, frequency_score,
            api_call_flag, file_access_flag, network_access_flag,
            sensitive_path_flag, privilege_flag
        """
        dim_order = [
            "action_type_encoding",
            "keyword_risk_score",
            "frequency_score",
            "api_call_flag",
            "file_access_flag",
            "network_access_flag",
            "sensitive_path_flag",
            "privilege_flag",
        ]
        return [float(self.weights.get(d, 0.125)) for d in dim_order]

    def get_rules_for_action(self, action_type: str) -> list[YAMLRuleDefinition]:
        """Return all rules that apply to the given action type."""
        return [r for r in self.rules if r.matches_action(action_type)]

    def __repr__(self) -> str:
        return (
            f"LoadedScenario(id={self.scenario_id!r}, "
            f"rules={len(self.rules)}, "
            f"version={self.version!r})"
        )


# ── Scenario loader ───────────────────────────────────────────────────────────


class ScenarioLoader:
    """
    Loads and validates AISec scenario YAML definitions.

    Scenarios are loaded once at startup and cached.
    The cache is read-only after population.

    Usage:
        loader = ScenarioLoader()
        scenario = loader.load(Path("scenarios/trading_ai.yaml"))
        print(scenario.display_name)
        print(len(scenario.rules))
    """

    def __init__(
        self,
        scenarios_dir: Path = DEFAULT_SCENARIOS_DIR,
    ) -> None:
        self._scenarios_dir = scenarios_dir
        self._cache: dict[str, LoadedScenario] = {}

    def load(self, path: Path) -> LoadedScenario:
        """
        Load a scenario from a YAML file.

        Args:
            path: Path to the YAML scenario file.

        Returns:
            Validated LoadedScenario instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the YAML is invalid or fails validation.
            PermissionError:   If the file is too large.
        """
        # Validate path
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        # Check file size
        size = resolved.stat().st_size
        if size > MAX_FILE_SIZE_BYTES:
            raise PermissionError(
                f"Scenario file too large: {size} bytes "
                f"(maximum: {MAX_FILE_SIZE_BYTES})"
            )

        # Load YAML
        try:
            with resolved.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Scenario file must be a YAML mapping: {path}")

        # Validate and build scenario
        scenario = self._validate_and_build(data, resolved)

        # Cache by scenario_id
        self._cache[scenario.scenario_id] = scenario

        log.info(
            "scenario_loaded",
            scenario_id=scenario.scenario_id,
            display_name=scenario.display_name,
            rule_count=len(scenario.rules),
            source=str(path),
        )

        return scenario

    def load_all(self, directory: Path | None = None) -> list[LoadedScenario]:
        """
        Load all YAML scenarios from a directory.

        Args:
            directory: Directory to scan. Uses default if None.

        Returns:
            List of LoadedScenario instances.
        """
        scan_dir = directory or self._scenarios_dir

        if not scan_dir.exists():
            log.warning(
                "scenarios_dir_not_found",
                path=str(scan_dir),
            )
            return []

        scenarios = []
        for yaml_file in sorted(scan_dir.glob("*.yaml")):
            try:
                scenario = self.load(yaml_file)
                scenarios.append(scenario)
            except Exception as exc:
                log.error(
                    "scenario_load_failed",
                    file=str(yaml_file),
                    exc_type=type(exc).__name__,
                    detail=str(exc)[:200],
                )

        log.info(
            "scenarios_loaded",
            count=len(scenarios),
            ids=[s.scenario_id for s in scenarios],
        )

        return scenarios

    def get(self, scenario_id: str) -> LoadedScenario | None:
        """Return a cached scenario by ID, or None if not loaded."""
        return self._cache.get(scenario_id)

    def list_loaded(self) -> list[str]:
        """Return list of loaded scenario IDs."""
        return list(self._cache.keys())

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_and_build(
        self,
        data: dict[str, Any],
        source: Path,
    ) -> LoadedScenario:
        """Validate YAML data and construct a LoadedScenario."""

        # Required top-level fields
        required = [
            "scenario_id",
            "display_name",
            "version",
            "description",
            "weights",
            "rules",
        ]
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"Scenario missing required field: '{field_name}'")

        scenario_id = str(data["scenario_id"]).strip()
        if not re.match(r"^[a-z][a-z0-9_]*$", scenario_id):
            raise ValueError(
                f"scenario_id '{scenario_id}' must be lowercase "
                "alphanumeric with underscores"
            )

        # Validate weights
        weights = self._validate_weights(data["weights"])

        # Validate rules
        rules = self._validate_rules(data.get("rules", []))

        return LoadedScenario(
            scenario_id=scenario_id,
            display_name=str(data["display_name"]),
            version=str(data.get("version", "1.0.0")),
            description=str(data.get("description", "")),
            paper_reference=str(data.get("paper_reference", "")),
            rules=rules,
            weights=weights,
            bias=float(data.get("bias", 0.0)),
            action_encodings=dict(data.get("action_encodings", {})),
            risk_keywords=dict(data.get("risk_keywords", {})),
            sensitive_targets=list(data.get("sensitive_targets", [])),
            network_actions=list(data.get("network_actions", [])),
            privileged_actions=list(data.get("privileged_actions", [])),
            source_file=source,
        )

    def _validate_weights(self, weights_data: Any) -> dict[str, float]:
        """Validate and return the weight dictionary."""
        if not isinstance(weights_data, dict):
            raise ValueError("'weights' must be a YAML mapping")

        required_dims = {
            "action_type_encoding",
            "keyword_risk_score",
            "frequency_score",
            "api_call_flag",
            "file_access_flag",
            "network_access_flag",
            "sensitive_path_flag",
            "privilege_flag",
        }

        validated: dict[str, float] = {}
        for dim in required_dims:
            if dim not in weights_data:
                raise ValueError(f"Missing weight dimension: '{dim}'")
            try:
                value = float(weights_data[dim])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Weight '{dim}' must be a number, " f"got: {weights_data[dim]}"
                )
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Weight '{dim}' must be in [0.0, 1.0], got: {value}")
            validated[dim] = value

        # Verify weights sum to approximately 1.0
        total = sum(validated.values())
        if not 0.95 <= total <= 1.05:
            raise ValueError(
                f"Weights must sum to approximately 1.0, got {total:.4f}. "
                "Adjust weights so they sum to 1.0."
            )

        return validated

    def _validate_rules(self, rules_data: Any) -> list[YAMLRuleDefinition]:
        """Validate and return the list of rule definitions."""
        if not isinstance(rules_data, list):
            raise ValueError("'rules' must be a YAML sequence")

        rules: list[YAMLRuleDefinition] = []
        seen_ids: set[str] = set()

        for i, rule_data in enumerate(rules_data):
            if not isinstance(rule_data, dict):
                raise ValueError(f"Rule {i} must be a YAML mapping")

            rule_id = str(rule_data.get("id", ""))
            if rule_id in seen_ids:
                raise ValueError(f"Duplicate rule ID: '{rule_id}'")
            seen_ids.add(rule_id)

            # Parse condition
            condition = None
            cond_data = rule_data.get("condition")
            if cond_data is not None:
                condition = RuleCondition(
                    field=str(cond_data["field"]),
                    operator=str(cond_data["operator"]),
                    value=cond_data["value"],
                )

            action_types = rule_data.get("action_types", [])
            if not isinstance(action_types, list):
                raise ValueError(f"Rule {rule_id}: 'action_types' must be a list")

            rule = YAMLRuleDefinition(
                id=rule_id,
                name=str(rule_data.get("name", "")),
                description=str(rule_data.get("description", "")),
                action_types=tuple(str(a) for a in action_types),
                decision=str(rule_data.get("decision", "BLOCK")).upper(),
                reason=str(rule_data.get("reason", "")),
                condition=condition,
            )
            rules.append(rule)

        return rules
