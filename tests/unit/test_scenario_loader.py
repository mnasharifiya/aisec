"""
Unit tests for the YAML scenario loader.
Run with: pytest tests/unit/test_scenario_loader.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aisec.scenarios.loader import (
    LoadedScenario,
    RuleCondition,
    ScenarioLoader,
    YAMLRuleDefinition,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def loader() -> ScenarioLoader:
    return ScenarioLoader()


def _write_scenario(tmp_path: Path, data: dict) -> Path:
    """Write a scenario dict to a YAML file and return the path."""
    path = tmp_path / "test_scenario.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def _valid_scenario(**overrides) -> dict:
    """Return a minimal valid scenario definition."""
    base = {
        "scenario_id": "test_ai",
        "display_name": "Test AI",
        "version": "1.0.0",
        "description": "Test scenario",
        "bias": -0.05,
        "weights": {
            "action_type_encoding": 0.125,
            "keyword_risk_score": 0.125,
            "frequency_score": 0.125,
            "api_call_flag": 0.125,
            "file_access_flag": 0.125,
            "network_access_flag": 0.125,
            "sensitive_path_flag": 0.125,
            "privilege_flag": 0.125,
        },
        "rules": [
            {
                "id": "TEST-001",
                "name": "Test Block",
                "description": "Test rule",
                "action_types": ["dangerous_action"],
                "decision": "BLOCK",
                "reason": "Test block reason",
                "condition": None,
            }
        ],
    }
    base.update(overrides)
    return base


# ── Loader tests ──────────────────────────────────────────────────────────────


class TestScenarioLoader:

    def test_loads_valid_scenario(self, loader: ScenarioLoader, tmp_path: Path) -> None:
        path = _write_scenario(tmp_path, _valid_scenario())
        scenario = loader.load(path)
        assert scenario.scenario_id == "test_ai"
        assert scenario.display_name == "Test AI"
        assert len(scenario.rules) == 1

    def test_raises_for_missing_file(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.yaml")

    def test_raises_for_missing_required_field(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        data = _valid_scenario()
        del data["scenario_id"]
        path = _write_scenario(tmp_path, data)
        with pytest.raises(ValueError, match="scenario_id"):
            loader.load(path)

    def test_raises_for_invalid_scenario_id_format(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        path = _write_scenario(
            tmp_path, _valid_scenario(scenario_id="Invalid-ID-With-Caps")
        )
        with pytest.raises(ValueError, match="lowercase"):
            loader.load(path)

    def test_raises_for_weights_not_summing_to_one(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        data = _valid_scenario()
        data["weights"]["action_type_encoding"] = 0.9
        path = _write_scenario(tmp_path, data)
        with pytest.raises(ValueError, match="sum to approximately 1.0"):
            loader.load(path)

    def test_raises_for_weight_out_of_range(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        data = _valid_scenario()
        data["weights"]["action_type_encoding"] = 1.5
        path = _write_scenario(tmp_path, data)
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            loader.load(path)

    def test_raises_for_invalid_decision(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        data = _valid_scenario()
        data["rules"][0]["decision"] = "INVALID_DECISION"
        path = _write_scenario(tmp_path, data)
        with pytest.raises(ValueError, match="invalid decision"):
            loader.load(path)

    def test_raises_for_invalid_rule_id_format(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        data = _valid_scenario()
        data["rules"][0]["id"] = "invalid-id"
        path = _write_scenario(tmp_path, data)
        with pytest.raises(ValueError, match="SCENARIO-NNN"):
            loader.load(path)

    def test_raises_for_duplicate_rule_ids(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        data = _valid_scenario()
        data["rules"].append(dict(data["rules"][0]))
        path = _write_scenario(tmp_path, data)
        with pytest.raises(ValueError, match="Duplicate rule ID"):
            loader.load(path)

    def test_caches_loaded_scenario(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        path = _write_scenario(tmp_path, _valid_scenario())
        loader.load(path)
        assert "test_ai" in loader.list_loaded()

    def test_get_returns_cached_scenario(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        path = _write_scenario(tmp_path, _valid_scenario())
        loader.load(path)
        scenario = loader.get("test_ai")
        assert scenario is not None
        assert scenario.scenario_id == "test_ai"

    def test_get_returns_none_for_unknown_id(self, loader: ScenarioLoader) -> None:
        assert loader.get("nonexistent") is None

    def test_load_all_loads_multiple_files(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        for sid in ["alpha_ai", "beta_ai"]:
            data = _valid_scenario(
                scenario_id=sid,
                display_name=f"{sid} display",
            )
            (tmp_path / f"{sid}.yaml").write_text(yaml.dump(data), encoding="utf-8")
        scenarios = loader.load_all(tmp_path)
        assert len(scenarios) == 2

    def test_load_all_returns_empty_for_missing_dir(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        result = loader.load_all(tmp_path / "nonexistent")
        assert result == []

    def test_loads_real_trading_scenario(self, loader: ScenarioLoader) -> None:
        """Load the actual trading_ai.yaml from the scenarios directory."""
        path = Path("scenarios") / "trading_ai.yaml"
        if not path.exists():
            pytest.skip("scenarios/trading_ai.yaml not found")
        scenario = loader.load(path)
        assert scenario.scenario_id == "trading_ai"
        assert len(scenario.rules) >= 4
        assert len(scenario.weights) == 8

    def test_loads_real_urban_scenario(self, loader: ScenarioLoader) -> None:
        path = Path("scenarios") / "urban_ai.yaml"
        if not path.exists():
            pytest.skip("scenarios/urban_ai.yaml not found")
        scenario = loader.load(path)
        assert scenario.scenario_id == "urban_ai"
        assert len(scenario.rules) >= 4

    def test_loads_real_healthcare_scenario(self, loader: ScenarioLoader) -> None:
        path = Path("scenarios") / "healthcare_ai.yaml"
        if not path.exists():
            pytest.skip("scenarios/healthcare_ai.yaml not found")
        scenario = loader.load(path)
        assert scenario.scenario_id == "healthcare_ai"
        assert len(scenario.rules) >= 4


# ── LoadedScenario tests ──────────────────────────────────────────────────────


class TestLoadedScenario:

    def test_get_weight_vector_returns_8_values(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        path = _write_scenario(tmp_path, _valid_scenario())
        scenario = loader.load(path)
        vector = scenario.get_weight_vector()
        assert len(vector) == 8
        assert all(isinstance(v, float) for v in vector)

    def test_get_rules_for_action_returns_matching_rules(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        path = _write_scenario(tmp_path, _valid_scenario())
        scenario = loader.load(path)
        rules = scenario.get_rules_for_action("dangerous_action")
        assert len(rules) == 1
        assert rules[0].id == "TEST-001"

    def test_get_rules_for_unknown_action_returns_empty(
        self, loader: ScenarioLoader, tmp_path: Path
    ) -> None:
        path = _write_scenario(tmp_path, _valid_scenario())
        scenario = loader.load(path)
        rules = scenario.get_rules_for_action("safe_action")
        assert rules == []


# ── RuleCondition tests ───────────────────────────────────────────────────────


class TestRuleCondition:

    def _make_rule(self, operator: str, value: any) -> YAMLRuleDefinition:
        """Helper — create a rule with a condition for testing."""
        return YAMLRuleDefinition(
            id="TEST-001",
            name="x",
            description="x",
            action_types=("x",),
            decision="BLOCK",
            reason="x",
            condition=RuleCondition(
                field="amount" if operator != "in" else "target",
                operator=operator,
                value=value,
            ),
        )

    def test_eq_operator(self) -> None:
        rule = YAMLRuleDefinition(
            id="TEST-001",
            name="x",
            description="x",
            action_types=("x",),
            decision="BLOCK",
            reason="x",
            condition=RuleCondition(field="after_hours", operator="eq", value=True),
        )
        assert rule.evaluate_condition({"after_hours": True}) is True
        assert rule.evaluate_condition({"after_hours": False}) is False

    def test_gte_operator(self) -> None:
        rule = self._make_rule("gte", 1_000_000)
        assert rule.evaluate_condition({"amount": 2_000_000}) is True
        assert rule.evaluate_condition({"amount": 1_000_000}) is True
        assert rule.evaluate_condition({"amount": 999_999}) is False

    def test_gt_operator(self) -> None:
        rule = YAMLRuleDefinition(
            id="TEST-001",
            name="x",
            description="x",
            action_types=("x",),
            decision="BLOCK",
            reason="x",
            condition=RuleCondition(field="count", operator="gt", value=50),
        )
        assert rule.evaluate_condition({"count": 51}) is True
        assert rule.evaluate_condition({"count": 50}) is False

    def test_in_operator(self) -> None:
        rule = self._make_rule("in", ["emergency_dispatch", "ambulance_routing"])
        assert rule.evaluate_condition({"target": "emergency_dispatch"}) is True
        assert rule.evaluate_condition({"target": "traffic_sensor"}) is False

    def test_missing_field_returns_false(self) -> None:
        rule = self._make_rule("gte", 1_000_000)
        assert rule.evaluate_condition({}) is False

    def test_invalid_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid operator"):
            RuleCondition(field="x", operator="invalid", value=1)

    def test_type_error_in_comparison_returns_false(self) -> None:
        rule = self._make_rule("gte", 1000)
        assert rule.evaluate_condition({"amount": "not_a_number"}) is False


# ── YAMLRuleDefinition tests ──────────────────────────────────────────────────


class TestYAMLRuleDefinition:

    def test_valid_rule_creates_successfully(self) -> None:
        rule = YAMLRuleDefinition(
            id="TEST-001",
            name="Test Rule",
            description="A test rule",
            action_types=("dangerous_action",),
            decision="BLOCK",
            reason="Test reason",
        )
        assert rule.id == "TEST-001"
        assert rule.decision == "BLOCK"

    def test_rejects_invalid_decision(self) -> None:
        with pytest.raises(ValueError, match="invalid decision"):
            YAMLRuleDefinition(
                id="TEST-001",
                name="x",
                description="x",
                action_types=("x",),
                decision="INVALID",
                reason="x",
            )

    def test_rejects_invalid_rule_id_format(self) -> None:
        with pytest.raises(ValueError, match="SCENARIO-NNN"):
            YAMLRuleDefinition(
                id="bad-id",
                name="x",
                description="x",
                action_types=("x",),
                decision="BLOCK",
                reason="x",
            )

    def test_matches_action_returns_true_for_listed_action(self) -> None:
        rule = YAMLRuleDefinition(
            id="TEST-001",
            name="x",
            description="x",
            action_types=("execute_trade", "execute_large_trade"),
            decision="BLOCK",
            reason="x",
        )
        assert rule.matches_action("execute_trade") is True
        assert rule.matches_action("execute_large_trade") is True
        assert rule.matches_action("read_market_data") is False

    def test_evaluate_condition_none_always_true(self) -> None:
        rule = YAMLRuleDefinition(
            id="TEST-001",
            name="x",
            description="x",
            action_types=("x",),
            decision="BLOCK",
            reason="x",
            condition=None,
        )
        assert rule.evaluate_condition({}) is True
        assert rule.evaluate_condition({"any": "data"}) is True


# ── Policy signing tests ─────────────────────────────────────────────────────


class TestPolicySigner:

    def test_sign_and_verify_unmodified_file(
        self,
        tmp_path: Path,
    ) -> None:
        from aisec.scenarios.loader import PolicySigner

        path = _write_scenario(tmp_path, _valid_scenario())

        signer = PolicySigner(secret_key="a" * 32)
        signer.sign_file(path)

        assert signer.verify_file(path) is True

    def test_verify_fails_after_modification(
        self,
        tmp_path: Path,
    ) -> None:
        from aisec.scenarios.loader import PolicySigner

        path = _write_scenario(tmp_path, _valid_scenario())

        signer = PolicySigner(secret_key="a" * 32)
        signer.sign_file(path)

        # Tamper with the file after signing.
        path.write_text(
            path.read_text(encoding="utf-8") + "\n# tampered\n",
            encoding="utf-8",
        )

        assert signer.verify_file(path) is False

    def test_verify_returns_false_when_sig_missing(
        self,
        tmp_path: Path,
    ) -> None:
        from aisec.scenarios.loader import PolicySigner

        path = _write_scenario(tmp_path, _valid_scenario())

        signer = PolicySigner(secret_key="a" * 32)

        assert signer.verify_file(path) is False

    def test_rejects_short_key(self) -> None:
        from aisec.scenarios.loader import PolicySigner

        with pytest.raises(ValueError, match="32 characters"):
            PolicySigner(secret_key="short")

    def test_different_keys_different_signatures(
        self,
        tmp_path: Path,
    ) -> None:
        from aisec.scenarios.loader import PolicySigner

        path = _write_scenario(tmp_path, _valid_scenario())

        signer1 = PolicySigner(secret_key="a" * 32)
        signer2 = PolicySigner(secret_key="b" * 32)

        sig1 = signer1.sign_file(path)
        sig2 = signer2.sign_file(path)

        assert sig1 != sig2


# ── Drone scenario tests ─────────────────────────────────────────────────────


class TestDroneScenario:

    def test_loads_drone_scenario(
        self,
        loader: ScenarioLoader,
    ) -> None:
        path = Path("scenarios") / "autonomous_drone.yaml"

        if not path.exists():
            pytest.skip("scenarios/autonomous_drone.yaml not found")

        scenario = loader.load(path)

        assert scenario.scenario_id == "autonomous_drone"
        assert len(scenario.rules) >= 6

    def test_drone_geofence_rule_present(
        self,
        loader: ScenarioLoader,
    ) -> None:
        path = Path("scenarios") / "autonomous_drone.yaml"

        if not path.exists():
            pytest.skip("scenarios/autonomous_drone.yaml not found")

        scenario = loader.load(path)
        rule_ids = [rule.id for rule in scenario.rules]

        assert "DRONE-001" in rule_ids
        assert "DRONE-004" in rule_ids  # Kill switch

    def test_drone_killswitch_rule_blocks(
        self,
        loader: ScenarioLoader,
    ) -> None:
        path = Path("scenarios") / "autonomous_drone.yaml"

        if not path.exists():
            pytest.skip("scenarios/autonomous_drone.yaml not found")

        scenario = loader.load(path)

        ks_rules = [rule for rule in scenario.rules if rule.id == "DRONE-004"]

        assert len(ks_rules) == 1
        assert ks_rules[0].decision == "BLOCK"
