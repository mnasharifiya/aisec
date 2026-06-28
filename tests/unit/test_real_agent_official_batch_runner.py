"""
Unit tests for the AISec v1.7 official real-agent batch runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.deployment_study.agents.run_official_batch import (
    BatchRunConfig,
    ManifestValidationError,
    build_run_plan,
    call_runner_once,
    load_manifest,
    make_study_run_id,
    run_batch,
    select_tasks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    PROJECT_ROOT / "experiments" / "deployment_study" / "real_agent_tasks_v1.json"
)


def _manifest() -> dict[str, Any]:
    return load_manifest(MANIFEST_PATH)


def _config(tmp_path: Path, *, live: bool = False, repetitions: int = 1) -> BatchRunConfig:
    return BatchRunConfig(
        manifest_path=MANIFEST_PATH,
        output_root=tmp_path,
        repetitions=repetitions,
        model_provider="groq",
        model_name="llama-3.3-70b-versatile",
        injection_policy="review",
        study_run_id="unit-test-study",
        live=live,
        allow_candidate_manifest=False,
        no_execute_all=False,
    )


def test_make_study_run_id_has_prefix() -> None:
    study_run_id = make_study_run_id()

    assert study_run_id.startswith("aisec-v1.7-official-")


def test_select_tasks_returns_all_manifest_tasks() -> None:
    tasks = select_tasks(manifest=_manifest())

    assert len(tasks) == 50


def test_select_tasks_filters_by_task_id() -> None:
    manifest = _manifest()

    selected = select_tasks(
        manifest=manifest,
        task_ids=("official_A_safe_trade_001",),
    )

    assert len(selected) == 1
    assert selected[0]["task_id"] == "official_A_safe_trade_001"


def test_select_tasks_rejects_missing_task_id() -> None:
    with pytest.raises(ValueError, match="not found"):
        select_tasks(manifest=_manifest(), task_ids=("missing-task",))


def test_select_tasks_limit() -> None:
    selected = select_tasks(manifest=_manifest(), limit=3)

    assert len(selected) == 3


def test_build_run_plan_multiplies_by_repetitions() -> None:
    tasks = select_tasks(manifest=_manifest(), limit=5)

    plan = build_run_plan(tasks=tasks, repetitions=3)

    assert len(plan) == 15
    assert plan[0].run_index == 1
    assert plan[-1].run_index == 15
    assert plan[-1].repetition_id == 3


def test_dry_plan_writes_summary_without_running_model(tmp_path: Path) -> None:
    config = _config(tmp_path, live=False, repetitions=1)

    summary = run_batch(config=config, runner=lambda **_: pytest.fail("runner called"))

    assert summary.mode == "dry_plan"
    assert summary.planned_run_count == 50
    assert summary.completed_run_count == 0
    assert summary.failed_run_count == 0

    summary_path = Path(summary.output_dir) / "batch_summary.json"
    plan_path = Path(summary.output_dir) / "run_plan.json"
    snapshot_path = Path(summary.output_dir) / "manifest_snapshot.json"

    assert summary_path.exists()
    assert plan_path.exists()
    assert snapshot_path.exists()


def test_live_candidate_manifest_requires_explicit_override(tmp_path: Path) -> None:
    config = _config(tmp_path, live=True, repetitions=1)

    with pytest.raises(ManifestValidationError, match="refusing live execution"):
        run_batch(config=config, runner=lambda **_: None)


def test_live_candidate_manifest_can_be_allowed_for_pilot(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"ok": True, "task_id": kwargs.get("task_id")}

    config = BatchRunConfig(
        manifest_path=MANIFEST_PATH,
        output_root=tmp_path,
        repetitions=1,
        model_provider="groq",
        model_name="llama-3.3-70b-versatile",
        injection_policy="review",
        study_run_id="unit-test-study",
        live=True,
        allow_candidate_manifest=True,
        no_execute_all=True,
        limit=2,
    )

    summary = run_batch(config=config, runner=fake_runner)

    assert summary.mode == "live"
    assert summary.planned_run_count == 2
    assert summary.completed_run_count == 2
    assert summary.failed_run_count == 0
    assert len(calls) == 2
    assert all(call["no_execute"] is True for call in calls)


def test_call_runner_once_filters_kwargs_for_strict_signature(tmp_path: Path) -> None:
    manifest = _manifest()
    task = manifest["tasks"][0]
    plan = build_run_plan(tasks=[task], repetitions=1)
    config = _config(tmp_path, live=True, repetitions=1)

    def strict_runner(prompt: str, task_id: str, no_execute: bool) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "task_id": task_id,
            "no_execute": no_execute,
        }

    result = call_runner_once(
        runner=strict_runner,
        planned=plan[0],
        config=config,
        output_dir=tmp_path,
    )

    assert result["task_id"] == "official_A_safe_trade_001"
    assert result["no_execute"] is False


def test_call_runner_once_reports_missing_required_parameter(tmp_path: Path) -> None:
    manifest = _manifest()
    task = manifest["tasks"][0]
    plan = build_run_plan(tasks=[task], repetitions=1)
    config = _config(tmp_path, live=True, repetitions=1)

    def incompatible_runner(required_unknown: str) -> None:
        raise AssertionError("should not be called")

    with pytest.raises(TypeError, match="requires unsupported parameters"):
        call_runner_once(
            runner=incompatible_runner,
            planned=plan[0],
            config=config,
            output_dir=tmp_path,
        )


def test_live_errors_are_preserved_not_hidden(tmp_path: Path) -> None:
    def failing_runner(**_: Any) -> None:
        raise RuntimeError("provider exploded")

    config = BatchRunConfig(
        manifest_path=MANIFEST_PATH,
        output_root=tmp_path,
        repetitions=1,
        model_provider="groq",
        model_name="llama-3.3-70b-versatile",
        injection_policy="review",
        study_run_id="unit-test-study",
        live=True,
        allow_candidate_manifest=True,
        no_execute_all=True,
        limit=1,
    )

    summary = run_batch(config=config, runner=failing_runner)

    assert summary.completed_run_count == 0
    assert summary.failed_run_count == 1
    assert summary.results[0].status == "error"
    assert summary.results[0].error_type == "RuntimeError"
    assert "provider exploded" in str(summary.results[0].error_message)


def test_summary_json_contains_reproducibility_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path, live=False, repetitions=1)

    summary = run_batch(config=config)

    summary_path = Path(summary.output_dir) / "batch_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert data["study_run_id"] == "unit-test-study"
    assert data["manifest"]["sha256"]
    assert "git_commit" in data["reproducibility"]
    assert "git_status_short" in data["reproducibility"]