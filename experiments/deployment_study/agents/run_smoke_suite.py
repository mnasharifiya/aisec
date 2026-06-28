"""
AISec v1.7 real-agent smoke-suite runner.

This runner executes a fixed manifest of live real-agent smoke tasks and
validates observed JSONL outputs against expected outcomes.

Important research boundary:
    This smoke suite is for regression validation only. It is not the
    official 100/500/1000-run evaluation and must not be reported as the
    main benchmark result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.deployment_study.agents.run_real_agent import (
    DEFAULT_OUTPUT_DIR,
    InjectionPolicy,
    run_once,
)
from experiments.deployment_study.schemas import (
    ExperimentGroup,
    GroundTruth,
    ThreatLabel,
)


DEFAULT_TASK_FILE = Path("experiments/deployment_study/real_agent_smoke_tasks.json")


@dataclass(frozen=True)
class SmokeTask:
    """One manifest-defined smoke task."""

    task_id: str
    task_class: str
    task_group: str
    ground_truth: str
    threat_label: str
    execute_allowed_tools: bool
    injection_policy: str
    prompt: str
    expected: Dict[str, Any]


@dataclass(frozen=True)
class SmokeManifest:
    """Metadata and tasks for the real-agent smoke suite."""

    manifest_version: str
    protocol_version: str
    suite_type: str
    status: str
    not_for_official_metric_reporting: bool
    purpose: str
    default_model_provider: str
    default_model_name: str
    default_injection_policy: str
    tasks: List[SmokeTask]


@dataclass(frozen=True)
class SmokeValidationResult:
    """Validation result for one smoke task."""

    task_id: str
    task_class: str
    passed: bool
    reason: str
    output_path: str | None
    observed: Dict[str, Any]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    """Return SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into records."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require_fields(
    *,
    item: Mapping[str, Any],
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(item))
    if missing:
        raise ValueError(f"{context} is missing fields: {missing}")


def load_smoke_manifest(path: Path) -> SmokeManifest:
    """
    Load and validate the smoke manifest.

    The manifest must be an object with metadata and a tasks list. A raw list
    is intentionally rejected because it lacks research context.
    """
    raw = read_json(path)

    if not isinstance(raw, dict):
        raise ValueError(
            "Smoke manifest must be a JSON object with metadata and tasks."
        )

    required_manifest_fields = {
        "manifest_version",
        "protocol_version",
        "suite_type",
        "status",
        "not_for_official_metric_reporting",
        "purpose",
        "default_model_provider",
        "default_model_name",
        "default_injection_policy",
        "tasks",
    }
    _require_fields(
        item=raw,
        required=required_manifest_fields,
        context="Smoke manifest",
    )

    if raw["suite_type"] != "real_agent_smoke_regression":
        raise ValueError(
            "suite_type must be 'real_agent_smoke_regression' for this runner."
        )

    if raw["not_for_official_metric_reporting"] is not True:
        raise ValueError(
            "Smoke manifest must explicitly set "
            "not_for_official_metric_reporting=true."
        )

    if not isinstance(raw["tasks"], list) or not raw["tasks"]:
        raise ValueError("Smoke manifest tasks must be a non-empty list.")

    required_task_fields = {
        "task_id",
        "task_class",
        "task_group",
        "ground_truth",
        "threat_label",
        "execute_allowed_tools",
        "injection_policy",
        "prompt",
        "expected",
    }

    tasks: List[SmokeTask] = []
    seen_task_ids: set[str] = set()

    for index, item in enumerate(raw["tasks"]):
        if not isinstance(item, dict):
            raise ValueError(f"Task at index {index} must be a JSON object.")

        _require_fields(
            item=item,
            required=required_task_fields,
            context=f"Smoke task at index {index}",
        )

        task_id = str(item["task_id"])
        if task_id in seen_task_ids:
            raise ValueError(f"Duplicate task_id in smoke manifest: {task_id}")
        seen_task_ids.add(task_id)

        expected = item["expected"]
        if not isinstance(expected, dict):
            raise ValueError(f"Task {task_id} expected field must be an object.")

        tasks.append(
            SmokeTask(
                task_id=task_id,
                task_class=str(item["task_class"]),
                task_group=str(item["task_group"]),
                ground_truth=str(item["ground_truth"]),
                threat_label=str(item["threat_label"]),
                execute_allowed_tools=bool(item["execute_allowed_tools"]),
                injection_policy=str(item["injection_policy"]),
                prompt=str(item["prompt"]),
                expected=dict(expected),
            )
        )

    return SmokeManifest(
        manifest_version=str(raw["manifest_version"]),
        protocol_version=str(raw["protocol_version"]),
        suite_type=str(raw["suite_type"]),
        status=str(raw["status"]),
        not_for_official_metric_reporting=bool(
            raw["not_for_official_metric_reporting"]
        ),
        purpose=str(raw["purpose"]),
        default_model_provider=str(raw["default_model_provider"]),
        default_model_name=str(raw["default_model_name"]),
        default_injection_policy=str(raw["default_injection_policy"]),
        tasks=tasks,
    )


def _proposal_records(records: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [record for record in records if record.get("record_type") == "real_agent_proposal"]


def _study_event_records(records: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        record
        for record in records
        if record.get("record_type") == "real_agent_study_event"
    ]


def _compare_expected(
    *,
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    key: str,
) -> str | None:
    if key not in expected:
        return None

    if observed.get(key) != expected[key]:
        return f"{key}: expected {expected[key]!r}, observed {observed.get(key)!r}"

    return None


def validate_smoke_output(
    *,
    task: SmokeTask,
    output_path: Path,
) -> SmokeValidationResult:
    """Validate one smoke-task JSONL output against manifest expectations."""
    records = read_jsonl(output_path)
    proposal_records = _proposal_records(records)
    study_records = _study_event_records(records)

    if not proposal_records:
        return SmokeValidationResult(
            task_id=task.task_id,
            task_class=task.task_class,
            passed=False,
            reason="missing real_agent_proposal record",
            output_path=str(output_path),
            observed={},
        )

    proposal = proposal_records[0]["data"]
    expected = task.expected

    proposed_tool_calls = proposal.get("proposed_tool_calls", [])
    proposed_count = len(proposed_tool_calls)
    expected_min = int(expected.get("proposed_tool_calls_min", 0))

    if proposed_count < expected_min:
        return SmokeValidationResult(
            task_id=task.task_id,
            task_class=task.task_class,
            passed=False,
            reason=(
                f"proposed_tool_calls_min: expected at least {expected_min}, "
                f"observed {proposed_count}"
            ),
            output_path=str(output_path),
            observed={"proposed_tool_calls": proposed_count},
        )

    expected_study_events = expected.get("study_events")
    if expected_study_events is not None and len(study_records) != expected_study_events:
        return SmokeValidationResult(
            task_id=task.task_id,
            task_class=task.task_class,
            passed=False,
            reason=(
                f"study_events: expected {expected_study_events}, "
                f"observed {len(study_records)}"
            ),
            output_path=str(output_path),
            observed={"study_events": len(study_records)},
        )

    if not study_records:
        return SmokeValidationResult(
            task_id=task.task_id,
            task_class=task.task_class,
            passed=False,
            reason="missing real_agent_study_event record",
            output_path=str(output_path),
            observed={"proposed_tool_calls": proposed_count},
        )

    first_event = study_records[0]
    data = first_event["data"]
    execution = first_event.get("execution", {})
    enforcement = first_event.get("enforcement", {})

    observed = {
        "proposed_tool_calls": proposed_count,
        "study_events": len(study_records),
        "action_type": data.get("action_type"),
        "decision": data.get("decision"),
        "rule_hits": data.get("rule_hits", []),
        "injection_detected": data.get("injection_detected"),
        "injection_confidence": data.get("injection_confidence"),
        "sandbox_executed": execution.get("sandbox_executed"),
        "base_decision": enforcement.get("base_decision"),
        "final_decision": enforcement.get("final_decision"),
        "injection_policy": enforcement.get("injection_policy"),
        "git_commit": data.get("git_commit"),
        "model_provider": data.get("model_provider"),
        "model_name": data.get("model_name"),
    }

    if observed["decision"] != observed["final_decision"]:
        return SmokeValidationResult(
            task_id=task.task_id,
            task_class=task.task_class,
            passed=False,
            reason=(
                "decision/final_decision mismatch: "
                f"decision={observed['decision']!r}, "
                f"final_decision={observed['final_decision']!r}"
            ),
            output_path=str(output_path),
            observed=observed,
        )

    for key in ("action_type", "decision", "injection_detected", "sandbox_executed"):
        mismatch = _compare_expected(expected=expected, observed=observed, key=key)
        if mismatch:
            return SmokeValidationResult(
                task_id=task.task_id,
                task_class=task.task_class,
                passed=False,
                reason=mismatch,
                output_path=str(output_path),
                observed=observed,
            )

    expected_rule_hit = expected.get("rule_hit")
    if expected_rule_hit is not None and expected_rule_hit not in observed["rule_hits"]:
        return SmokeValidationResult(
            task_id=task.task_id,
            task_class=task.task_class,
            passed=False,
            reason=(
                f"rule_hit: expected {expected_rule_hit!r} in "
                f"{observed['rule_hits']!r}"
            ),
            output_path=str(output_path),
            observed=observed,
        )

    return SmokeValidationResult(
        task_id=task.task_id,
        task_class=task.task_class,
        passed=True,
        reason="passed",
        output_path=str(output_path),
        observed=observed,
    )


def run_smoke_suite(
    *,
    manifest: SmokeManifest,
    live: bool,
    output_dir: Path,
    suite_id: str,
    repetition_id: int,
    fail_fast: bool,
) -> List[SmokeValidationResult]:
    """Run all smoke tasks and validate outputs."""
    if not live:
        raise ValueError("Real-agent smoke suite requires --live.")

    results: List[SmokeValidationResult] = []

    for task in manifest.tasks:
        print(f"\n=== Running {task.task_id} ({task.task_class}) ===")

        try:
            output_path = run_once(
                prompt=task.prompt,
                dry_run=False,
                study_run_id=suite_id,
                task_id=task.task_id,
                task_group=task.task_group,
                repetition_id=repetition_id,
                group=ExperimentGroup(task.task_group),
                ground_truth=GroundTruth(task.ground_truth),
                threat_label=ThreatLabel(task.threat_label),
                agent_id="langchain_groq_agent",
                output_dir=output_dir,
                execute_allowed_tools=task.execute_allowed_tools,
                injection_policy=InjectionPolicy(task.injection_policy),
            )

            validation = validate_smoke_output(task=task, output_path=output_path)

        except Exception as exc:
            validation = SmokeValidationResult(
                task_id=task.task_id,
                task_class=task.task_class,
                passed=False,
                reason=f"runner_exception: {type(exc).__name__}: {exc}",
                output_path=None,
                observed={},
            )

        results.append(validation)

        status = "PASS" if validation.passed else "FAIL"
        print(f"[{status}] {task.task_id}: {validation.reason}")
        print(json.dumps(validation.observed, sort_keys=True))

        if fail_fast and not validation.passed:
            break

    return results


def write_summary(
    *,
    path: Path,
    manifest: SmokeManifest,
    manifest_path: Path,
    manifest_hash: str,
    suite_id: str,
    results: List[SmokeValidationResult],
) -> None:
    """Write a machine-readable smoke-suite summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    passed = sum(1 for result in results if result.passed)
    total = len(results)

    summary = {
        "record_type": "real_agent_smoke_suite_summary",
        "timestamp_utc": utc_now_iso(),
        "suite_id": suite_id,
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest": {
            "manifest_version": manifest.manifest_version,
            "protocol_version": manifest.protocol_version,
            "suite_type": manifest.suite_type,
            "status": manifest.status,
            "not_for_official_metric_reporting": (
                manifest.not_for_official_metric_reporting
            ),
            "purpose": manifest.purpose,
            "default_model_provider": manifest.default_model_provider,
            "default_model_name": manifest.default_model_name,
            "default_injection_policy": manifest.default_injection_policy,
        },
        "results": [asdict(result) for result in results],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run AISec v1.7 live real-agent smoke suite."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Required. Runs live Groq/LangChain smoke tasks.",
    )
    parser.add_argument(
        "--task-file",
        default=str(DEFAULT_TASK_FILE),
        help="Path to smoke task manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for JSONL outputs.",
    )
    parser.add_argument(
        "--suite-id",
        default=f"aisec-real-agent-smoke-live-{uuid.uuid4().hex[:8]}",
        help="Shared study_run_id for all smoke tasks.",
    )
    parser.add_argument("--repetition-id", type=int, default=0)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed smoke task.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if not args.live:
        raise SystemExit("Use --live to run the real-agent smoke suite.")

    manifest_path = Path(args.task_file)
    manifest = load_smoke_manifest(manifest_path)
    manifest_hash = file_sha256(manifest_path)

    results = run_smoke_suite(
        manifest=manifest,
        live=args.live,
        output_dir=Path(args.output_dir),
        suite_id=args.suite_id,
        repetition_id=args.repetition_id,
        fail_fast=args.fail_fast,
    )

    summary_path = Path(args.output_dir) / f"{args.suite_id}_summary.json"
    write_summary(
        path=summary_path,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        suite_id=args.suite_id,
        results=results,
    )

    passed = sum(1 for result in results if result.passed)
    total = len(results)

    print("\n=== AISec real-agent smoke-suite summary ===")
    print(f"suite_id: {args.suite_id}")
    print(f"passed: {passed}/{total}")
    print(f"summary: {summary_path}")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()