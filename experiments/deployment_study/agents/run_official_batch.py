"""
Official batch runner for the AISec v1.7 real-agent evaluation.

This runner is intentionally conservative.

It does not freeze the experiment.
It does not compute final metrics.
It does not delete failures.
It does not silently skip malformed runs.

Its job is to:
    1. load the validated official task manifest,
    2. build a reproducible run plan,
    3. optionally execute each task repeatedly through the real-agent runner,
    4. preserve completed runs, no-tool cases, provider errors, and execution errors,
    5. write a batch summary suitable for later metrics analysis.

Design principle:
    A failed run is data. It must be preserved, not hidden.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.deployment_study.agents.validate_real_agent_tasks import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    ManifestValidationError,
    load_manifest,
    validate_file,
)
from experiments.deployment_study.agents.run_real_agent import run_once  # noqa: E402
from experiments.deployment_study.schemas import (  # noqa: E402
    ExperimentGroup,
    GroundTruth,
    ThreatLabel,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments" / "deployment_study" / "results" / "official_real_agent"
)

BATCH_RUNNER_VERSION = "1.0"


# =============================================================================
# Helpers
# =============================================================================


def utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def short_hash(value: str, length: int = 8) -> str:
    """Return a short stable hash for identifiers."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    commit = result.stdout.strip()
    return commit or "unknown"


def get_git_status_short() -> str:
    """Return short git status for reproducibility metadata."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"

    return result.stdout.strip()


def ensure_json_serializable(value: Any) -> Any:
    """Convert arbitrary values into JSON-safe public records."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(key): ensure_json_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [ensure_json_serializable(item) for item in value]

    if hasattr(value, "to_json_dict") and callable(value.to_json_dict):
        return ensure_json_serializable(value.to_json_dict())

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return ensure_json_serializable(value.to_dict())

    return repr(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(ensure_json_serializable(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def expand_runner_return(value: Any) -> Any:
    """
    Expand run_once return values into JSON-safe records.

    The current real-agent runner returns a Path to a JSONL file. For later
    analysis, the batch summary should preserve both the path and the records.
    """
    if not isinstance(value, Path):
        return value

    payload: dict[str, Any] = {
        "result_path": str(value),
        "result_file_exists": value.exists(),
        "records": [],
        "record_count": 0,
    }

    if not value.exists() or not value.is_file():
        return payload

    if value.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with value.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    records.append(
                        {
                            "record_parse_error": True,
                            "line_number": line_number,
                            "error": exc.msg,
                            "raw_line": stripped,
                        }
                    )
                    continue

                if isinstance(parsed, dict):
                    records.append(parsed)
                else:
                    records.append({"non_object_record": parsed})

        payload["records"] = records
        payload["record_count"] = len(records)
        return payload

    if value.suffix.lower() == ".json":
        try:
            parsed = json.loads(value.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload["json_parse_error"] = exc.msg
            return payload

        payload["records"] = parsed if isinstance(parsed, list) else [parsed]
        payload["record_count"] = len(payload["records"])
        return payload

    payload["text_preview"] = value.read_text(
        encoding="utf-8", errors="replace"
    )[:2000]
    return payload


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True)
class BatchRunConfig:
    """Configuration for one official batch run."""

    manifest_path: Path
    output_root: Path
    repetitions: int
    model_provider: str
    model_name: str
    injection_policy: str
    study_run_id: str
    live: bool = False
    allow_candidate_manifest: bool = False
    no_execute_all: bool = False
    task_ids: tuple[str, ...] = ()
    limit: int | None = None
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")

        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1 when provided")

        if not self.study_run_id.strip():
            raise ValueError("study_run_id must be non-empty")


@dataclass(frozen=True)
class PlannedTaskRun:
    """One planned task execution."""

    run_index: int
    task_id: str
    task_group: str
    repetition_id: int
    task: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "task_id": self.task_id,
            "task_group": self.task_group,
            "repetition_id": self.repetition_id,
        }


@dataclass
class TaskRunResult:
    """Result of one planned task execution."""

    run_index: int
    task_id: str
    task_group: str
    repetition_id: int
    status: str
    started_at: str
    finished_at: str
    duration_ms: float
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    runner_return: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "task_id": self.task_id,
            "task_group": self.task_group,
            "repetition_id": self.repetition_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 3),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "runner_return": ensure_json_serializable(self.runner_return),
        }


@dataclass
class BatchRunSummary:
    """Summary of a dry plan or live batch run."""

    batch_runner_version: str
    study_run_id: str
    mode: str
    manifest_path: str
    manifest_sha256: str
    manifest_status: str
    manifest_task_count: int
    repetitions: int
    planned_run_count: int
    completed_run_count: int
    failed_run_count: int
    model_provider: str
    model_name: str
    injection_policy: str
    git_commit: str
    git_status_short: str
    output_dir: str
    started_at: str
    finished_at: str
    duration_ms: float
    group_counts: dict[str, int]
    task_ids: list[str]
    results: list[TaskRunResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_runner_version": self.batch_runner_version,
            "study_run_id": self.study_run_id,
            "mode": self.mode,
            "manifest": {
                "path": self.manifest_path,
                "sha256": self.manifest_sha256,
                "status": self.manifest_status,
                "task_count": self.manifest_task_count,
            },
            "configuration": {
                "repetitions": self.repetitions,
                "planned_run_count": self.planned_run_count,
                "model_provider": self.model_provider,
                "model_name": self.model_name,
                "injection_policy": self.injection_policy,
            },
            "execution": {
                "completed_run_count": self.completed_run_count,
                "failed_run_count": self.failed_run_count,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration_ms": round(self.duration_ms, 3),
            },
            "reproducibility": {
                "git_commit": self.git_commit,
                "git_status_short": self.git_status_short,
            },
            "output_dir": self.output_dir,
            "group_counts": self.group_counts,
            "task_ids": self.task_ids,
            "results": [result.to_dict() for result in self.results],
        }


# =============================================================================
# Planning
# =============================================================================


def make_study_run_id(prefix: str = "aisec-v1.7-official") -> str:
    """Create a stable readable study-run identifier."""
    now = utc_now_iso()
    random_part = uuid.uuid4().hex[:8]
    return f"{prefix}-{short_hash(now + random_part)}"


def select_tasks(
    *,
    manifest: Mapping[str, Any],
    task_ids: Sequence[str] = (),
    limit: int | None = None,
) -> list[Mapping[str, Any]]:
    """Select tasks from the manifest by optional task IDs and limit."""
    tasks = list(manifest["tasks"])

    if task_ids:
        requested = set(task_ids)
        tasks = [task for task in tasks if task["task_id"] in requested]

        found = {task["task_id"] for task in tasks}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"requested task_id values not found: {missing}")

    if limit is not None:
        tasks = tasks[:limit]

    return tasks


def build_run_plan(
    *,
    tasks: Sequence[Mapping[str, Any]],
    repetitions: int,
) -> list[PlannedTaskRun]:
    """Build a deterministic task-run plan."""
    plan: list[PlannedTaskRun] = []
    run_index = 0

    for repetition_id in range(1, repetitions + 1):
        for task_item in tasks:
            run_index += 1
            plan.append(
                PlannedTaskRun(
                    run_index=run_index,
                    task_id=str(task_item["task_id"]),
                    task_group=str(task_item["task_group"]),
                    repetition_id=repetition_id,
                    task=task_item,
                )
            )

    return plan


def make_output_dir(config: BatchRunConfig) -> Path:
    """Return the output directory for this batch run."""
    return config.output_root / config.study_run_id


def write_run_plan(
    *,
    output_dir: Path,
    config: BatchRunConfig,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    plan: Sequence[PlannedTaskRun],
) -> Path:
    """Write the planned runs to disk."""
    path = output_dir / "run_plan.json"
    payload = {
        "study_run_id": config.study_run_id,
        "mode": "live" if config.live else "dry_plan",
        "manifest": {
            "path": str(config.manifest_path),
            "sha256": manifest_hash,
            "status": manifest["status"],
            "task_count": manifest["task_count"],
        },
        "configuration": {
            "repetitions": config.repetitions,
            "model_provider": config.model_provider,
            "model_name": config.model_name,
            "injection_policy": config.injection_policy,
            "no_execute_all": config.no_execute_all,
            "limit": config.limit,
            "task_ids": list(config.task_ids),
        },
        "planned_run_count": len(plan),
        "planned_runs": [item.to_dict() for item in plan],
    }
    write_json(path, payload)
    return path


# =============================================================================
# Runner invocation
# =============================================================================


def _runner_kwargs_for_task(
    *,
    planned: PlannedTaskRun,
    config: BatchRunConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Create candidate keyword arguments for run_real_agent.run_once."""
    task = planned.task
    execute_allowed = bool(task["execute_allowed_tools"]) and not config.no_execute_all

    return {
        "prompt": task["prompt"],
        "dry_run": False,
        "study_run_id": config.study_run_id,
        "task_id": task["task_id"],
        "task_group": task["task_group"],
        "repetition_id": planned.repetition_id,
        "group": ExperimentGroup(str(task["task_group"])),
        "ground_truth": GroundTruth(str(task["ground_truth"])),
        "threat_label": ThreatLabel(str(task["threat_label"])),
        "agent_id": (
            f"{config.study_run_id}:{planned.task_id}:r{planned.repetition_id}"
        ),
        "output_dir": output_dir,
        "execute_allowed_tools": execute_allowed,
        "execute_tools": execute_allowed,
        "no_execute": not execute_allowed,
        "injection_policy": task.get("injection_policy", config.injection_policy),
        "model_provider": config.model_provider,
        "model_name": config.model_name,
        "live": True,
    }


def call_runner_once(
    *,
    runner: Callable[..., Any],
    planned: PlannedTaskRun,
    config: BatchRunConfig,
    output_dir: Path,
) -> Any:
    """
    Call the existing real-agent runner.

    The real-agent runner has evolved during v1.7 development. To avoid fragile
    coupling, this function inspects the runner signature and passes only
    supported keyword arguments unless the runner accepts **kwargs.
    """
    candidate_kwargs = _runner_kwargs_for_task(
        planned=planned,
        config=config,
        output_dir=output_dir,
    )

    signature = inspect.signature(runner)
    parameters = signature.parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_var_kwargs:
        kwargs = candidate_kwargs
    else:
        kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in parameters
        }

    missing_required = []
    for name, parameter in parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }:
            continue

        if parameter.default is not inspect.Parameter.empty:
            continue

        if name not in kwargs:
            missing_required.append(name)

    if missing_required:
        raise TypeError(
            "run_once signature requires unsupported parameters: "
            f"{missing_required}. Supported candidate kwargs: "
            f"{sorted(candidate_kwargs)}"
        )

    return runner(**kwargs)


def run_planned_task(
    *,
    runner: Callable[..., Any],
    planned: PlannedTaskRun,
    config: BatchRunConfig,
    output_dir: Path,
) -> TaskRunResult:
    """Execute one planned task and preserve success or failure."""
    started_at = utc_now_iso()
    start = time.perf_counter()

    try:
        runner_return = call_runner_once(
            runner=runner,
            planned=planned,
            config=config,
            output_dir=output_dir,
        )
        runner_return = expand_runner_return(runner_return)
    except Exception as exc:
        finished_at = utc_now_iso()
        duration_ms = (time.perf_counter() - start) * 1000.0
        return TaskRunResult(
            run_index=planned.run_index,
            task_id=planned.task_id,
            task_group=planned.task_group,
            repetition_id=planned.repetition_id,
            status="error",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
            runner_return=None,
        )

    finished_at = utc_now_iso()
    duration_ms = (time.perf_counter() - start) * 1000.0
    return TaskRunResult(
        run_index=planned.run_index,
        task_id=planned.task_id,
        task_group=planned.task_group,
        repetition_id=planned.repetition_id,
        status="completed",
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        runner_return=runner_return,
    )


# =============================================================================
# Batch execution
# =============================================================================


def _validate_execution_gate(
    *,
    manifest: Mapping[str, Any],
    config: BatchRunConfig,
) -> None:
    """
    Prevent accidental official execution from an unfrozen manifest.

    Candidate manifests may be dry-planned. Live execution from a candidate
    manifest requires an explicit override.
    """
    status = str(manifest.get("status", ""))

    if not config.live:
        return

    if status == "frozen":
        return

    if config.allow_candidate_manifest:
        return

    raise ManifestValidationError(
        "refusing live execution because manifest status is "
        f"{status!r}. Freeze the manifest or pass --allow-candidate-manifest "
        "for a clearly labeled non-official pilot run."
    )


def run_batch(
    *,
    config: BatchRunConfig,
    runner: Callable[..., Any] = run_once,
) -> BatchRunSummary:
    """Run or dry-plan a batch evaluation."""
    started_at = utc_now_iso()
    start = time.perf_counter()

    manifest = validate_file(config.manifest_path)
    _validate_execution_gate(manifest=manifest, config=config)

    manifest_hash = file_sha256(config.manifest_path)
    selected_tasks = select_tasks(
        manifest=manifest,
        task_ids=config.task_ids,
        limit=config.limit,
    )
    plan = build_run_plan(tasks=selected_tasks, repetitions=config.repetitions)

    output_dir = make_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        output_dir / "manifest_snapshot.json",
        manifest,
    )
    write_run_plan(
        output_dir=output_dir,
        config=config,
        manifest=manifest,
        manifest_hash=manifest_hash,
        plan=plan,
    )

    results: list[TaskRunResult] = []

    if config.live:
        for planned in plan:
            result = run_planned_task(
                runner=runner,
                planned=planned,
                config=config,
                output_dir=output_dir,
            )
            results.append(result)

            if result.status == "error" and config.fail_fast:
                break

    finished_at = utc_now_iso()
    duration_ms = (time.perf_counter() - start) * 1000.0

    completed = sum(1 for result in results if result.status == "completed")
    failed = sum(1 for result in results if result.status == "error")

    group_counts = dict(
        sorted(collections.Counter(task["task_group"] for task in selected_tasks).items())
    )

    summary = BatchRunSummary(
        batch_runner_version=BATCH_RUNNER_VERSION,
        study_run_id=config.study_run_id,
        mode="live" if config.live else "dry_plan",
        manifest_path=str(config.manifest_path),
        manifest_sha256=manifest_hash,
        manifest_status=str(manifest["status"]),
        manifest_task_count=int(manifest["task_count"]),
        repetitions=config.repetitions,
        planned_run_count=len(plan),
        completed_run_count=completed,
        failed_run_count=failed,
        model_provider=config.model_provider,
        model_name=config.model_name,
        injection_policy=config.injection_policy,
        git_commit=get_git_commit(),
        git_status_short=get_git_status_short(),
        output_dir=str(output_dir),
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        group_counts=group_counts,
        task_ids=[str(task["task_id"]) for task in selected_tasks],
        results=results,
    )

    write_json(output_dir / "batch_summary.json", summary.to_dict())
    return summary


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-plan the AISec v1.7 official real-agent batch."
    )

    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to official task manifest JSON.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where official batch outputs will be written.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions per selected task.",
    )
    parser.add_argument(
        "--study-run-id",
        default="",
        help="Optional explicit study run ID. Generated when omitted.",
    )
    parser.add_argument(
        "--model-provider",
        default="groq",
        help="Model provider label to record.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        help="Model name to use/record.",
    )
    parser.add_argument(
        "--injection-policy",
        default="review",
        choices=["record_only", "review", "block"],
        help="Prompt-injection enforcement policy.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Run only a specific task ID. Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected tasks.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Actually execute model calls through run_real_agent.run_once.",
    )
    parser.add_argument(
        "--allow-candidate-manifest",
        action="store_true",
        help=(
            "Allow live execution when manifest status is candidate_not_frozen. "
            "Use only for clearly labeled pilot/debug runs."
        ),
    )
    parser.add_argument(
        "--no-execute-all",
        action="store_true",
        help="Force sandbox tools not to execute even when the manifest allows them.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first task execution error.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    study_run_id = args.study_run_id.strip() or make_study_run_id()

    try:
        config = BatchRunConfig(
            manifest_path=Path(args.manifest),
            output_root=Path(args.output_root),
            repetitions=args.repetitions,
            model_provider=args.model_provider,
            model_name=args.model_name,
            injection_policy=args.injection_policy,
            study_run_id=study_run_id,
            live=args.live,
            allow_candidate_manifest=args.allow_candidate_manifest,
            no_execute_all=args.no_execute_all,
            task_ids=tuple(args.task_id),
            limit=args.limit,
            fail_fast=args.fail_fast,
        )
        summary = run_batch(config=config)
    except Exception as exc:
        print(f"batch runner failed: {exc}", file=sys.stderr)
        return 1

    print("batch runner completed")
    print(f"mode: {summary.mode}")
    print(f"study_run_id: {summary.study_run_id}")
    print(f"manifest_status: {summary.manifest_status}")
    print(f"planned_run_count: {summary.planned_run_count}")
    print(f"completed_run_count: {summary.completed_run_count}")
    print(f"failed_run_count: {summary.failed_run_count}")
    print(f"output_dir: {summary.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())