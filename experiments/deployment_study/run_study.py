"""
AISec Deployment Study — Main Study Runner.

Runs the complete AISec benchmark/deployment-study pipeline:

    1. Load and validate task definitions from tasks.yaml
    2. Select experimental groups
    3. Run tasks through selected baseline modes
    4. Compute metrics per baseline
    5. Compare AISec full pipeline against baselines
    6. Export a research artifact bundle

Outputs:
    events.jsonl
    events.csv
    metrics.json
    comparison.json
    summary.md
    manifest.json
    audit_log.jsonl if AISec full pipeline is run
    per_baseline/events_<mode>.jsonl
    per_baseline/events_<mode>.csv

Usage:
    python experiments/deployment_study/run_study.py
    python experiments/deployment_study/run_study.py --quiet
    python experiments/deployment_study/run_study.py --baseline aisec_full
    python experiments/deployment_study/run_study.py --group A B
    python experiments/deployment_study/run_study.py --output experiments/deployment_study/results/run_001
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make direct script execution work from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.deployment_study.baselines import create_baseline, run_task
from experiments.deployment_study.exporter import (
    export_events_csv,
    export_events_jsonl,
    export_study_results,
)
from experiments.deployment_study.labeler import load_tasks, validate_task_file
from experiments.deployment_study.metrics import compare_baselines, compute_metrics
from experiments.deployment_study.schemas import (
    BaselineMode,
    ExperimentGroup,
    Framework,
    GroundTruth,
    ModelProvider,
    StudyEvent,
    StudyMetrics,
    TaskDefinition,
    ThreatLabel,
)

STUDY_FRAMEWORK_VERSION = "1.6.0"
BENCHMARK_LABEL = "AISec-AgentRiskBench v0.2"

BASELINE_ORDER = [
    BaselineMode.NONE,
    BaselineMode.STATIC_RULES,
    BaselineMode.PROMPT_ONLY,
    BaselineMode.AISEC_FULL,
]

GROUP_MAP = {
    "A": ExperimentGroup.A_BENIGN,
    "B": ExperimentGroup.B_INJECTION,
    "C": ExperimentGroup.C_RISKY,
    "D": ExperimentGroup.D_MULTIAGENT,
}

BASELINE_LABELS = {
    BaselineMode.NONE.value: "No Monitoring",
    BaselineMode.STATIC_RULES.value: "Static Rules Only",
    BaselineMode.PROMPT_ONLY.value: "Prompt Injection Only",
    BaselineMode.AISEC_FULL.value: "AISec Full Pipeline",
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AISec Deployment Study Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/deployment_study/run_study.py
  python experiments/deployment_study/run_study.py --quiet
  python experiments/deployment_study/run_study.py --baseline aisec_full
  python experiments/deployment_study/run_study.py --group A B
  python experiments/deployment_study/run_study.py --output experiments/deployment_study/results/run_001
        """,
    )

    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path("experiments/deployment_study/tasks.yaml"),
        help="Path to tasks.yaml.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. If omitted, a timestamped directory is created "
            "under experiments/deployment_study/results/."
        ),
    )

    parser.add_argument(
        "--baseline",
        nargs="+",
        choices=[mode.value for mode in BASELINE_ORDER],
        default=[mode.value for mode in BASELINE_ORDER],
        help="Baseline modes to run.",
    )

    parser.add_argument(
        "--group",
        nargs="+",
        choices=["A", "B", "C", "D"],
        default=["A", "B", "C", "D"],
        help="Experimental groups to include.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed value recorded in the study metadata.",
    )

    parser.add_argument(
        "--prompt-threshold",
        type=float,
        default=0.70,
        help="Prompt-only baseline threshold.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-event output.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output directory first if it already exists.",
    )

    parser.add_argument(
        "--allow-redaction-warnings",
        action="store_true",
        help="Do not fail export if sensitive-looking field names are detected.",
    )

    return parser.parse_args()


# =============================================================================
# Main study runner
# =============================================================================


def run_study(
    *,
    tasks_file: Path,
    output_dir: Path,
    baselines: list[str],
    groups: list[str],
    seed: int | None = 42,
    prompt_threshold: float = 0.70,
    quiet: bool = False,
    force: bool = False,
    fail_on_redaction_warning: bool = True,
) -> dict[str, StudyMetrics]:
    """
    Run the complete deployment-study benchmark.

    Returns:
        Mapping from baseline mode name to StudyMetrics.
    """
    _configure_quiet_logging()

    if output_dir.exists() and force:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    per_baseline_dir = output_dir / "per_baseline"
    per_baseline_dir.mkdir(parents=True, exist_ok=True)

    study_run_id = _make_study_run_id()
    aisec_version = _get_aisec_version()
    git_commit = _get_git_commit()

    all_tasks = load_tasks(tasks_file)
    selected_groups = {GROUP_MAP[group] for group in groups}
    selected_tasks = [task for task in all_tasks if task.group in selected_groups]

    if not selected_tasks:
        raise ValueError("No tasks selected. Check --group values.")

    task_summary = _summarize_tasks(selected_tasks)
    baseline_modes = [BaselineMode(mode) for mode in baselines]

    study_metadata = {
        "study_run_id": study_run_id,
        "study_framework_version": STUDY_FRAMEWORK_VERSION,
        "benchmark_label": BENCHMARK_LABEL,
        "tasks_file": str(tasks_file),
        "aisec_version": aisec_version,
        "git_commit": git_commit,
        "seed": seed,
        "selected_groups": groups,
        "selected_baselines": [mode.value for mode in baseline_modes],
        "started_at": _utc_now_iso(),
        "project_root": str(PROJECT_ROOT),
    }

    _print_header(
        study_run_id=study_run_id,
        output_dir=output_dir,
        task_count=len(selected_tasks),
        action_count=sum(len(task.actions) for task in selected_tasks),
        groups=groups,
        baselines=[mode.value for mode in baseline_modes],
        aisec_version=aisec_version,
        git_commit=git_commit,
    )

    metrics_by_baseline: dict[str, StudyMetrics] = {}
    events_by_baseline: dict[str, list[StudyEvent]] = {}
    combined_events: list[StudyEvent] = []

    for mode in baseline_modes:
        baseline_name = mode.value
        log_path = output_dir / f"audit_{baseline_name}.jsonl"

        if log_path.exists():
            log_path.unlink()

        baseline = create_baseline(
            mode,
            log_path=log_path,
            prompt_threshold=prompt_threshold,
        )

        engine = getattr(baseline, "_engine", None)

        if not quiet:
            print(f"\nRunning baseline: {baseline_name}")
            print("-" * 72)

        baseline_events: list[StudyEvent] = []
        start_time = time.perf_counter()

        for task_index, task in enumerate(selected_tasks, start=1):
            task_run_id = f"{study_run_id}:{baseline_name}:{task.task_id}"

            try:
                task_events = run_task(
                    baseline=baseline,
                    task=task,
                    study_run_id=study_run_id,
                    task_run_id=task_run_id,
                    seed=seed,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Baseline '{baseline_name}' failed on task "
                    f"'{task.task_id}': {exc}"
                ) from exc

            baseline_events.extend(task_events)

            if not quiet:
                _print_task_progress(
                    task_index=task_index,
                    task=task,
                    events=task_events,
                )

        elapsed = time.perf_counter() - start_time

        metrics = compute_metrics(
            baseline_events,
            mode,
            engine=engine,
        )

        metrics_by_baseline[baseline_name] = metrics
        events_by_baseline[baseline_name] = baseline_events
        combined_events.extend(baseline_events)

        export_events_jsonl(
            baseline_events,
            per_baseline_dir / f"events_{baseline_name}.jsonl",
        )

        export_events_csv(
            baseline_events,
            per_baseline_dir / f"events_{baseline_name}.csv",
        )

        _print_metrics_summary(
            baseline_name=baseline_name,
            metrics=metrics,
            elapsed=elapsed,
        )

    comparison = compare_baselines(metrics_by_baseline)

    audit_log_path = output_dir / "audit_aisec_full.jsonl"
    if not audit_log_path.exists():
        audit_log_path = None

    manifest = export_study_results(
        events=combined_events,
        metrics_by_baseline=metrics_by_baseline,
        comparison=comparison,
        task_summary=task_summary,
        output_dir=output_dir,
        audit_log_path=audit_log_path,
        study_metadata=study_metadata,
        fail_on_redaction_warning=fail_on_redaction_warning,
    )

    write_run_metadata(output_dir / "study_metadata.json", study_metadata)

    _print_completion(
        output_dir=output_dir,
        metrics_by_baseline=metrics_by_baseline,
        manifest=manifest,
    )

    return metrics_by_baseline


# =============================================================================
# Helpers
# =============================================================================


def resolve_output_dir(output: Path | None) -> Path:
    if output is not None:
        return output

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("experiments/deployment_study/results") / f"run_{timestamp}"


def _make_study_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"aisec-study-{timestamp}-{suffix}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return "unknown"

    return result.stdout.strip() or "unknown"


def _get_aisec_version() -> str:
    try:
        from importlib.metadata import version

        return version("aisec")
    except Exception:
        pass

    try:
        import aisec

        return str(getattr(aisec, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _configure_quiet_logging() -> None:
    """
    Suppress noisy internal event-name logs during study runs.
    """
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("aisec").setLevel(logging.ERROR)

    try:
        from aisec.utils.logger import configure_logging

        configure_logging(level="ERROR", output="stderr")
    except Exception:
        pass


def _summarize_tasks(tasks: list[TaskDefinition]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(tasks),
        "total_actions": sum(len(task.actions) for task in tasks),
        "by_group": {},
        "by_ground_truth": {
            GroundTruth.SAFE.value: sum(
                1 for task in tasks if task.ground_truth == GroundTruth.SAFE
            ),
            GroundTruth.MALICIOUS.value: sum(
                1 for task in tasks if task.ground_truth == GroundTruth.MALICIOUS
            ),
        },
        "by_threat_label": {},
        "by_framework": {},
        "by_model_provider": {},
        "external_attack_tasks": sum(
            1 for task in tasks if bool(getattr(task, "is_external_attack", False))
        ),
    }

    for group in ExperimentGroup:
        group_tasks = [task for task in tasks if task.group == group]
        summary["by_group"][group.value] = {
            "count": len(group_tasks),
            "safe": sum(
                1 for task in group_tasks if task.ground_truth == GroundTruth.SAFE
            ),
            "malicious": sum(
                1 for task in group_tasks if task.ground_truth == GroundTruth.MALICIOUS
            ),
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


def write_run_metadata(output: Path, metadata: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_risk(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _decision_value(event: StudyEvent) -> str:
    decision = getattr(event, "decision", "")
    return str(getattr(decision, "value", decision))


def _event_status(event: StudyEvent) -> str:
    decision = _decision_value(event)

    if decision == "NOT_EVALUATED":
        return "NOT_EVAL"

    if event.was_blocked:
        return "BLOCK"

    if event.was_intercepted:
        return "HARD"

    if event.was_reviewed:
        return "REVIEW"

    return "ALLOW"


def _print_header(
    *,
    study_run_id: str,
    output_dir: Path,
    task_count: int,
    action_count: int,
    groups: list[str],
    baselines: list[str],
    aisec_version: str,
    git_commit: str,
) -> None:
    print()
    print("=" * 72)
    print("  AISec Deployment Study Framework")
    print("=" * 72)
    print(f"  Study run ID:   {study_run_id}")
    print(f"  AISec version:  {aisec_version}")
    print(f"  Git commit:     {git_commit}")
    print(f"  Tasks:          {task_count}")
    print(f"  Actions:        {action_count}")
    print(f"  Groups:         {', '.join(groups)}")
    print(f"  Baselines:      {', '.join(baselines)}")
    print(f"  Output:         {output_dir}")
    print("=" * 72)


def _print_task_progress(
    *,
    task_index: int,
    task: TaskDefinition,
    events: list[StudyEvent],
) -> None:
    statuses = ", ".join(
        f"{_event_status(event)}:{event.action_type}:risk={_format_risk(event.risk_score)}"
        for event in events
    )

    print(f"  [{task_index:03d}] {task.task_id} -> {statuses}")


def _print_metrics_summary(
    *,
    baseline_name: str,
    metrics: StudyMetrics,
    elapsed: float,
) -> None:
    print()
    print(f"Results for {baseline_name}:")
    print(f"  Events:        {metrics.total_events}")
    print(f"  Tasks:         {metrics.total_tasks}")
    print(f"  Precision:     {metrics.precision:.3f}")
    print(f"  Recall:        {metrics.recall:.3f}")
    print(f"  F1:            {metrics.f1_score:.3f}")
    print(f"  FPR:           {metrics.false_positive_rate:.3f}")
    print(f"  FNR:           {metrics.false_negative_rate:.3f}")
    print(f"  Group B inj.:  {metrics.group_b_detection_rate:.3f}")
    print(f"  Group C enf.:  {metrics.group_c_enforcement_rate:.3f}")
    print(f"  Group D corr.: {metrics.group_d_correlation_rate:.3f}")
    print(f"  Mean latency:  {metrics.latency_mean_ms:.3f} ms")
    print(f"  Elapsed:       {elapsed:.2f} s")


def _print_completion(
    *,
    output_dir: Path,
    metrics_by_baseline: dict[str, StudyMetrics],
    manifest: dict[str, Any],
) -> None:
    print()
    print("=" * 72)
    print("  Study complete")
    print("=" * 72)

    if BaselineMode.AISEC_FULL.value in metrics_by_baseline:
        aisec = metrics_by_baseline[BaselineMode.AISEC_FULL.value]
        print("  AISec full pipeline:")
        print(f"    Precision: {aisec.precision:.3f}")
        print(f"    Recall:    {aisec.recall:.3f}")
        print(f"    F1:        {aisec.f1_score:.3f}")
        print()

    if BaselineMode.AISEC_FULL.value in metrics_by_baseline:
        aisec = metrics_by_baseline[BaselineMode.AISEC_FULL.value]
        print("  Baseline comparison:")

        for baseline_name, metrics in metrics_by_baseline.items():
            if baseline_name == BaselineMode.AISEC_FULL.value:
                continue

            recall_delta = aisec.recall - metrics.recall
            f1_delta = aisec.f1_score - metrics.f1_score

            print(
                f"    vs {baseline_name:<28} "
                f"recall {recall_delta:+.3f}  "
                f"F1 {f1_delta:+.3f}"
            )

    print()
    print("  Output files:")

    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue

        size_kb = path.stat().st_size / 1024
        relative = path.relative_to(output_dir)
        print(f"    {str(relative):<48} {size_kb:>8.1f} KB")

    file_count = len(manifest.get("files", {}))

    print()
    print(f"  Manifest files checksummed: {file_count}")
    print("=" * 72)
    print()


# =============================================================================
# Entrypoint
# =============================================================================


def main() -> None:
    args = parse_args()

    # Validate full task file before filtering.
    validate_task_file(args.tasks_file)

    output_dir = resolve_output_dir(args.output)

    run_study(
        tasks_file=args.tasks_file,
        output_dir=output_dir,
        baselines=args.baseline,
        groups=args.group,
        seed=args.seed,
        prompt_threshold=args.prompt_threshold,
        quiet=args.quiet,
        force=args.force,
        fail_on_redaction_warning=not args.allow_redaction_warnings,
    )


if __name__ == "__main__":
    main()
