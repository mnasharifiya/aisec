"""
AISec Deployment Study — Dataset Exporter.

Exports deployment-study results in reproducible, paper-ready formats.

Outputs:
    events.jsonl       — structured event dataset, one JSON object per line
    events.csv         — flat event dataset for statistical analysis
    metrics.json       — metrics by baseline
    comparison.json    — AISec-vs-baseline comparison
    summary.md         — human-readable study summary
    manifest.json      — reproducibility manifest with checksums
    audit_log.jsonl    — optional copy of AISec audit log

Design principle:
    Public research artifacts must be reproducible, sanitized, checksummed,
    and honest about limitations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.deployment_study.schemas import StudyEvent, StudyMetrics

EXPORTER_VERSION = "1.0"

DEFAULT_OUTPUT_FILES = {
    "events_jsonl": "events.jsonl",
    "events_csv": "events.csv",
    "metrics_json": "metrics.json",
    "comparison_json": "comparison.json",
    "summary_md": "summary.md",
    "manifest_json": "manifest.json",
    "audit_log": "audit_log.jsonl",
}

SENSITIVE_KEYWORDS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "credential",
    "jwt",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session",
    "token",
}

BASELINE_ORDER = [
    "baseline_none",
    "baseline_static_rules",
    "baseline_prompt_only",
    "aisec_full",
]

BASELINE_LABELS = {
    "baseline_none": "No Monitoring",
    "baseline_static_rules": "Static Rules Only",
    "baseline_prompt_only": "Prompt Injection Only",
    "aisec_full": "AISec Full Pipeline",
}

GROUP_DESCRIPTIONS = {
    "A": "Benign tasks / false-positive measurement",
    "B": "Prompt injection tasks",
    "C": "Risky tool-use tasks",
    "D": "Multi-agent coordination tasks",
}


# =============================================================================
# High-level export API
# =============================================================================


def export_study_results(
    *,
    events: list[StudyEvent],
    metrics_by_baseline: dict[str, StudyMetrics],
    comparison: dict[str, Any],
    task_summary: dict[str, Any],
    output_dir: Path,
    audit_log_path: Path | None = None,
    study_metadata: dict[str, Any] | None = None,
    fail_on_redaction_warning: bool = True,
) -> dict[str, Any]:
    """
    Export the complete deployment-study artifact bundle.

    Args:
        events:
            All StudyEvent records across baselines.

        metrics_by_baseline:
            Mapping from baseline mode to StudyMetrics.

        comparison:
            Baseline comparison dictionary from metrics.compare_baselines().

        task_summary:
            Summary dictionary from labeler.get_task_summary().

        output_dir:
            Directory where all artifacts will be written.

        audit_log_path:
            Optional AISec audit log path to copy into output_dir.

        study_metadata:
            Optional metadata such as study_run_id, aisec_version, git_commit,
            model names, benchmark version, and notes.

        fail_on_redaction_warning:
            If True, abort export when obvious sensitive fields are detected.

    Returns:
        Manifest dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study_metadata = dict(study_metadata or {})
    generated_at = utc_now_iso()

    redaction_warnings = scan_events_for_sensitive_content(events)
    if redaction_warnings and fail_on_redaction_warning:
        preview = "; ".join(redaction_warnings[:5])
        raise ValueError(
            "Sensitive-looking content detected in export data. "
            f"Refusing public export. Examples: {preview}"
        )

    paths = {
        key: output_dir / filename for key, filename in DEFAULT_OUTPUT_FILES.items()
    }

    event_count_jsonl = export_events_jsonl(events, paths["events_jsonl"])
    event_count_csv = export_events_csv(events, paths["events_csv"])
    export_metrics_json(metrics_by_baseline, paths["metrics_json"])
    export_comparison_json(comparison, paths["comparison_json"])
    export_summary_markdown(
        metrics_by_baseline=metrics_by_baseline,
        task_summary=task_summary,
        output=paths["summary_md"],
        study_metadata=study_metadata,
        event_count=len(events),
        redaction_warnings=redaction_warnings,
        audit_log_included=audit_log_path is not None and Path(audit_log_path).exists(),
    )

    audit_log_copied = False
    if audit_log_path is not None:
        audit_log_copied = export_audit_log(
            source=audit_log_path,
            output=paths["audit_log"],
            required=False,
        )

    exported_files = [
        paths["events_jsonl"],
        paths["events_csv"],
        paths["metrics_json"],
        paths["comparison_json"],
        paths["summary_md"],
    ]

    if audit_log_copied:
        exported_files.append(paths["audit_log"])

    manifest = build_manifest(
        output_dir=output_dir,
        exported_files=exported_files,
        generated_at=generated_at,
        events=events,
        metrics_by_baseline=metrics_by_baseline,
        task_summary=task_summary,
        comparison=comparison,
        study_metadata=study_metadata,
        redaction_warnings=redaction_warnings,
        audit_log_copied=audit_log_copied,
        event_count_jsonl=event_count_jsonl,
        event_count_csv=event_count_csv,
    )

    write_json(paths["manifest_json"], manifest)
    return manifest


# =============================================================================
# Individual exporters
# =============================================================================


def export_events_jsonl(events: list[StudyEvent], output: Path) -> int:
    """
    Export events as newline-delimited JSON.

    JSONL is the primary research dataset format.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="\n") as fh:
        for event in events:
            record = event_to_json_dict(event)
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    return len(events)


def export_events_csv(events: list[StudyEvent], output: Path) -> int:
    """
    Export events as flat CSV.

    CSV uses StudyEvent.to_flat_dict() when available.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    records = [event_to_flat_dict(event) for event in events]
    fieldnames = collect_fieldnames(records)

    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for record in records:
            writer.writerow(record)

    return len(events)


def export_metrics_json(
    metrics_by_baseline: dict[str, StudyMetrics],
    output: Path,
) -> None:
    """
    Export metrics by baseline as JSON.
    """
    export_data = {
        "generated_at": utc_now_iso(),
        "metrics_by_baseline": {
            name: metric_to_dict(metric)
            for name, metric in sorted(metrics_by_baseline.items())
        },
    }

    write_json(output, export_data)


def export_comparison_json(
    comparison: dict[str, Any],
    output: Path,
) -> None:
    """
    Export baseline comparison as JSON.
    """
    export_data = {
        "generated_at": utc_now_iso(),
        "comparison": make_json_safe(comparison),
    }

    write_json(output, export_data)


def export_audit_log(
    source: Path,
    output: Path,
    *,
    required: bool = False,
) -> bool:
    """
    Copy AISec audit log into the study artifact directory.

    Args:
        source: Original audit log path.
        output: Destination audit log path.
        required: If True, missing source raises FileNotFoundError.

    Returns:
        True if copied, False if source did not exist and required=False.
    """
    source = Path(source)

    if not source.exists():
        if required:
            raise FileNotFoundError(f"Audit log not found: {source}")
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    return True


def export_summary_markdown(
    *,
    metrics_by_baseline: dict[str, StudyMetrics],
    task_summary: dict[str, Any],
    output: Path,
    study_metadata: dict[str, Any] | None = None,
    event_count: int = 0,
    redaction_warnings: list[str] | None = None,
    audit_log_included: bool = False,
) -> None:
    """
    Export a human-readable Markdown study summary.

    Suitable for GitHub, paper appendices, and reviewer inspection.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    study_metadata = dict(study_metadata or {})
    redaction_warnings = list(redaction_warnings or [])

    now = datetime.now(timezone.utc)
    now_text = now.strftime("%Y-%m-%d %H:%M UTC")
    year = now.year

    benchmark_label = str(
        study_metadata.get("benchmark_label", "AISec-AgentRiskBench v0.2")
    )
    study_run_id = str(study_metadata.get("study_run_id", "unknown"))
    aisec_version = str(study_metadata.get("aisec_version", "unknown"))
    git_commit = str(study_metadata.get("git_commit", "unknown"))

    lines: list[str] = [
        "# AISec Deployment Study — Results Summary",
        "",
        f"Generated: {now_text}",
        "",
        "## Reproducibility Metadata",
        "",
        f"- Study run ID: `{study_run_id}`",
        f"- Benchmark: `{benchmark_label}`",
        f"- AISec version: `{aisec_version}`",
        f"- Git commit: `{git_commit}`",
        f"- Exporter version: `{EXPORTER_VERSION}`",
        f"- Total exported events: `{event_count}`",
        f"- Audit log included: `{audit_log_included}`",
        "",
        "## Important Limitation",
        "",
        (
            "This artifact currently represents a controlled benchmark run. "
            "If the task summary shows all frameworks and model providers as "
            "`simulated`, the results should be described as simulated benchmark "
            "evidence, not as a real-world production deployment study."
        ),
        "",
        "## Study Design",
        "",
        f"- Total tasks: {task_summary.get('total', 0)}",
        f"- Total actions: {task_summary.get('total_actions', 0)}",
        f"- Safe tasks: {task_summary.get('by_ground_truth', {}).get('safe', 0)}",
        f"- Malicious tasks: "
        f"{task_summary.get('by_ground_truth', {}).get('malicious', 0)}",
        f"- External attack tasks: {task_summary.get('external_attack_tasks', 0)}",
        "",
        "### Task Distribution by Group",
        "",
        "| Group | Description | Tasks | Safe | Malicious | Actions |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for group, description in GROUP_DESCRIPTIONS.items():
        group_data = task_summary.get("by_group", {}).get(group, {})
        lines.append(
            f"| {group} | {description} | "
            f"{group_data.get('count', 0)} | "
            f"{group_data.get('safe', 0)} | "
            f"{group_data.get('malicious', 0)} | "
            f"{group_data.get('actions', 0)} |"
        )

    lines.extend(
        [
            "",
            "### Framework and Model Provider Distribution",
            "",
            "**Frameworks**",
            "",
        ]
    )

    for name, count in sorted(task_summary.get("by_framework", {}).items()):
        if count:
            lines.append(f"- `{name}`: {count}")

    lines.extend(["", "**Model providers**", ""])

    for name, count in sorted(task_summary.get("by_model_provider", {}).items()):
        if count:
            lines.append(f"- `{name}`: {count}")

    lines.extend(
        [
            "",
            "## Results by Baseline Mode",
            "",
            "### Core Metrics",
            "",
            "| Mode | Events | Tasks | Precision | Recall | F1 | FPR | FNR | Accuracy |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for mode_key in BASELINE_ORDER:
        metric = metrics_by_baseline.get(mode_key)
        if metric is None:
            continue

        label = BASELINE_LABELS.get(mode_key, mode_key)
        label = f"**{label}**" if mode_key == "aisec_full" else label

        lines.append(
            f"| {label} "
            f"| {metric.total_events} "
            f"| {metric.total_tasks} "
            f"| {metric.precision:.3f} "
            f"| {metric.recall:.3f} "
            f"| {metric.f1_score:.3f} "
            f"| {metric.false_positive_rate:.3f} "
            f"| {metric.false_negative_rate:.3f} "
            f"| {metric.accuracy:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Per-Group Metrics",
            "",
            (
                "| Mode | Group A FPR | Group B Injection Detection | "
                "Group C Enforcement | Group D Correlation |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )

    for mode_key in BASELINE_ORDER:
        metric = metrics_by_baseline.get(mode_key)
        if metric is None:
            continue

        label = BASELINE_LABELS.get(mode_key, mode_key)
        label = f"**{label}**" if mode_key == "aisec_full" else label

        lines.append(
            f"| {label} "
            f"| {metric.group_a_fpr:.3f} "
            f"| {metric.group_b_detection_rate:.3f} "
            f"| {metric.group_c_enforcement_rate:.3f} "
            f"| {metric.group_d_correlation_rate:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Intervention Metrics",
            "",
            "| Mode | Hard Block Rate | Human Review Rate | Total Intervention Rate |",
            "|---|---:|---:|---:|",
        ]
    )

    for mode_key in BASELINE_ORDER:
        metric = metrics_by_baseline.get(mode_key)
        if metric is None:
            continue

        label = BASELINE_LABELS.get(mode_key, mode_key)
        label = f"**{label}**" if mode_key == "aisec_full" else label

        lines.append(
            f"| {label} "
            f"| {metric.hard_block_rate:.3f} "
            f"| {metric.human_review_rate:.3f} "
            f"| {metric.intervention_rate:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Latency",
            "",
            "| Mode | Mean (ms) | Median (ms) | P95 (ms) | P99 (ms) |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for mode_key in BASELINE_ORDER:
        metric = metrics_by_baseline.get(mode_key)
        if metric is None:
            continue

        label = BASELINE_LABELS.get(mode_key, mode_key)

        lines.append(
            f"| {label} "
            f"| {metric.latency_mean_ms:.3f} "
            f"| {metric.latency_median_ms:.3f} "
            f"| {metric.latency_p95_ms:.3f} "
            f"| {metric.latency_p99_ms:.3f} |"
        )

    if "aisec_full" in metrics_by_baseline:
        metric = metrics_by_baseline["aisec_full"]
        lines.extend(
            [
                "",
                "## AISec Full Pipeline Security Summary",
                "",
                f"- Safe state activations: {metric.safe_state_activation_count}",
                f"- Correlation alerts: {metric.correlation_alert_count}",
                f"- Prompt injection detections: {metric.prompt_injection_alert_count}",
                f"- Audit chain intact: {metric.audit_chain_intact}",
            ]
        )

    if redaction_warnings:
        lines.extend(
            [
                "",
                "## Redaction Warnings",
                "",
                (
                    "The exporter detected sensitive-looking field names. "
                    "Review these before public release."
                ),
                "",
            ]
        )
        for warning in redaction_warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "---",
            "",
            f"*Generated by AISec Deployment Study Framework, {year}.*",
        ]
    )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# Manifest and checksums
# =============================================================================


def build_manifest(
    *,
    output_dir: Path,
    exported_files: list[Path],
    generated_at: str,
    events: list[StudyEvent],
    metrics_by_baseline: dict[str, StudyMetrics],
    task_summary: dict[str, Any],
    comparison: dict[str, Any],
    study_metadata: dict[str, Any],
    redaction_warnings: list[str],
    audit_log_copied: bool,
    event_count_jsonl: int,
    event_count_csv: int,
) -> dict[str, Any]:
    """
    Build a reproducibility manifest for exported artifacts.
    """
    files = {}

    for path in exported_files:
        if not path.exists():
            continue

        relative_name = str(path.relative_to(output_dir))
        files[relative_name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    baseline_modes = sorted(metrics_by_baseline.keys())

    return {
        "generated_at": generated_at,
        "exporter_version": EXPORTER_VERSION,
        "study_metadata": make_json_safe(study_metadata),
        "dataset": {
            "total_events": len(events),
            "jsonl_events_written": event_count_jsonl,
            "csv_events_written": event_count_csv,
            "total_tasks": task_summary.get("total", 0),
            "total_actions": task_summary.get("total_actions", 0),
            "baseline_modes": baseline_modes,
            "audit_log_copied": audit_log_copied,
        },
        "task_summary": make_json_safe(task_summary),
        "metrics_available": {
            name: metric_to_dict(metric)
            for name, metric in sorted(metrics_by_baseline.items())
        },
        "comparison_present": bool(comparison),
        "redaction": {
            "warning_count": len(redaction_warnings),
            "warnings": list(redaction_warnings),
        },
        "files": files,
    }


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum for a file."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# =============================================================================
# Redaction and safety
# =============================================================================


def scan_events_for_sensitive_content(events: list[StudyEvent]) -> list[str]:
    """
    Scan exported event records for obvious sensitive field names.

    This is a defensive final check. It does not replace proper upstream
    redaction, but it reduces the chance of accidental public leakage.
    """
    warnings: list[str] = []

    for index, event in enumerate(events):
        record = event_to_json_dict(event)
        _scan_mapping_for_sensitive_keys(
            record,
            path=f"events[{index}]",
            warnings=warnings,
        )

    return warnings


def _scan_mapping_for_sensitive_keys(
    value: Any,
    *,
    path: str,
    warnings: list[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.lower()

            if any(keyword in key_lower for keyword in SENSITIVE_KEYWORDS):
                warnings.append(f"{path}.{key_text}")

            _scan_mapping_for_sensitive_keys(
                child,
                path=f"{path}.{key_text}",
                warnings=warnings,
            )
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            _scan_mapping_for_sensitive_keys(
                child,
                path=f"{path}[{index}]",
                warnings=warnings,
            )


# =============================================================================
# Conversion helpers
# =============================================================================


def event_to_json_dict(event: StudyEvent) -> dict[str, Any]:
    """Convert StudyEvent to JSON-friendly dictionary."""
    if hasattr(event, "to_json_dict"):
        return make_json_safe(event.to_json_dict())

    if hasattr(event, "to_dict"):
        return make_json_safe(event.to_dict())

    if is_dataclass(event):
        return make_json_safe(asdict(event))

    raise TypeError(f"Unsupported event type: {type(event)!r}")


def event_to_flat_dict(event: StudyEvent) -> dict[str, Any]:
    """Convert StudyEvent to flat CSV-friendly dictionary."""
    if hasattr(event, "to_flat_dict"):
        return make_json_safe(event.to_flat_dict())

    record = event_to_json_dict(event)
    flat: dict[str, Any] = {}

    for key, value in record.items():
        if isinstance(value, list):
            flat[key] = ";".join(str(item) for item in value)
        elif isinstance(value, dict):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value

    return flat


def metric_to_dict(metric: StudyMetrics) -> dict[str, Any]:
    """Convert StudyMetrics to JSON-friendly dictionary."""
    if hasattr(metric, "to_dict"):
        return make_json_safe(metric.to_dict())

    if is_dataclass(metric):
        return make_json_safe(asdict(metric))

    raise TypeError(f"Unsupported metric type: {type(metric)!r}")


def make_json_safe(value: Any) -> Any:
    """
    Convert common Python objects into JSON-serializable structures.
    """
    if isinstance(value, dict):
        return {str(key): make_json_safe(child) for key, child in value.items()}

    if isinstance(value, list):
        return [make_json_safe(child) for child in value]

    if isinstance(value, tuple):
        return [make_json_safe(child) for child in value]

    if isinstance(value, set):
        return sorted(make_json_safe(child) for child in value)

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return make_json_safe(asdict(value))

    if hasattr(value, "value"):
        return make_json_safe(value.value)

    return value


def collect_fieldnames(records: list[dict[str, Any]]) -> list[str]:
    """
    Collect stable CSV fieldnames from all records.
    """
    if not records:
        return [
            "event_id",
            "study_run_id",
            "task_run_id",
            "task_id",
            "group",
            "ground_truth",
            "threat_label",
            "baseline_mode",
            "agent_id",
            "framework",
            "model_provider",
            "model_name",
            "action_type",
            "target",
            "payload_summary",
            "decision",
            "risk_score",
            "rule_hits",
            "was_blocked",
            "was_intercepted",
            "was_reviewed",
            "injection_detected",
            "injection_confidence",
            "correlation_alerts",
            "temporal_alerts",
            "safe_state_active",
            "latency_ms",
            "audit_entry_id",
            "schema_version",
            "aisec_version",
            "git_commit",
            "seed",
            "framework_version",
            "timestamp",
        ]

    preferred = collect_fieldnames([])
    seen = set(preferred)
    extra: list[str] = []

    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                extra.append(key)

    return preferred + sorted(extra)


def write_json(output: Path, data: dict[str, Any]) -> None:
    """Write pretty JSON with stable ordering."""
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as fh:
        json.dump(make_json_safe(data), fh, indent=2, sort_keys=True)
        fh.write("\n")


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
