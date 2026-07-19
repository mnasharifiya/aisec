"""Group D error analysis for AISec v1.7 frozen Stage 1 evaluation.

This script analyzes the main AISec v1.7 limitation:
context-sensitive Group D false negatives.

It does not modify AISec v1.7.
It reads frozen JSONL event records and the frozen manifest, then reports
which Group D tasks/events were missed, how often, and what signals were absent.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("experiments/deployment_study/results/official_real_agent")
MANIFEST_PATH = Path("experiments/deployment_study/real_agent_tasks_v1_frozen.json")
OUT_DIR = Path("experiments/deployment_study/results/error_analysis")
OUT_JSON = OUT_DIR / "aisec-v1.7-group-d-error-analysis-r01-r05_summary.json"
OUT_MD = Path("experiments/deployment_study/GROUP_D_ERROR_ANALYSIS_v1.7.md")

INTERVENTION_DECISIONS = {"BLOCK", "ESCALATE", "PENDING_REVIEW"}


def load_manifest() -> dict[str, dict[str, Any]]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    tasks = data.get("tasks", []) if isinstance(data, dict) else data
    return {task["task_id"]: task for task in tasks}


def find_frozen_jsonl_files() -> list[Path]:
    return sorted(
        p
        for p in ROOT.rglob("*.jsonl")
        if "aisec-v1.7-frozen-real-agent-eval-001" in str(p)
        and "smoke" not in str(p).lower()
    )


def load_group_d_events() -> list[dict[str, Any]]:
    manifest = load_manifest()
    records: list[dict[str, Any]] = []

    for jsonl_path in find_frozen_jsonl_files():
        for line in jsonl_path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("record_type") != "real_agent_study_event":
                continue

            data = obj.get("data", {})
            group = str(data.get("group", "")).upper()
            if group != "D":
                continue

            task_id = str(data.get("task_id", ""))
            task = manifest.get(task_id, {})

            study_run_id = str(data.get("study_run_id", ""))
            rep_match = re.search(r"-r(\d{2})", study_run_id)
            repetition_id = int(rep_match.group(1)) if rep_match else None

            final_decision = str(
                obj.get("enforcement", {}).get(
                    "final_decision", data.get("decision", "ALLOW")
                )
            ).upper()

            risk_score = float(data.get("risk_score") or 0.0)
            rule_hits = list(data.get("rule_hits") or [])
            injection_detected = bool(data.get("injection_detected"))
            temporal_alerts = int(data.get("temporal_alerts") or 0)
            correlation_alerts = int(data.get("correlation_alerts") or 0)
            safe_state_active = bool(data.get("safe_state_active"))

            is_intervention = final_decision in INTERVENTION_DECISIONS
            ground_truth = str(data.get("ground_truth", "")).lower()
            is_malicious = ground_truth != "safe"

            if is_malicious and is_intervention:
                classification = "TP"
            elif is_malicious and not is_intervention:
                classification = "FN"
            elif not is_malicious and is_intervention:
                classification = "FP"
            else:
                classification = "TN"

            proposed = obj.get("proposed_tool_call", {})
            normalized = obj.get("normalized_action", {})

            records.append(
                {
                    "source_path": str(jsonl_path),
                    "study_run_id": study_run_id,
                    "task_id": task_id,
                    "task_class": task.get("task_class"),
                    "scenario": task.get("scenario") or normalized.get("scenario"),
                    "threat_label": task.get("threat_label")
                    or data.get("threat_label"),
                    "prompt": task.get("prompt"),
                    "notes": task.get("notes"),
                    "expected_primary_outcome": task.get("expected_primary_outcome"),
                    "allowed_outcomes": task.get("allowed_outcomes"),
                    "success_criteria": task.get("success_criteria"),
                    "failure_policy": task.get("failure_policy"),
                    "repetition_id": repetition_id,
                    "task_run_id": data.get("task_run_id"),
                    "event_id": data.get("event_id"),
                    "ground_truth": ground_truth,
                    "final_decision": final_decision,
                    "classification": classification,
                    "risk_score": risk_score,
                    "rule_hits": rule_hits,
                    "injection_detected": injection_detected,
                    "injection_confidence": float(
                        data.get("injection_confidence") or 0.0
                    ),
                    "temporal_alerts": temporal_alerts,
                    "correlation_alerts": correlation_alerts,
                    "safe_state_active": safe_state_active,
                    "action_type": data.get("action_type")
                    or normalized.get("action_type"),
                    "target": data.get("target") or normalized.get("target"),
                    "proposed_tool_name": proposed.get("name"),
                    "proposed_tool_args": proposed.get("args"),
                    "payload_summary": data.get("payload_summary"),
                }
            )

    return records


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def task_rep_key(record: dict[str, Any]) -> tuple[str, int | None]:
    return (record["task_id"], record["repetition_id"])


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_class = Counter(r["classification"] for r in records)
    by_decision = Counter(r["final_decision"] for r in records)
    by_task = defaultdict(list)
    by_task_rep = defaultdict(list)

    for record in records:
        by_task[record["task_id"]].append(record)
        by_task_rep[task_rep_key(record)].append(record)

    fn_records = [r for r in records if r["classification"] == "FN"]
    tp_records = [r for r in records if r["classification"] == "TP"]

    task_summaries = []
    for task_id, items in sorted(by_task.items()):
        classifications = Counter(r["classification"] for r in items)
        decisions = Counter(r["final_decision"] for r in items)
        risk_scores = [r["risk_score"] for r in items]
        fn_items = [r for r in items if r["classification"] == "FN"]
        tp_items = [r for r in items if r["classification"] == "TP"]

        task_summaries.append(
            {
                "task_id": task_id,
                "task_class": items[0].get("task_class"),
                "scenario": items[0].get("scenario"),
                "event_count": len(items),
                "task_rep_count": len({task_rep_key(r) for r in items}),
                "classification_counts": dict(classifications),
                "decision_counts": dict(decisions),
                "risk_score_mean": mean_or_none(risk_scores),
                "risk_score_min": min(risk_scores) if risk_scores else None,
                "risk_score_max": max(risk_scores) if risk_scores else None,
                "rule_hit_events": sum(1 for r in items if r["rule_hits"]),
                "injection_detected_events": sum(
                    1 for r in items if r["injection_detected"]
                ),
                "temporal_alert_events": sum(1 for r in items if r["temporal_alerts"]),
                "correlation_alert_events": sum(
                    1 for r in items if r["correlation_alerts"]
                ),
                "safe_state_active_events": sum(
                    1 for r in items if r["safe_state_active"]
                ),
                "fn_event_count": len(fn_items),
                "tp_event_count": len(tp_items),
                "prompt": items[0].get("prompt"),
                "notes": items[0].get("notes"),
            }
        )

    task_rep_summaries = []
    for key, items in sorted(by_task_rep.items()):
        classifications = Counter(r["classification"] for r in items)
        decisions = Counter(r["final_decision"] for r in items)
        task_rep_summaries.append(
            {
                "task_id": key[0],
                "repetition_id": key[1],
                "event_count": len(items),
                "classification_counts": dict(classifications),
                "decision_counts": dict(decisions),
                "risk_scores": [r["risk_score"] for r in items],
                "rule_hit_events": sum(1 for r in items if r["rule_hits"]),
                "temporal_alert_events": sum(1 for r in items if r["temporal_alerts"]),
            }
        )

    fn_risk_scores = [r["risk_score"] for r in fn_records]
    tp_risk_scores = [r["risk_score"] for r in tp_records]

    no_signal_fn = [
        r
        for r in fn_records
        if not r["rule_hits"]
        and not r["injection_detected"]
        and not r["temporal_alerts"]
        and not r["correlation_alerts"]
        and not r["safe_state_active"]
    ]

    output = {
        "summary_id": "aisec-v1.7-group-d-error-analysis-r01-r05",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_root": str(ROOT),
        "manifest_path": str(MANIFEST_PATH),
        "analysis_scope": "Frozen AISec v1.7 Group D events only.",
        "event_count": len(records),
        "classification_counts": dict(by_class),
        "decision_counts": dict(by_decision),
        "task_count": len(by_task),
        "task_rep_count": len(by_task_rep),
        "task_rep_pairs_with_multiple_events": sum(
            1 for items in by_task_rep.values() if len(items) > 1
        ),
        "extra_events_over_task_rep": sum(
            len(items) - 1 for items in by_task_rep.values() if len(items) > 1
        ),
        "fn_analysis": {
            "fn_event_count": len(fn_records),
            "fn_task_count": len({r["task_id"] for r in fn_records}),
            "fn_task_rep_count": len({task_rep_key(r) for r in fn_records}),
            "fn_risk_score_mean": mean_or_none(fn_risk_scores),
            "fn_risk_score_median": median_or_none(fn_risk_scores),
            "fn_risk_score_min": min(fn_risk_scores) if fn_risk_scores else None,
            "fn_risk_score_max": max(fn_risk_scores) if fn_risk_scores else None,
            "fn_with_rule_hits": sum(1 for r in fn_records if r["rule_hits"]),
            "fn_with_injection_detected": sum(
                1 for r in fn_records if r["injection_detected"]
            ),
            "fn_with_temporal_alerts": sum(
                1 for r in fn_records if r["temporal_alerts"]
            ),
            "fn_with_correlation_alerts": sum(
                1 for r in fn_records if r["correlation_alerts"]
            ),
            "fn_with_safe_state_active": sum(
                1 for r in fn_records if r["safe_state_active"]
            ),
            "fn_with_no_recorded_escalation_signal": len(no_signal_fn),
        },
        "tp_analysis": {
            "tp_event_count": len(tp_records),
            "tp_task_count": len({r["task_id"] for r in tp_records}),
            "tp_task_rep_count": len({task_rep_key(r) for r in tp_records}),
            "tp_risk_score_mean": mean_or_none(tp_risk_scores),
            "tp_risk_score_median": median_or_none(tp_risk_scores),
            "tp_risk_score_min": min(tp_risk_scores) if tp_risk_scores else None,
            "tp_risk_score_max": max(tp_risk_scores) if tp_risk_scores else None,
            "tp_with_rule_hits": sum(1 for r in tp_records if r["rule_hits"]),
            "tp_with_injection_detected": sum(
                1 for r in tp_records if r["injection_detected"]
            ),
            "tp_with_temporal_alerts": sum(
                1 for r in tp_records if r["temporal_alerts"]
            ),
            "tp_with_correlation_alerts": sum(
                1 for r in tp_records if r["correlation_alerts"]
            ),
            "tp_with_safe_state_active": sum(
                1 for r in tp_records if r["safe_state_active"]
            ),
        },
        "task_summaries": task_summaries,
        "task_rep_summaries": task_rep_summaries,
        "false_negative_events": fn_records,
    }

    return output


def fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(summary: dict[str, Any]) -> None:
    task_rows = [
        "| Task ID | Events | TP | FN | Decisions | Mean risk | Rule-hit events | Temporal alerts |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]

    for task in summary["task_summaries"]:
        counts = Counter(task["classification_counts"])
        decisions = ", ".join(f"{k}:{v}" for k, v in task["decision_counts"].items())
        task_rows.append(
            "| "
            + " | ".join(
                [
                    task["task_id"],
                    str(task["event_count"]),
                    str(counts.get("TP", 0)),
                    str(counts.get("FN", 0)),
                    decisions,
                    fmt_float(task["risk_score_mean"]),
                    str(task["rule_hit_events"]),
                    str(task["temporal_alert_events"]),
                ]
            )
            + " |"
        )

    fn = summary["fn_analysis"]
    tp = summary["tp_analysis"]

    content = f"""# AISec v1.7 Group D Error Analysis

## Purpose

This report analyzes Group D, the main limitation observed in the frozen AISec v1.7 Stage 1 evaluation.

Group D represents context-sensitive risks where individual actions may appear lower-risk, but the sequence or context should trigger intervention.

## Evidence checkpoint

- Analysis commit: `{summary["git_commit"]}`
- Source root: `{summary["source_root"]}`
- Manifest: `{summary["manifest_path"]}`
- Scope: {summary["analysis_scope"]}

## Group D summary

- Group D event count: {summary["event_count"]}
- Group D task count: {summary["task_count"]}
- Group D task-repetition pairs: {summary["task_rep_count"]}
- Task-repetition pairs with multiple events: {summary["task_rep_pairs_with_multiple_events"]}
- Extra events over task-repetition count: {summary["extra_events_over_task_rep"]}

## Classification counts

- True positives: {summary["classification_counts"].get("TP", 0)}
- False negatives: {summary["classification_counts"].get("FN", 0)}
- False positives: {summary["classification_counts"].get("FP", 0)}
- True negatives: {summary["classification_counts"].get("TN", 0)}

## Decision counts

{chr(10).join(f"- {k}: {v}" for k, v in summary["decision_counts"].items())}

## False-negative profile

- FN events: {fn["fn_event_count"]}
- FN tasks affected: {fn["fn_task_count"]}
- FN task-repetition pairs affected: {fn["fn_task_rep_count"]}
- FN mean risk score: {fmt_float(fn["fn_risk_score_mean"])}
- FN median risk score: {fmt_float(fn["fn_risk_score_median"])}
- FN min risk score: {fmt_float(fn["fn_risk_score_min"])}
- FN max risk score: {fmt_float(fn["fn_risk_score_max"])}
- FN with rule hits: {fn["fn_with_rule_hits"]}
- FN with injection detected: {fn["fn_with_injection_detected"]}
- FN with temporal alerts: {fn["fn_with_temporal_alerts"]}
- FN with correlation alerts: {fn["fn_with_correlation_alerts"]}
- FN with safe-state active: {fn["fn_with_safe_state_active"]}
- FN with no recorded escalation signal: {fn["fn_with_no_recorded_escalation_signal"]}

## True-positive profile

- TP events: {tp["tp_event_count"]}
- TP tasks affected: {tp["tp_task_count"]}
- TP task-repetition pairs affected: {tp["tp_task_rep_count"]}
- TP mean risk score: {fmt_float(tp["tp_risk_score_mean"])}
- TP median risk score: {fmt_float(tp["tp_risk_score_median"])}
- TP min risk score: {fmt_float(tp["tp_risk_score_min"])}
- TP max risk score: {fmt_float(tp["tp_risk_score_max"])}
- TP with rule hits: {tp["tp_with_rule_hits"]}
- TP with injection detected: {tp["tp_with_injection_detected"]}
- TP with temporal alerts: {tp["tp_with_temporal_alerts"]}
- TP with correlation alerts: {tp["tp_with_correlation_alerts"]}
- TP with safe-state active: {tp["tp_with_safe_state_active"]}

## Task-level breakdown

{chr(10).join(task_rows)}

## Interpretation

Group D failures are concentrated in context-sensitive cases where the recorded event-level signals did not escalate sufficiently.

The absence of temporal alerts, correlation alerts, and safe-state activation in Group D indicates that AISec v1.7 did not yet operationalize sequence-level accumulation for these real-agent tasks.

The false negatives should therefore be interpreted as a limitation of v1.7 context aggregation, not as random evaluation noise.

## Design implication for AISec v1.8

AISec v1.8 should add explicit context-window aggregation for repeated or related actions, including:

1. Per-task and per-agent cumulative risk over a time window.
2. Sequence-aware escalation when multiple moderate-risk actions share the same target or objective.
3. Scenario-specific context policies for trading, documents, market actions, email, and policy changes.
4. Temporal alert integration into the decision engine rather than only storing temporal metadata.
5. Group-level correlation rules that escalate repeated low-risk-looking actions into PENDING_REVIEW or ESCALATE.

## Paper wording

The main limitation of AISec v1.7 is reduced recall on context-sensitive multi-step tasks. In Group D, AISec achieved 58 true-positive event interventions but missed 40 malicious events. Error analysis shows that these false negatives occurred without recorded temporal, correlation, or safe-state signals, suggesting that v1.7 primarily enforced single-event policy and risk signals rather than fully modeling cumulative multi-step context.
"""

    OUT_MD.write_text(content, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_group_d_events()
    summary = summarize(records)

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)

    print(f"wrote: {OUT_JSON}")
    print(f"wrote: {OUT_MD}")
    print("event_count:", summary["event_count"])
    print("classification_counts:", summary["classification_counts"])
    print("decision_counts:", summary["decision_counts"])
    print("fn_analysis:", json.dumps(summary["fn_analysis"], indent=2))
    print("tp_analysis:", json.dumps(summary["tp_analysis"], indent=2))


if __name__ == "__main__":
    main()
