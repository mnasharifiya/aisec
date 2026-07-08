import json
from pathlib import Path

base = Path("experiments/deployment_study/results/official_real_agent")
manifest_path = Path("experiments/deployment_study/real_agent_tasks_v1_frozen.json")

manifest = json.load(open(manifest_path, encoding="utf-8"))
tasks = manifest.get("tasks", [])

expected_by_group = {}
for t in tasks:
    group = t.get("group") or t.get("task_group") or t.get("category")
    task_id = t.get("task_id") or t.get("id")
    expected_by_group.setdefault(group, []).append(task_id)

groups = ["A", "B", "C", "D", "N"]

summary = {
    "summary_id": "aisec-v1.7-frozen-real-agent-eval-001-r01-full-pass",
    "manifest_status": "frozen",
    "manifest_path": str(manifest_path),
    "repetition_round": "r01",
    "expected_task_counts": {g: len(expected_by_group.get(g, [])) for g in groups},
    "groups": {},
    "overall": {
        "attempt_count": 0,
        "events_used_for_metrics": 0,
        "confusion_matrix": {
            "true_positives": 0,
            "true_negatives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        },
        "operational_outcome_counts": {},
    },
}

for group in groups:
    pattern = f"aisec-v1.7-frozen-real-agent-eval-001-{group}-r01-*"
    expected = expected_by_group.get(group, [])

    attempts = []
    seen_task_ids = set()
    events_used = 0
    tp = tn = fp = fn = 0
    outcome_counts = {}

    for run_dir in sorted(base.glob(pattern)):
        analysis_path = run_dir / "official_analysis.json"
        if not analysis_path.exists():
            continue

        d = json.load(open(analysis_path, encoding="utf-8"))
        cm = d.get("confusion_matrix", {})

        tp += cm.get("true_positives", 0)
        tn += cm.get("true_negatives", 0)
        fp += cm.get("false_positives", 0)
        fn += cm.get("false_negatives", 0)

        events_used += d.get("event_sources", {}).get("events_used_for_metrics", 0)

        for k, v in d.get("operational", {}).get("outcome_counts", {}).items():
            outcome_counts[k] = outcome_counts.get(k, 0) + v

        for r in d.get("task_runs", []):
            task_id = r.get("task_id")
            seen_task_ids.add(task_id)
            attempts.append(
                {
                    "source_run": run_dir.name,
                    "task_id": task_id,
                    "outcome": r.get("outcome"),
                    "event_count": r.get("event_count", 0),
                }
            )

    missing_seen_tasks = [t for t in expected if t not in seen_task_ids]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    summary["groups"][group] = {
        "expected_task_count": len(expected),
        "seen_task_count": len(seen_task_ids),
        "missing_seen_tasks": missing_seen_tasks,
        "attempt_count": len(attempts),
        "events_used_for_metrics": events_used,
        "confusion_matrix": {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "metrics_valid_events_only": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "accuracy": accuracy,
        },
        "operational_outcome_counts": outcome_counts,
        "attempts": attempts,
    }

    summary["overall"]["attempt_count"] += len(attempts)
    summary["overall"]["events_used_for_metrics"] += events_used
    summary["overall"]["confusion_matrix"]["true_positives"] += tp
    summary["overall"]["confusion_matrix"]["true_negatives"] += tn
    summary["overall"]["confusion_matrix"]["false_positives"] += fp
    summary["overall"]["confusion_matrix"]["false_negatives"] += fn

    for k, v in outcome_counts.items():
        summary["overall"]["operational_outcome_counts"][k] = (
            summary["overall"]["operational_outcome_counts"].get(k, 0) + v
        )

cm = summary["overall"]["confusion_matrix"]
tp = cm["true_positives"]
tn = cm["true_negatives"]
fp = cm["false_positives"]
fn = cm["false_negatives"]

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0
fnr = fn / (fn + tp) if (fn + tp) else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

summary["overall"]["metrics_valid_events_only"] = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "false_positive_rate": fpr,
    "false_negative_rate": fnr,
    "accuracy": accuracy,
}

summary["interpretation_notes"] = [
    "A-D groups completed one frozen r01 pass.",
    "N group produced no AISec-classified events; NO_TOOL_CALL and TOOL_SCHEMA_MISMATCH are reported operationally.",
    "TOOL_SCHEMA_MISMATCH for official_N_malformed_trade_001 was reproduced on retry and should not be retried further in r01.",
    "Provider RATE_LIMIT outcomes are operational exclusions and are not counted as AISec classification failures.",
    "Group D context-sensitive tasks produced the observed false negatives and should be discussed in failure analysis without post-hoc tuning.",
]

out = base / "aisec-v1.7-frozen-real-agent-eval-001-r01_full_pass_summary.json"
json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)

print(
    json.dumps(
        {
            "summary_id": summary["summary_id"],
            "expected_task_counts": summary["expected_task_counts"],
            "overall_attempt_count": summary["overall"]["attempt_count"],
            "overall_events_used_for_metrics": summary["overall"][
                "events_used_for_metrics"
            ],
            "overall_confusion_matrix": summary["overall"]["confusion_matrix"],
            "overall_metrics_valid_events_only": summary["overall"][
                "metrics_valid_events_only"
            ],
            "overall_operational_outcome_counts": summary["overall"][
                "operational_outcome_counts"
            ],
            "group_seen_counts": {
                g: {
                    "seen_task_count": summary["groups"][g]["seen_task_count"],
                    "expected_task_count": summary["groups"][g]["expected_task_count"],
                    "events_used_for_metrics": summary["groups"][g][
                        "events_used_for_metrics"
                    ],
                    "confusion_matrix": summary["groups"][g]["confusion_matrix"],
                    "operational": summary["groups"][g]["operational_outcome_counts"],
                    "missing_seen_tasks": summary["groups"][g]["missing_seen_tasks"],
                }
                for g in groups
            },
        },
        indent=2,
    )
)

print()
print("wrote:", out)
