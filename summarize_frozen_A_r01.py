import json
from pathlib import Path

base = Path("experiments/deployment_study/results/official_real_agent")
pattern = "aisec-v1.7-frozen-real-agent-eval-001-A-r01-chunk-*"

expected = [
    "official_A_safe_trade_001",
    "official_A_safe_trade_002",
    "official_A_safe_trade_003",
    "official_A_safe_market_search_001",
    "official_A_safe_market_search_002",
    "official_A_safe_status_001",
    "official_A_safe_document_001",
    "official_A_safe_config_001",
    "official_A_safe_email_001",
    "official_A_safe_trade_004",
]

attempts = []
final_by_task = {}
tp = tn = fp = fn = 0
events_used = 0
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
        outcome = r.get("outcome")
        event_count = r.get("event_count", 0)

        attempts.append(
            {
                "source_run": run_dir.name,
                "task_id": task_id,
                "outcome": outcome,
                "event_count": event_count,
            }
        )

        if task_id in expected and event_count > 0:
            final_by_task[task_id] = {
                "outcome": outcome,
                "event_count": event_count,
                "source_run": run_dir.name,
            }

valid_tasks = sorted(final_by_task.keys())
missing_valid_tasks = [t for t in expected if t not in final_by_task]

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0
fnr = fn / (fn + tp) if (fn + tp) else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

summary = {
    "summary_id": "aisec-v1.7-frozen-real-agent-eval-001-A-r01-combined",
    "manifest_status": "frozen",
    "group": "A",
    "repetition_round": "r01",
    "expected_task_count": len(expected),
    "valid_completed_task_count": len(valid_tasks),
    "missing_valid_task_count": len(missing_valid_tasks),
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
    "valid_tasks": valid_tasks,
    "missing_valid_tasks": missing_valid_tasks,
    "attempts": attempts,
}

out = base / "aisec-v1.7-frozen-real-agent-eval-001-A-r01_combined_summary.json"
json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)

print(
    json.dumps(
        {
            "summary_id": summary["summary_id"],
            "valid_completed_task_count": summary["valid_completed_task_count"],
            "expected_task_count": summary["expected_task_count"],
            "attempt_count": summary["attempt_count"],
            "events_used_for_metrics": summary["events_used_for_metrics"],
            "confusion_matrix": summary["confusion_matrix"],
            "metrics_valid_events_only": summary["metrics_valid_events_only"],
            "operational_outcome_counts": summary["operational_outcome_counts"],
            "missing_valid_tasks": summary["missing_valid_tasks"],
        },
        indent=2,
    )
)

print()
print("wrote:", out)
