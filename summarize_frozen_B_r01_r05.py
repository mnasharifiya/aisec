import json
from pathlib import Path

base = Path("experiments/deployment_study/results/official_real_agent")
expected = [
    "official_B_injection_access_policy_001",
    "official_B_injection_document_001",
    "official_B_injection_document_002",
    "official_B_injection_email_001",
    "official_B_injection_market_001",
    "official_B_injection_market_002",
    "official_B_injection_status_001",
    "official_B_injection_web_001",
    "official_B_injection_web_002",
    "official_B_injection_web_003",
]
rounds = ["r01", "r02", "r03", "r04", "r05"]

attempts = []
valid_by_task_round = {}
tp = tn = fp = fn = 0
events_used = 0
outcome_counts = {}

for r in rounds:
    pattern = f"aisec-v1.7-frozen-real-agent-eval-001-B-{r}-*"

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

        for tr in d.get("task_runs", []):
            task_id = tr.get("task_id")
            outcome = tr.get("outcome")
            event_count = tr.get("event_count", 0)

            attempts.append(
                {
                    "round": r,
                    "source_run": run_dir.name,
                    "task_id": task_id,
                    "outcome": outcome,
                    "event_count": event_count,
                }
            )

            if task_id in expected and event_count > 0:
                valid_by_task_round[(task_id, r)] = {
                    "outcome": outcome,
                    "event_count": event_count,
                    "source_run": run_dir.name,
                }

missing = []
for task_id in expected:
    for r in rounds:
        if (task_id, r) not in valid_by_task_round:
            missing.append({"task_id": task_id, "round": r})

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0
fnr = fn / (fn + tp) if (fn + tp) else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

summary = {
    "summary_id": "aisec-v1.7-frozen-real-agent-eval-001-B-r01-r05-combined",
    "manifest_status": "frozen",
    "group": "B",
    "rounds": rounds,
    "expected_task_count": len(expected),
    "expected_valid_task_rounds": len(expected) * len(rounds),
    "valid_completed_task_rounds": len(valid_by_task_round),
    "missing_valid_task_rounds": missing,
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

out = base / "aisec-v1.7-frozen-real-agent-eval-001-B-r01-r05_combined_summary.json"
json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)

print(
    json.dumps(
        {
            "summary_id": summary["summary_id"],
            "expected_valid_task_rounds": summary["expected_valid_task_rounds"],
            "valid_completed_task_rounds": summary["valid_completed_task_rounds"],
            "attempt_count": summary["attempt_count"],
            "events_used_for_metrics": summary["events_used_for_metrics"],
            "confusion_matrix": summary["confusion_matrix"],
            "metrics_valid_events_only": summary["metrics_valid_events_only"],
            "operational_outcome_counts": summary["operational_outcome_counts"],
            "missing_valid_task_rounds": summary["missing_valid_task_rounds"],
        },
        indent=2,
    )
)

print()
print("wrote:", out)
