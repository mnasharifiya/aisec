import json
from pathlib import Path

base = Path("experiments/deployment_study/results/official_real_agent")

summary_files = {
    "A": base / "aisec-v1.7-frozen-real-agent-eval-001-A-r01-r05_combined_summary.json",
    "B": base / "aisec-v1.7-frozen-real-agent-eval-001-B-r01-r05_combined_summary.json",
    "C": base / "aisec-v1.7-frozen-real-agent-eval-001-C-r01-r05_progress_summary.json",
    "D": base / "aisec-v1.7-frozen-real-agent-eval-001-D-r01-r05_combined_summary.json",
}

groups = {}
tp = tn = fp = fn = 0
events_used = 0
expected_valid_task_rounds = 0
valid_completed_task_rounds = 0
operational = {}

for group, path in summary_files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing summary for Group {group}: {path}")

    d = json.load(open(path, encoding="utf-8"))
    groups[group] = d

    cm = d["confusion_matrix"]
    tp += cm.get("true_positives", 0)
    tn += cm.get("true_negatives", 0)
    fp += cm.get("false_positives", 0)
    fn += cm.get("false_negatives", 0)

    events_used += d.get("events_used_for_metrics", 0)
    expected_valid_task_rounds += d.get("expected_valid_task_rounds", 0)
    valid_completed_task_rounds += d.get("valid_completed_task_rounds", 0)

    for k, v in d.get("operational_outcome_counts", {}).items():
        operational[k] = operational.get(k, 0) + v

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0
fnr = fn / (fn + tp) if (fn + tp) else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

summary = {
    "summary_id": "aisec-v1.7-frozen-real-agent-eval-001-security-groups-A-B-C-D-r01-r05-combined",
    "manifest_status": "frozen",
    "included_groups": ["A", "B", "C", "D"],
    "note": "N/control tasks are reported separately because they are operational/no-tool controls and do not produce AISec-classified security events.",
    "expected_valid_task_rounds": expected_valid_task_rounds,
    "valid_completed_task_rounds": valid_completed_task_rounds,
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
    "operational_outcome_counts": operational,
    "group_summaries": {
        g: {
            "expected_valid_task_rounds": groups[g].get("expected_valid_task_rounds"),
            "valid_completed_task_rounds": groups[g].get("valid_completed_task_rounds"),
            "events_used_for_metrics": groups[g].get("events_used_for_metrics"),
            "confusion_matrix": groups[g].get("confusion_matrix"),
            "metrics_valid_events_only": groups[g].get("metrics_valid_events_only"),
            "operational_outcome_counts": groups[g].get("operational_outcome_counts"),
            "missing_valid_task_rounds": groups[g].get("missing_valid_task_rounds", []),
        }
        for g in ["A", "B", "C", "D"]
    },
}

out = (
    base
    / "aisec-v1.7-frozen-real-agent-eval-001-security-groups-A-B-C-D-r01-r05_combined_summary.json"
)
json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)

print(
    json.dumps(
        {
            "summary_id": summary["summary_id"],
            "expected_valid_task_rounds": summary["expected_valid_task_rounds"],
            "valid_completed_task_rounds": summary["valid_completed_task_rounds"],
            "events_used_for_metrics": summary["events_used_for_metrics"],
            "confusion_matrix": summary["confusion_matrix"],
            "metrics_valid_events_only": summary["metrics_valid_events_only"],
            "operational_outcome_counts": summary["operational_outcome_counts"],
        },
        indent=2,
    )
)

print()
print("wrote:", out)
