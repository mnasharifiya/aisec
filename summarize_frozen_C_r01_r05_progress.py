import json
from pathlib import Path

base = Path("experiments/deployment_study/results/official_real_agent")

tasks = [
    "official_C_risky_trade_001",
    "official_C_risky_trade_002",
    "official_C_risky_trade_003",
    "official_C_market_manipulation_001",
    "official_C_market_manipulation_002",
    "official_C_market_manipulation_003",
    "official_C_risk_override_001",
    "official_C_risk_override_002",
    "official_C_urban_curfew_001",
    "official_C_urban_curfew_002",
    "official_C_urban_lockdown_001",
    "official_C_urban_lockdown_002",
    "official_C_urban_power_001",
    "official_C_urban_power_002",
    "official_C_urban_traffic_001",
]

rounds = ["r01", "r02", "r03", "r04", "r05"]

valid = {}
attempts = []
tp = tn = fp = fn = 0
events_used = 0
outcomes = {}

for r in rounds:
    pattern = f"aisec-v1.7-frozen-real-agent-eval-001-C-{r}-*"
    for run_dir in sorted(base.glob(pattern)):
        p = run_dir / "official_analysis.json"
        if not p.exists():
            continue

        d = json.load(open(p, encoding="utf-8"))
        cm = d.get("confusion_matrix", {})

        tp += cm.get("true_positives", 0)
        tn += cm.get("true_negatives", 0)
        fp += cm.get("false_positives", 0)
        fn += cm.get("false_negatives", 0)
        events_used += d.get("event_sources", {}).get("events_used_for_metrics", 0)

        for k, v in d.get("operational", {}).get("outcome_counts", {}).items():
            outcomes[k] = outcomes.get(k, 0) + v

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

            if task_id in tasks and event_count > 0:
                valid[(task_id, r)] = {
                    "outcome": outcome,
                    "event_count": event_count,
                    "source_run": run_dir.name,
                }

missing = []
for r in rounds:
    for task_id in tasks:
        if (task_id, r) not in valid:
            missing.append({"round": r, "task_id": task_id})

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0
fnr = fn / (fn + tp) if (fn + tp) else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

summary = {
    "summary_id": "aisec-v1.7-frozen-real-agent-eval-001-C-r01-r05-progress",
    "manifest_status": "frozen",
    "group": "C",
    "rounds": rounds,
    "expected_valid_task_rounds": len(tasks) * len(rounds),
    "valid_completed_task_rounds": len(valid),
    "remaining_valid_task_rounds": len(missing),
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
    "operational_outcome_counts": outcomes,
    "missing_valid_task_rounds": missing,
    "attempts": attempts,
}

out = base / "aisec-v1.7-frozen-real-agent-eval-001-C-r01-r05_progress_summary.json"
json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)

print(
    json.dumps(
        {
            "summary_id": summary["summary_id"],
            "expected_valid_task_rounds": summary["expected_valid_task_rounds"],
            "valid_completed_task_rounds": summary["valid_completed_task_rounds"],
            "remaining_valid_task_rounds": summary["remaining_valid_task_rounds"],
            "events_used_for_metrics": summary["events_used_for_metrics"],
            "confusion_matrix": summary["confusion_matrix"],
            "metrics_valid_events_only": summary["metrics_valid_events_only"],
            "operational_outcome_counts": summary["operational_outcome_counts"],
        },
        indent=2,
    )
)

print()
print("FIRST 20 MISSING:")
for m in missing[:20]:
    print(m["round"], m["task_id"])

print()
print("wrote:", out)
