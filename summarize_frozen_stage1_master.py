import json
from pathlib import Path

base = Path("experiments/deployment_study/results/official_real_agent")

security_path = (
    base
    / "aisec-v1.7-frozen-real-agent-eval-001-security-groups-A-B-C-D-r01-r05_combined_summary.json"
)
control_path = (
    base / "aisec-v1.7-frozen-real-agent-eval-001-N-r01-r05_control_summary.json"
)

if not security_path.exists():
    raise FileNotFoundError(f"Missing security summary: {security_path}")

if not control_path.exists():
    raise FileNotFoundError(f"Missing N/control summary: {control_path}")

security = json.load(open(security_path, encoding="utf-8"))
control = json.load(open(control_path, encoding="utf-8"))

summary = {
    "summary_id": "aisec-v1.7-frozen-real-agent-eval-001-stage1-r01-r05-master-summary",
    "manifest_status": "frozen",
    "evaluation_stage": "Stage 1",
    "repetitions": ["r01", "r02", "r03", "r04", "r05"],
    "benchmark_task_rounds": {
        "security_groups_A_B_C_D_expected": security["expected_valid_task_rounds"],
        "security_groups_A_B_C_D_completed": security["valid_completed_task_rounds"],
        "control_group_N_expected": control["expected_control_task_rounds"],
        "control_group_N_accepted": control["accepted_control_task_rounds"],
        "total_expected": security["expected_valid_task_rounds"]
        + control["expected_control_task_rounds"],
        "total_completed_or_accepted": security["valid_completed_task_rounds"]
        + control["accepted_control_task_rounds"],
    },
    "security_event_metrics_scope": "A+B+C+D only. N/control is excluded from precision, recall, F1, FPR, and accuracy because it produced no AISec-classified security events.",
    "security_events_used_for_metrics": security["events_used_for_metrics"],
    "security_confusion_matrix": security["confusion_matrix"],
    "security_metrics_valid_events_only": security["metrics_valid_events_only"],
    "control_group_N": {
        "events_used_for_metrics": control["events_used_for_metrics"],
        "operational_outcome_counts": control["operational_outcome_counts"],
        "missing_control_task_rounds": control["missing_control_task_rounds"],
        "unexpected_control_outcomes_count": len(
            control["unexpected_control_outcomes"]
        ),
    },
    "operational_exclusions_note": "RATE_LIMIT, NO_TOOL_CALL, and TOOL_SCHEMA_MISMATCH outcomes are reported as operational/control outcomes and are not counted as AISec classification failures.",
    "security_group_summaries": security["group_summaries"],
    "control_summary": control,
}

out = base / "aisec-v1.7-frozen-real-agent-eval-001-stage1-r01-r05_master_summary.json"
json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)

print(
    json.dumps(
        {
            "summary_id": summary["summary_id"],
            "benchmark_task_rounds": summary["benchmark_task_rounds"],
            "security_events_used_for_metrics": summary[
                "security_events_used_for_metrics"
            ],
            "security_confusion_matrix": summary["security_confusion_matrix"],
            "security_metrics_valid_events_only": summary[
                "security_metrics_valid_events_only"
            ],
            "control_group_N": summary["control_group_N"],
        },
        indent=2,
    )
)

print()
print("wrote:", out)
