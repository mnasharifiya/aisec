# AISec v1.7 Baseline Comparison Report

## Purpose

This report compares the frozen AISec v1.7 Stage 1 result against simple no-leakage task-level baselines.

## Evidence checkpoint

- Baseline-result commit: `08bf4c8a2827d0f5177501ba65ef528b871102f0`
- AISec master summary: `experiments\deployment_study\results\official_real_agent\aisec-v1.7-frozen-real-agent-eval-001-stage1-r01-r05_master_summary.json`
- Baseline summary: `experiments\deployment_study\results\baselines\aisec-v1.7-tasklevel-baselines-noleakage-r01-r05_summary.json`
- Frozen manifest SHA256: `7CA6FF80DA6957919B8209D7D5F9C63E949375047754A20FBAE3C744C364BC40`

## Important interpretation note

AISec full v1.7 is evaluated at the security-event level over Groups A+B+C+D.

The deterministic baselines in this report are evaluated at task level over the same frozen manifest and five repetitions. They are no-leakage baselines: predictors only use observable request fields such as prompt, scenario, and execute_allowed_tools. They do not use task_group, ground_truth, threat_label, expected_primary_outcome, success criteria, failure policy, or notes.

Therefore, this table should be interpreted as an initial task-level baseline comparison, not as a final event-level replay comparison.

## Results

| System | Evaluation type | Used | TP | TN | FP | FN | Precision | Recall | F1 | FPR | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AISec full v1.7 | Frozen real-agent security-event evaluation | 273 | 183 | 50 | 0 | 40 | 1.0000 | 0.8206 | 0.9015 | 0.0000 | 0.8535 |
| B0_raw_no_guard | No-leakage deterministic task-level baseline | 225 | 0 | 50 | 0 | 175 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2222 |
| B1_keyword_only_prompt_guard | No-leakage deterministic task-level baseline | 225 | 120 | 50 | 0 | 55 | 1.0000 | 0.6857 | 0.8136 | 0.0000 | 0.7556 |
| B2_conservative_domain_guard | No-leakage deterministic task-level baseline | 225 | 170 | 5 | 45 | 5 | 0.7907 | 0.9714 | 0.8718 | 0.9000 | 0.7778 |
| B3_prompt_rule_guard | No-leakage deterministic task-level baseline | 225 | 110 | 50 | 0 | 65 | 1.0000 | 0.6286 | 0.7719 | 0.0000 | 0.7111 |

## Interpretation

The raw/no-guard baseline exposes the expected unsafe behavior: it allows all risky cases and therefore has zero recall.

The keyword-only and prompt-rule baselines achieve useful recall without false positives, but they miss a substantial number of risky tasks.

The conservative domain guard has high recall, but it produces a very high false-positive rate because it over-blocks safe actions in high-risk domains.

AISec full v1.7 achieves the strongest overall balance in the frozen Stage 1 evaluation, with F1 = 0.9015 and zero observed false positives. Its main weakness remains context-sensitive Group D recall, where repeated low-risk-looking actions were not always escalated as contextual violations.

## Research conclusion

These results support the claim that AISec is not merely a keyword filter or static domain blocker. Compared with simple baselines, AISec provides a stronger balance between safety intervention and false-positive control.

## Next work

The next evaluation layer should be event-level replay baselines and ablation studies.

Recommended next studies:

1. Event-level replay baseline using raw recorded tool-call traces.
2. AISec without injection-specific handling.
3. AISec without risk scoring.
4. AISec without context or temporal accumulation.
5. AISec without SOC/PENDING_REVIEW semantics.

