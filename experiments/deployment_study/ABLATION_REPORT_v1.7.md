# AISec v1.7 Event-Level Ablation Report

## Purpose

This report summarizes event-level decision-layer ablations for the frozen AISec v1.7 Stage 1 evaluation.

The goal is to estimate how AISec performance changes when specific decision-layer mechanisms are removed or simplified.

## Evidence checkpoint

- Ablation-replay commit: `85e6eab08585da238499263db9f01baf2d9cbbcf`
- Ablation summary: `experiments\deployment_study\results\ablations\aisec-v1.7-eventlevel-decision-ablation-r01-r05_summary.json`
- Evaluation type: `event_level_decision_layer_ablation_replay`
- Security event records used: 273
- Frozen JSONL files found: 302

## Scope

This is a counterfactual replay over frozen AISec v1.7 event records. It does not rerun agents, does not modify AISec v1.7, and does not change the frozen Stage 1 result.

Only Groups A+B+C+D are included. Candidate, smoke, and control records are excluded.

## Record counts

- Group A events: 50
- Group B events: 50
- Group C events: 75
- Group D events: 98
- Safe events: 50
- Malicious events: 223
- Records with rule hits: 178
- Records with injection detected: 50
- Records with temporal alerts: 0
- Records with correlation alerts: 0
- Records with safe-state active: 0

## Results

| Variant | Events | TP | TN | FP | FN | Precision | Recall | F1 | FPR | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0_recorded_AISec_full_v1.7 | 273 | 183 | 50 | 0 | 40 | 1.0000 | 0.8206 | 0.9015 | 0.0000 | 0.8535 |
| D1_no_prompt_injection_rule | 273 | 133 | 50 | 0 | 90 | 1.0000 | 0.5964 | 0.7472 | 0.0000 | 0.6703 |
| D2_no_rule_layer_score_only | 273 | 83 | 50 | 0 | 140 | 1.0000 | 0.3722 | 0.5425 | 0.0000 | 0.4872 |
| D3_no_score_thresholds_rule_only | 273 | 178 | 50 | 0 | 45 | 1.0000 | 0.7982 | 0.8878 | 0.0000 | 0.8352 |
| D4_no_pending_review_state | 273 | 103 | 50 | 0 | 120 | 1.0000 | 0.4619 | 0.6319 | 0.0000 | 0.5604 |
| D5_block_only_enforcement | 273 | 73 | 50 | 0 | 150 | 1.0000 | 0.3274 | 0.4932 | 0.0000 | 0.4505 |
| D6_no_escalate_state | 273 | 183 | 50 | 0 | 40 | 1.0000 | 0.8206 | 0.9015 | 0.0000 | 0.8535 |

## Interpretation

The replay reproduces the frozen AISec v1.7 result exactly:

- TP = 183
- TN = 50
- FP = 0
- FN = 40
- F1 = 0.9015

This confirms that the ablation replay is grounded in the same 273 security events used by the frozen Stage 1 master summary.

Removing prompt-injection-specific rule handling causes a major performance drop, reducing F1 from 0.9015 to 0.7472. This shows that prompt-injection handling is a major contributor to AISec's intervention capability.

Using score thresholds alone performs poorly, with F1 = 0.5425. This indicates that the rule layer is essential and that risk scoring alone is not sufficient for this benchmark.

Using rules without score thresholds performs close to the full system, with F1 = 0.8878. This suggests that the rule layer carries much of the current v1.7 security signal, while risk scoring provides additional coverage.

Removing the PENDING_REVIEW state causes a large drop to F1 = 0.6319. This shows that human-review-aware decision states are central to AISec's layered enforcement model.

Using only hard BLOCK decisions performs weakest among the AISec variants, with F1 = 0.4932. This supports the argument that runtime safety for autonomous agents should not rely only on binary allow/block enforcement.

Removing ESCALATE as a distinct state does not change the confusion matrix because both ESCALATE and BLOCK are counted as security interventions in the current metric definition. This should be reported as a metric-equivalence finding, not as evidence that escalation has no operational value.

## Research conclusion

The ablation results support AISec's layered design. The strongest contributors in v1.7 are the rule layer, prompt-injection handling, and PENDING_REVIEW / human-review semantics. Risk scoring contributes additional coverage but is not sufficient alone.

The results also show that AISec's strong performance is not caused by a single monolithic detector. It comes from combining policy rules, prompt-injection handling, risk thresholds, and human-review-aware intervention states.

## Limitation

This is an event-level decision-layer replay, not a new live-agent ablation run. It estimates counterfactual decision outcomes using recorded frozen event fields such as decision, risk_score, rule_hits, and injection_detected. A future study should run live ablation variants as separate versioned systems.

## Next work

1. Create Group D error analysis.
2. Create paper-ready results tables.
3. Add confidence intervals for baseline and ablation metrics.
4. Optionally run a second-model validation.
