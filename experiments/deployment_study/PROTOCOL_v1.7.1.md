# AISec Deployment Study Protocol Amendment — v1.7.1

**Amendment status:** Active clarification amendment  
**Parent protocol:** PROTOCOL_v1.7.md  
**Parent protocol version:** 1.7.0  
**Amendment version:** 1.7.1  
**Date:** 2026-07-04  
**Project:** AISec — Runtime Security Monitoring and Policy Enforcement for Autonomous AI Agents  
**Author:** Muhammad Muttaka  

## 1. Purpose of This Amendment

This amendment clarifies frozen-manifest identity, operational exclusion handling, candidate-run handling, and intermediate run targets for the AISec v1.7 real-agent evaluation.

This amendment does not change task labels, task prompts, baseline definitions, metric formulas, model selection, AISec thresholds, or success criteria.

It is a methodological clarification intended to make the frozen evaluation more auditable and reviewer-resistant.

## 2. Frozen Task Manifest Identity

The official frozen task manifest for AISec v1.7 is:

`experiments/deployment_study/tasks_frozen_v1_7.yaml`

SHA256:

`53B58E3239252DAE0F55E9B66FF634FA33185576519561B27F12E008B7022EFC`

This file must not be edited after freeze.

Any change to task content, task prompts, labels, expected outcomes, scenarios, group assignments, or metadata requires a new frozen manifest version and a new amendment.

## 3. Pilot, Intermediate, Main, and Full Evaluation Targets

The parent protocol defines:

- Pilot evaluation: 50 tasks × 2 repetitions = 100 real-agent runs
- Main evaluation: 50 tasks × 10 repetitions = 500 real-agent runs
- Preferred full two-model evaluation: 50 tasks × 10 repetitions × 2 models = 1,000 real-agent runs

This amendment adds an intermediate validation milestone:

- Intermediate single-model validation: 50 tasks × 5 valid repetitions = 250 valid real-agent runs

The 250-run milestone is not a replacement for the 500-run main evaluation. It is used as a clean intermediate result for early analysis, debugging, and supervisor review.

Final Q1-level claims should prioritize the 500-run or 1,000-run evaluation if completed.

## 4. Candidate and Frozen Evidence Separation

Runs marked with:

`manifest_status: candidate_not_frozen`

are treated as candidate-development or pilot evidence.

These runs may be used to discuss:

- pipeline development
- runner validation
- provider behavior
- rate-limit behavior
- debugging
- preliminary detection trends

They must not be presented as the sole final frozen performance evidence.

Final frozen performance claims must be based on runs that use the frozen task manifest and record the frozen protocol/amendment identity.

## 5. Operational Exclusion Categories

The following outcomes are classified as operational exclusions when they prevent AISec from producing a valid security classification event:

- `RATE_LIMIT`
- `PROVIDER_ERROR`
- `TOOL_SCHEMA_MISMATCH`
- `NO_TOOL_CALL`
- malformed provider response without a valid AISec-classified event
- local infrastructure interruption unrelated to AISec logic

Operational exclusions are not counted as true positives, true negatives, false positives, or false negatives.

They must be reported separately as deployment reliability outcomes.

## 6. Metric Inclusion Rule

A run is included in security classification metrics only if it produces at least one valid AISec-classified event.

Included security outcomes include:

- `ALLOW`
- `PENDING_REVIEW`
- `BLOCK`
- `ESCALATE`
- `SAFE_STATE`

For binary intervention analysis, `PENDING_REVIEW` must be reported clearly as human review intervention.

For hard-block analysis, `PENDING_REVIEW` may be reported separately from `BLOCK`, `ESCALATE`, and `SAFE_STATE`.

## 7. Required Reporting for Operational Reliability

The final report must include:

- total planned runs
- completed runs
- failed runs
- valid metric-bearing events
- excluded operational outcomes
- rate-limited runs
- tool-schema mismatch runs
- provider-error runs
- no-tool-call runs
- per-task valid run coverage
- per-group valid run coverage
- per-group operational exclusions

Operational exclusions must not be hidden or silently removed.

## 8. Current Candidate Runs

Existing runs with `manifest_status: candidate_not_frozen` remain valid as candidate-development evidence.

They should be preserved and may be discussed as pilot evidence, but they should be separated from final frozen evaluation results.

## 9. No Change to Core Methodology

This amendment does not change:

- task labels
- task prompts
- task group assignments
- baseline definitions
- AISec thresholds
- detection logic
- metric formulas
- model selection
- tool list
- success criteria

## 10. Claim Boundary

AISec v1.7 should be described as an enterprise-oriented research prototype for runtime monitoring and policy enforcement of autonomous AI-agent tool use.

AISec v1.7 must not be described as:

- production-certified
- universally secure
- fully enterprise-proven
- a complete solution to autonomous AI safety
- validated in real financial, medical, drone, or government systems

unless such validation is actually performed and documented.
