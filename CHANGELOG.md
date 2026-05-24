# Changelog

All notable changes to AISec are documented here.

## [1.0.0] — 2025-05-06

### Added
- Core analysis engine — full 5-layer pipeline
- Feature vector builder — 8-dimensional encoding
- Risk scorer — R(x) = sigmoid(wᵀx + b)
- Rule engine — Scenario A (Trading AI) and Scenario B (Urban AI)
- Decision engine — ALLOW / BLOCK / ESCALATE / PENDING_REVIEW
- Hash-chain audit logger — SHA-256, tamper-evident, append-only
- Live monitor CLI — real-time event streaming
- SOC console — interactive analyst environment
- Statistics dashboard — comprehensive security metrics
- Audit log CLI — inspect, verify, and export audit entries
- Simulated trading agent — Scenario A test environment
- Simulated urban agent — Scenario B test environment
- 130+ passing tests across unit, integration, and simulation
- Apache 2.0 open source license