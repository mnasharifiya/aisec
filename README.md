# AISec  Runtime Security Monitoring for Autonomous AI Agents

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-130%2B%20passing-brightgreen.svg)]()

> A CLI-first runtime security platform that monitors autonomous AI agents,
> scores their actions for risk, enforces policy decisions in real time,
> and preserves tamper-evident audit trails.

Built on the research framework published in:
**"A Layered Cybersecurity Framework for Enforcing 
Human Control Over Autonomous AI Systems "**
Muhammad Muttaka, Astana IT University, 2025.

---

## What Is AISec?

Autonomous AI agents: trading bots, smart city controllers, autonomous
drones, can cause serious harm when they act without human oversight.
AISec is a security layer that sits between an AI agent and the systems
it controls, intercepting every action, scoring it for risk, and blocking
dangerous behaviour before it executes.

Think of it as **Nmap for AI behaviour**; a professional CLI security
tool designed for security operations centres (SOC) working with
autonomous AI systems.

---

## Key Features

- **Real-time monitoring**: live terminal display of every AI action
- **Risk scoring**: mathematical model R(x) = sigmoid(wᵀx + b)
- **Rule engine**: hard policy rules for trading AI and urban AI scenarios
- **Human-in-the-loop**: SOC console where analysts approve or block actions
- **Tamper-evident audit log**: SHA-256 hash chain, any modification detected
- **Scenario-specific**: built-in rules for Scenario A (Trading AI) and Scenario B (Urban AI)
- **Open source**: Apache 2.0, community scenarios welcome

---

## Installation

```bash
pip install aisec
```

Or install from source:

```bash
git clone https://github.com/MNasharifiya/aisec.git
cd aisec
pip install -e .
```

---

## Quick Start

```bash
# Show the AISec logo and system status
aisec

# Monitor AI agents in real time
aisec monitor --scenario trading_ai --steps 30

# Enter the SOC analyst console
aisec soc --scenario both

# View audit log
aisec logs

# Verify audit chain integrity
aisec logs --verify

# View security statistics dashboard
aisec stats
```

---

## Commands

| Command | Description |
|---------|-------------|
| `aisec` | Show logo and system status |
| `aisec monitor` | Live stream of AI agent actions |
| `aisec soc` | Interactive SOC analyst console |
| `aisec stats` | Security statistics dashboard |
| `aisec logs` | Inspect the audit log |
| `aisec logs --verify` | Verify hash chain integrity |
| `aisec logs --export FILE` | Export audit log to file |

---

## Scenarios

AISec ships with two built-in threat scenarios from the paper:

### Scenario A: Autonomous Trading AI
Monitors a financial trading agent for:
- Large trades exceeding $1,000,000 (immediate block)
- News feed manipulation (immediate block)
- Risk limit overrides (escalate)
- After-hours trading (human review)

### Scenario B: Smart City Urban AI
Monitors a city infrastructure controller for:
- Curfew or movement restrictions (immediate block)
- Power grid shutdowns (escalate)
- Emergency services interference (immediate block)
- Mass traffic redirects affecting 50+ intersections (human review)

---

## Architecture

AI Agent
↓
Interceptor (SDK wrapper)
↓
Feature Vector Builder  →  [a, k, f, p, d, n, s, u]
↓
Risk Scorer             →  R(x) = sigmoid(wᵀx + b)
↓
Rule Engine             →  R1: permission, R2: HITL, R3: rollback
↓
Decision Engine         →  ALLOW / BLOCK / ESCALATE / PENDING_REVIEW
↓
SOC Queue               →  Human analyst review
↓
Hash-Chain Audit Log    →  SHA-256, tamper-evident, append-only

---

## Decision Thresholds

| Risk Score | Decision |
|------------|----------|
| >= 0.80 | BLOCK immediately |
| >= 0.60 | PENDING_REVIEW — human required |
| >= 0.30 | ALLOW with monitoring |
| < 0.30 | ALLOW |

Rules always override the scorer for BLOCK and ESCALATE decisions.

---

## Adding a New Scenario

AISec is designed for community extension. To add a new scenario:

1. Create `aisec/scenarios/your_scenario/rules.yaml`
2. Add rule functions to `aisec/core/rules.py`
3. Register them in `SCENARIO_RULES`
4. Add tests in `tests/simulation/`
5. Submit a pull request

See `docs/CONTRIBUTING.md` for full guidelines.

---

## Research Foundation

AISec implements the five-layer control framework from:

> Muhammad Muttaka (2025). *A Layered Cybersecurity Framework for Enforcing 
Human Control Over Autonomous AI Systems *. School of Cybersecurity,
> Astana IT University.

The framework formalises three enforceable rules:
- **R1** ∀a∉P: execute(a) = denied
- **R2** ∀a∈H: blocked unless h(a) = True
- **R3** anomaly_detected = True → system ∈ S

---

## Development

```bash
# Clone and install
git clone https://github.com/MNasharifiya/aisec.git
cd aisec
python -m venv venv
venv\Scripts\activate       # Windows
pip install -e .
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Format code
black aisec/ tests/
```

---

## Test Coverage

tests/unit/          — 99 unit tests
tests/integration/   — 15 integration tests
tests/simulation/    — 16 simulation tests
─────────────────────────────────────────
Total                — 130+ passing
---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## Author

**Muhammad Muttaka**
School of Cybersecurity, Astana IT University, Astana, Kazakhstan
GitHub: [@MNasharifiya](https://github.com/MNasharifiya)

---

## Roadmap

| Version | Features |
|---------|----------|
| v1.0 | Core engine, CLI, Trading AI, Urban AI, SOC console |
| v2.0 | Web dashboard, incident assignment, REST API |
| v3.0 | Phone and smartwatch alerts |
| v4.0 | ML-based anomaly detection, distributed deployment |
