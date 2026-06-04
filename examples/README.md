# AISec Examples

Ready-to-run examples showing AISec integration with major AI frameworks.

## Prerequisites

```bash
pip install aisec
```

## Examples

### 1. LangChain Trading Bot
```bash
python examples/langchain_trading_bot.py
```
Demonstrates AISec monitoring a LangChain agent that controls
an autonomous financial trading system. Shows how dangerous
tool calls (market manipulation, large trades) are blocked
while safe calls (market data reads) proceed normally.

### 2. AutoGen Smart City Controller
```bash
python examples/autogen_city_controller.py
```
Demonstrates AISec monitoring a Microsoft AutoGen agent
controlling smart city infrastructure. Shows how urban-specific
rules fire for curfews, power grid shutdowns, and emergency
service interference.

### 3. OpenAI Financial Advisor
```bash
python examples/openai_financial_advisor.py
```
Demonstrates AISec intercepting OpenAI GPT-4 tool calls
for a financial advisor agent. Shows batch analysis and
the difference between raise_on_block=True and False.

### 4. Custom Scenario (HR AI)
Shows how to define a completely new security scenario
for an autonomous HR AI agent using only YAML — no Python
code required.

```bash
# Load the custom scenario
python -c "
from aisec.scenarios.loader import ScenarioLoader
from pathlib import Path
loader = ScenarioLoader()
s = loader.load(Path('examples/custom_scenario.yaml'))
print(f'Loaded: {s.display_name} with {len(s.rules)} rules')
"
```

## Building Your Own Integration

```python
from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Event, Scenario

engine = AnalysisEngine()

event = Event(
    action_type="your_action",
    agent_id="your_agent",
    target="your_target",
    scenario=Scenario.TRADING_AI,
    raw_payload={"your": "data"},
)

result = engine.analyse(event)
print(result.decision)       # ALLOW / BLOCK / ESCALATE / PENDING_REVIEW
print(result.risk_score)     # 0.0 to 1.0
print(result.analysis.explanation)
```

## Adding a Custom Scenario

1. Copy `examples/custom_scenario.yaml`
2. Modify `scenario_id`, `rules`, `weights`
3. Place in `scenarios/` directory
4. AISec loads it automatically on startup

No Python code required — pure YAML configuration.