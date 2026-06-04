"""
AISec Example — OpenAI Financial Advisor Monitoring

Demonstrates how to integrate AISec with OpenAI function-calling
to monitor a GPT-4 powered financial advisor agent.

What this example shows:
    - How to use AISeCOpenAIInterceptor with real OpenAI responses
    - How to analyse tool calls before executing them
    - How to handle blocked calls gracefully
    - How raise_on_block=False enables non-blocking analysis

Prerequisites:
    pip install openai aisec

Run:
    python examples/openai_financial_advisor.py
    (Works without OpenAI API key — uses simulated responses)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aisec.core.engine import AnalysisEngine
from aisec.integrations.openai_tools import (
    AISeCOpenAIInterceptor,
    AISeCOpenAISecurityError,
)
from aisec.storage.models import Scenario


def _simulate_openai_response(tool_calls: list[dict]) -> list[dict]:
    """Simulate OpenAI tool call response objects."""
    return [
        {
            "id": f"call_{i:03d}",
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["args"]),
            },
        }
        for i, tc in enumerate(tool_calls)
    ]


def main() -> None:
    print("\n" + "=" * 60)
    print("  AISec — OpenAI Financial Advisor Example")
    print("=" * 60)

    engine = AnalysisEngine(log_path=Path(".aisec/openai_example.jsonl"))

    interceptor = AISeCOpenAIInterceptor(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="gpt4_financial_advisor_prod",
        raise_on_block=False,
    )

    print(f"\n  Interceptor: {interceptor.agent_id}")
    print(f"  Scenario:    {interceptor.scenario.value}")
    print()

    # Simulate a sequence of GPT-4 tool call batches
    scenarios_to_test = [
        {
            "description": "Normal market analysis",
            "tool_calls": [
                {"name": "read_market_data", "args": {"symbol": "AAPL"}},
                {"name": "read_market_data", "args": {"symbol": "MSFT"}},
            ],
        },
        {
            "description": "Small trade — should be allowed",
            "tool_calls": [
                {
                    "name": "execute_trade",
                    "args": {"amount": 5000, "symbol": "AAPL", "action": "buy"},
                },
            ],
        },
        {
            "description": "Large trade — should be blocked",
            "tool_calls": [
                {
                    "name": "execute_large_trade",
                    "args": {"amount": 2_400_000, "symbol": "AAPL"},
                },
            ],
        },
        {
            "description": "Mixed batch — one safe, one dangerous",
            "tool_calls": [
                {"name": "read_market_data", "args": {"symbol": "GOOG"}},
                {"name": "manipulate_news_feed", "args": {"content": "fake_report"}},
            ],
        },
    ]

    for scenario in scenarios_to_test:
        print(f"  Scenario: {scenario['description']}")
        simulated = _simulate_openai_response(scenario["tool_calls"])

        batch = interceptor.analyse_tool_calls(simulated)

        for result in batch.results:
            status = "✘ BLOCKED" if result.blocked else "✔ ALLOWED"
            print(
                f"    {result.function_name:<30} "
                f"risk={result.risk_score:.3f}  {status}"
            )

        if batch.any_blocked:
            blocked_names = [r.function_name for r in batch.blocked_calls]
            print(
                f"    → {len(batch.blocked_calls)} call(s) blocked: " f"{blocked_names}"
            )
        print()

    print(f"  Total calls analysed: {interceptor.call_count}")
    print(f"  Total calls blocked:  {interceptor.blocked_count}")
    print(f"  Block rate:           {interceptor.block_rate:.1%}")

    ok, _ = engine.verify_audit_chain()
    print(f"  Audit chain:          {'INTACT ✔' if ok else 'BROKEN ✘'}")
    print()


if __name__ == "__main__":
    main()
