"""
AISec Example — LangChain Trading Bot Monitoring

Demonstrates how to integrate AISec with a LangChain agent
to monitor an autonomous trading AI in real time.

What this example shows:
    - How to wrap a LangChain agent with AISec interception
    - How dangerous tool calls are automatically blocked
    - How the audit trail is maintained
    - How to inspect results after each tool call

Prerequisites:
    pip install langchain langchain-core aisec

Run:
    python examples/langchain_trading_bot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from aisec.core.engine import AnalysisEngine
from aisec.integrations.langchain import (
    AISeCCallbackHandler,
    AISeCSecurityError,
)
from aisec.storage.models import Scenario

# ── Simulated trading tools ───────────────────────────────────────────────────
# In a real deployment these would call actual trading APIs.
# Here we simulate the tools to demonstrate AISec interception.


def read_market_data(symbol: str = "AAPL") -> str:
    """Read current market data for a symbol."""
    return f"Market data for {symbol}: price=182.50, volume=45M, trend=UP"


def execute_trade(
    amount: float = 100.0, symbol: str = "AAPL", action: str = "buy"
) -> str:
    """Execute a trade order."""
    return f"Trade executed: {action} {symbol} amount=${amount:,.0f}"


def execute_large_trade(amount: float = 2_400_000.0, symbol: str = "AAPL") -> str:
    """Execute a large block trade — should be intercepted by AISec."""
    return f"LARGE TRADE: {symbol} amount=${amount:,.0f}"


def manipulate_news_feed(content: str = "fake") -> str:
    """Manipulate market news feed — should always be blocked by AISec."""
    return f"News feed manipulated: {content}"


def override_risk_limit(new_limit: float = 999_999_999) -> str:
    """Override risk limits — should be escalated by AISec."""
    return f"Risk limit set to ${new_limit:,.0f}"


# ── Main demonstration ────────────────────────────────────────────────────────


def main() -> None:
    print("\n" + "=" * 60)
    print("  AISec — LangChain Trading Bot Example")
    print("=" * 60)

    # Initialise AISec engine
    engine = AnalysisEngine(log_path=Path(".aisec/langchain_example.jsonl"))

    # Create AISec handler — wraps every LangChain tool call
    handler = AISeCCallbackHandler(
        engine=engine,
        scenario=Scenario.TRADING_AI,
        agent_id="langchain_trading_bot_v1",
    )

    print("\n  AISec handler initialised.")
    print(f"  Agent ID: {handler.agent_id}")
    print(f"  Scenario: {handler.scenario.value}")
    print()

    # Define tool calls to demonstrate — mix of safe and dangerous
    demo_calls = [
        ("read_market_data", {"symbol": "AAPL"}, False),
        ("read_market_data", {"symbol": "MSFT"}, False),
        ("execute_trade", {"amount": 800, "symbol": "AAPL", "action": "buy"}, False),
        ("execute_large_trade", {"amount": 2_400_000}, True),
        ("manipulate_news_feed", {"content": "fake_earnings"}, True),
        ("override_risk_limit", {"new_limit": 999_999_999}, True),
        ("read_market_data", {"symbol": "GOOG"}, False),
    ]

    tools = {
        "read_market_data": read_market_data,
        "execute_trade": execute_trade,
        "execute_large_trade": execute_large_trade,
        "manipulate_news_feed": manipulate_news_feed,
        "override_risk_limit": override_risk_limit,
    }

    print(f"  {'Tool':<28} {'Expected':<12} {'Result'}")
    print(f"  {'─'*28} {'─'*12} {'─'*20}")

    for tool_name, kwargs, expect_block in demo_calls:
        serialized = {"name": tool_name, "id": ["tools", tool_name]}
        input_str = " ".join(f"{k}={v}" for k, v in kwargs.items())

        try:
            from uuid import uuid4

            handler.on_tool_start(
                serialized=serialized,
                input_str=input_str,
                run_id=uuid4(),
            )
            # If we reach here, action was allowed
            result = "✔ ALLOWED"
            status = "ALLOWED"
        except AISeCSecurityError as e:
            result = f"✘ {e.decision.value}"
            status = e.decision.value

        expected = "BLOCK" if expect_block else "ALLOW"
        match = "✔" if (expect_block == (status != "ALLOW")) else "✘ MISMATCH"
        print(f"  {tool_name:<28} {expected:<12} {result}  {match}")

    print()
    print(f"  Total calls intercepted: {handler.call_count}")
    print(f"  Audit entries written:   {engine.audit_count()}")

    ok, errors = engine.verify_audit_chain()
    chain_status = "INTACT ✔" if ok else f"BROKEN ✘ ({len(errors)} errors)"
    print(f"  Audit chain:             {chain_status}")
    print()


if __name__ == "__main__":
    main()
