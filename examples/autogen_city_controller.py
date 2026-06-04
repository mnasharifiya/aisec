"""
AISec Example — AutoGen Smart City Controller Monitoring

Demonstrates how to integrate AISec with an AutoGen agent
to monitor an autonomous smart city AI controller.

What this example shows:
    - How to wrap an AutoGen function_map with AISec
    - How urban-specific rules fire for city control actions
    - How ESCALATE differs from BLOCK in practice
    - How to use block_on_review=True for conservative deployment

Prerequisites:
    pip install pyautogen aisec

Run:
    python examples/autogen_city_controller.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aisec.core.engine import AnalysisEngine
from aisec.integrations.autogen import (
    AISeCAutoGenWrapper,
    AISeCAutoGenSecurityError,
)
from aisec.storage.models import Scenario

# ── Simulated city control functions ─────────────────────────────────────────


def read_sensor(sensor_id: str = "traffic_01") -> str:
    return f"Sensor {sensor_id}: normal operation"


def adjust_traffic_flow(intersection: str = "42", green_duration: int = 45) -> str:
    return f"Traffic flow adjusted at intersection {intersection}"


def set_curfew(zone: str = "ALL", duration_hours: int = 48) -> str:
    """Impose city curfew — blocked unconditionally by AISec."""
    return f"Curfew imposed: zone={zone} duration={duration_hours}h"


def shutdown_power_grid(zone: str = "North") -> str:
    """Shutdown power grid — escalated by AISec."""
    return f"Power grid shutdown: zone={zone}"


def mass_traffic_redirect(affected_intersections: int = 120) -> str:
    """Mass traffic redirect — reviewed if > 50 intersections."""
    return f"Traffic redirected: {affected_intersections} intersections"


def adjust_routing(target: str = "ambulance_routing") -> str:
    """Adjust routing — blocked if targeting emergency services."""
    return f"Routing adjusted for {target}"


# ── Main demonstration ────────────────────────────────────────────────────────


def main() -> None:
    print("\n" + "=" * 60)
    print("  AISec — AutoGen Smart City Controller Example")
    print("=" * 60)

    engine = AnalysisEngine(log_path=Path(".aisec/autogen_example.jsonl"))

    wrapper = AISeCAutoGenWrapper(
        engine=engine,
        scenario=Scenario.URBAN_AI,
        agent_id="autogen_urban_ctrl_v1",
        block_on_review=True,
    )

    # Original function map — what the AutoGen agent would use
    original_map = {
        "read_sensor": read_sensor,
        "adjust_traffic_flow": adjust_traffic_flow,
        "set_curfew": set_curfew,
        "shutdown_power_grid": shutdown_power_grid,
        "mass_traffic_redirect": mass_traffic_redirect,
        "adjust_routing": adjust_routing,
    }

    # Wrap with AISec — every call now passes through the engine
    safe_map = wrapper.wrap_function_map(original_map)

    print(f"\n  Wrapped {len(safe_map)} functions with AISec interception.")
    print(f"  Agent: {wrapper.agent_id} | Scenario: {wrapper.scenario.value}")
    print()

    # Demonstrate calls
    demo_calls = [
        ("read_sensor", {"sensor_id": "traffic_north"}, False),
        ("adjust_traffic_flow", {"intersection": "42"}, False),
        ("mass_traffic_redirect", {"affected_intersections": 10}, False),
        ("mass_traffic_redirect", {"affected_intersections": 120}, True),
        ("set_curfew", {"zone": "ALL", "duration_hours": 48}, True),
        ("shutdown_power_grid", {"zone": "North"}, True),
        ("adjust_routing", {"target": "ambulance_routing"}, True),
        ("read_sensor", {"sensor_id": "power_monitor_1"}, False),
    ]

    print(f"  {'Function':<28} {'Expected':<12} {'Result'}")
    print(f"  {'─'*28} {'─'*12} {'─'*20}")

    for func_name, kwargs, expect_block in demo_calls:
        try:
            safe_map[func_name](**kwargs)
            result = "✔ ALLOWED"
            blocked = False
        except AISeCAutoGenSecurityError as e:
            result = f"✘ {e.decision.value}"
            blocked = True

        expected = "BLOCK" if expect_block else "ALLOW"
        match = "✔" if (expect_block == blocked) else "✘ MISMATCH"
        print(f"  {func_name:<28} {expected:<12} {result}  {match}")

    print()
    print(f"  Calls intercepted: {wrapper.call_count}")
    print(f"  Calls blocked:     {wrapper.blocked_count}")
    print(f"  Block rate:        {wrapper.block_rate:.1%}")
    print(f"  Audit entries:     {engine.audit_count()}")

    ok, _ = engine.verify_audit_chain()
    print(f"  Audit chain:       {'INTACT ✔' if ok else 'BROKEN ✘'}")
    print()


if __name__ == "__main__":
    main()
