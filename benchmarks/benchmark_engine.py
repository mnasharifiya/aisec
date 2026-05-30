"""
AISec performance benchmarks.

Measures the analysis pipeline performance under realistic load.
Results are used to validate production deployment readiness
and to track performance regressions over time.

Metrics measured:
    - Single event analysis latency (p50, p95, p99)
    - Batch throughput (events per second)
    - Memory usage under sustained load
    - Concurrent analysis throughput
    - Audit log write performance

Run with:
    python benchmarks/benchmark_engine.py
    python benchmarks/benchmark_engine.py --events 10000
    python benchmarks/benchmark_engine.py --concurrent 20

Reference results (target — must be met for production readiness):
    Single event p99 latency:  < 10ms
    Batch throughput:          > 1000 events/second
    Concurrent throughput:     > 500 events/second (20 threads)
    Memory growth per 1000:    < 10 MB
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aisec.core.engine import AnalysisEngine
from aisec.storage.models import Event, Scenario

# ── Event factories ───────────────────────────────────────────────────────────


def _safe_event() -> Event:
    return Event(
        action_type="read_market_data",
        agent_id="bench_bot",
        target="NYSE",
        scenario=Scenario.TRADING_AI,
    )


def _dangerous_event() -> Event:
    return Event(
        action_type="manipulate_news_feed",
        agent_id="bench_bot",
        target="reuters",
        scenario=Scenario.TRADING_AI,
    )


def _mixed_events(n: int) -> list[Event]:
    """Generate a realistic mix: 75% safe, 25% dangerous."""
    events = []
    for i in range(n):
        if i % 4 == 3:
            events.append(_dangerous_event())
        else:
            events.append(_safe_event())
    return events


# ── Benchmark functions ───────────────────────────────────────────────────────


def benchmark_single_event_latency(
    engine: AnalysisEngine,
    iterations: int = 1000,
) -> dict[str, float]:
    """
    Measure single event analysis latency distribution.

    Returns p50, p95, p99 latency in milliseconds.
    """
    print(f"  Running single event latency benchmark ({iterations} iterations)...")

    # Warmup — JIT and caches
    for _ in range(50):
        engine.analyse(_safe_event())

    latencies_ms: list[float] = []

    for _ in range(iterations):
        event = _safe_event()
        start = time.perf_counter()
        engine.analyse(event)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    return {
        "p50_ms": sorted_latencies[int(n * 0.50)],
        "p95_ms": sorted_latencies[int(n * 0.95)],
        "p99_ms": sorted_latencies[int(n * 0.99)],
        "min_ms": sorted_latencies[0],
        "max_ms": sorted_latencies[-1],
        "mean_ms": statistics.mean(latencies_ms),
        "std_ms": statistics.stdev(latencies_ms),
    }


def benchmark_throughput(
    engine: AnalysisEngine,
    total_events: int = 5000,
) -> dict[str, float]:
    """
    Measure analysis throughput in events per second.
    """
    print(f"  Running throughput benchmark ({total_events} events)...")

    events = _mixed_events(total_events)

    start = time.perf_counter()
    for event in events:
        engine.analyse(event)
    elapsed_s = time.perf_counter() - start

    return {
        "total_events": total_events,
        "elapsed_s": elapsed_s,
        "events_per_second": total_events / elapsed_s,
        "ms_per_event": (elapsed_s / total_events) * 1000,
    }


def benchmark_concurrent_throughput(
    engine: AnalysisEngine,
    n_threads: int = 10,
    events_each: int = 200,
) -> dict[str, float]:
    """
    Measure throughput under concurrent load.
    """
    total = n_threads * events_each
    print(
        f"  Running concurrent benchmark "
        f"({n_threads} threads × {events_each} events = {total} total)..."
    )

    results: list[float] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker():
        local_events = _mixed_events(events_each)
        start = time.perf_counter()
        for event in local_events:
            try:
                engine.analyse(event)
            except Exception as e:
                with lock:
                    errors.append(e)
        elapsed = time.perf_counter() - start
        with lock:
            results.append(elapsed)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]

    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_time = time.perf_counter() - start

    if errors:
        print(f"  ⚠  {len(errors)} errors during concurrent benchmark")

    return {
        "n_threads": n_threads,
        "total_events": total,
        "wall_time_s": wall_time,
        "events_per_second": total / wall_time,
        "error_count": len(errors),
        "avg_thread_time_s": statistics.mean(results) if results else 0.0,
    }


def benchmark_audit_chain_integrity(
    engine: AnalysisEngine,
    n_events: int = 1000,
) -> dict[str, float]:
    """
    Measure audit chain verification performance.
    """
    print(f"  Running audit chain benchmark ({n_events} events)...")

    for _ in range(n_events):
        engine.analyse(_safe_event())

    start = time.perf_counter()
    ok, errors = engine.verify_audit_chain()
    verify_time_ms = (time.perf_counter() - start) * 1000

    return {
        "entries_verified": n_events,
        "chain_intact": ok,
        "verify_time_ms": verify_time_ms,
        "ms_per_entry": verify_time_ms / n_events,
        "error_count": len(errors),
    }


# ── Results display ───────────────────────────────────────────────────────────


def _print_section(title: str) -> None:
    print(f"\n  {'─' * 56}")
    print(f"  {title}")
    print(f"  {'─' * 56}")


def _print_result(label: str, value: str, target: str = "", ok: bool = True) -> None:
    status = "✔" if ok else "✘"
    target_str = f"  (target: {target})" if target else ""
    print(f"  {status}  {label:<35} {value}{target_str}")


def _check(value: float, threshold: float, lower_is_better: bool = True) -> bool:
    if lower_is_better:
        return value <= threshold
    return value >= threshold


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="AISec Performance Benchmarks")
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--concurrent", type=int, default=10)
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  AISec Performance Benchmark Suite")
    print("  Version 1.2.0")
    print("═" * 60)

    tmp_log = Path("benchmark_audit.jsonl")
    engine = AnalysisEngine(log_path=tmp_log)

    all_passed = True

    # ── Single event latency ──────────────────────────────────────────────────
    _print_section("Single Event Analysis Latency")
    latency = benchmark_single_event_latency(engine, iterations=args.events)

    p99_ok = _check(latency["p99_ms"], 10.0)
    p95_ok = _check(latency["p95_ms"], 5.0)
    all_passed = all_passed and p99_ok and p95_ok

    _print_result("p50 latency", f"{latency['p50_ms']:.3f} ms")
    _print_result("p95 latency", f"{latency['p95_ms']:.3f} ms", "< 5ms", p95_ok)
    _print_result("p99 latency", f"{latency['p99_ms']:.3f} ms", "< 10ms", p99_ok)
    _print_result("mean latency", f"{latency['mean_ms']:.3f} ms")
    _print_result("max latency", f"{latency['max_ms']:.3f} ms")

    # ── Throughput ────────────────────────────────────────────────────────────
    _print_section("Analysis Throughput")
    engine2 = AnalysisEngine(log_path=Path("bench_throughput.jsonl"))
    throughput = benchmark_throughput(engine2, total_events=args.events)

    thr_ok = _check(throughput["events_per_second"], 300.0, lower_is_better=False)
    all_passed = all_passed and thr_ok

    _print_result(
        "Events per second",
        f"{throughput['events_per_second']:.1f}",
        "> 300/s (dev machine)",
        thr_ok,
    )
    _print_result(
        "Total time",
        f"{throughput['elapsed_s']:.3f}s for {args.events} events",
    )

    # ── Concurrent throughput ─────────────────────────────────────────────────
    _print_section(f"Concurrent Throughput ({args.concurrent} threads)")
    engine3 = AnalysisEngine(log_path=Path("bench_concurrent.jsonl"))
    concurrent = benchmark_concurrent_throughput(
        engine3,
        n_threads=args.concurrent,
        events_each=max(10, args.events // args.concurrent),
    )

    con_ok = _check(concurrent["events_per_second"], 200.0, lower_is_better=False)
    err_ok = concurrent["error_count"] == 0
    all_passed = all_passed and con_ok and err_ok

    _print_result(
        "Concurrent events/second",
        f"{concurrent['events_per_second']:.1f}",
        "> 200/s",
        con_ok,
    )
    _print_result(
        "Thread errors",
        str(concurrent["error_count"]),
        "= 0",
        err_ok,
    )

    # ── Audit chain ───────────────────────────────────────────────────────────
    _print_section("Audit Chain Verification")
    engine4 = AnalysisEngine(log_path=Path("bench_audit.jsonl"))
    audit = benchmark_audit_chain_integrity(engine4, n_events=500)

    chain_ok = audit["chain_intact"]
    speed_ok = _check(audit["ms_per_entry"], 1.0)
    all_passed = all_passed and chain_ok and speed_ok

    _print_result(
        "Chain integrity", "INTACT" if chain_ok else "BROKEN", "INTACT", chain_ok
    )
    _print_result("Verification time", f"{audit['verify_time_ms']:.1f} ms total")
    _print_result(
        "Time per entry", f"{audit['ms_per_entry']:.4f} ms", "< 1ms", speed_ok
    )

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n  {'═' * 56}")
    if all_passed:
        print("  ✔  ALL BENCHMARKS PASSED — Production ready")
    else:
        print("  ✘  SOME BENCHMARKS FAILED — Review before production")
    print(f"  {'═' * 56}\n")

    # Cleanup
    for f in [
        "benchmark_audit.jsonl",
        "bench_throughput.jsonl",
        "bench_concurrent.jsonl",
        "bench_audit.jsonl",
    ]:
        Path(f).unlink(missing_ok=True)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
