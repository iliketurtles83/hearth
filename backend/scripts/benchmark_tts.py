from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tts
from tts import TTSError


DEFAULT_PROMPTS = [
    "System online.",
    "Weather in Tallinn is 14 degrees Celsius with light rain.",
    "I queued five tracks from Daft Punk and started playback.",
]


@dataclass
class EngineBenchmarkResult:
    engine: str
    available: bool
    error_code: str | None
    error: str | None
    cold_start_ms: float | None
    warm_mean_ms: float | None
    warm_p95_ms: float | None
    bytes_mean: float | None


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    idx = int(round(0.95 * (len(values) - 1)))
    return values[idx]


async def _bench_engine(engine: str, prompts: list[str], iterations: int) -> EngineBenchmarkResult:
    latencies: list[float] = []
    sizes: list[int] = []
    cold_start: float | None = None

    try:
        tts.clear_engine_cache()
        for i in range(iterations):
            for prompt in prompts:
                t0 = time.perf_counter()
                audio = await tts.synthesize(prompt, engine_name=engine)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if cold_start is None:
                    cold_start = elapsed_ms
                else:
                    latencies.append(elapsed_ms)
                sizes.append(len(audio))

        warm_mean = statistics.mean(latencies) if latencies else None
        warm_p95 = _p95(latencies) if latencies else None
        return EngineBenchmarkResult(
            engine=engine,
            available=True,
            error_code=None,
            error=None,
            cold_start_ms=cold_start,
            warm_mean_ms=warm_mean,
            warm_p95_ms=warm_p95,
            bytes_mean=statistics.mean(sizes) if sizes else None,
        )
    except TTSError as exc:
        return EngineBenchmarkResult(
            engine=engine,
            available=False,
            error_code=exc.code,
            error=exc.message,
            cold_start_ms=None,
            warm_mean_ms=None,
            warm_p95_ms=None,
            bytes_mean=None,
        )


def _pick_winner(results: list[EngineBenchmarkResult]) -> str | None:
    available = [r for r in results if r.available and r.warm_mean_ms is not None]
    if not available:
        return None
    return min(available, key=lambda r: (r.warm_mean_ms or 10**9, r.cold_start_ms or 10**9)).engine


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.1f}"


def _print_summary(results: list[EngineBenchmarkResult], winner: str | None) -> None:
    print("engine,available,cold_start_ms,warm_mean_ms,warm_p95_ms,bytes_mean,error_code")
    for r in results:
        print(
            f"{r.engine},{str(r.available).lower()},{_fmt(r.cold_start_ms)},{_fmt(r.warm_mean_ms)},{_fmt(r.warm_p95_ms)},{_fmt(r.bytes_mean)},{r.error_code or ''}"
        )

    if winner:
        print(f"\nwinner={winner}")
    else:
        print("\nwinner=none")


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark local TTS engines")
    p.add_argument("--engines", default="piper,kokoro", help="Comma-separated engine list")
    p.add_argument("--iterations", type=int, default=int(os.getenv("TTS_BENCH_ITERATIONS", "3")))
    p.add_argument(
        "--prompts-json",
        default="",
        help="Optional JSON array of prompts; defaults to built-in prompts",
    )
    p.add_argument("--json-out", default="", help="Optional path to write JSON results")
    return p


async def _run(args: argparse.Namespace) -> int:
    engines = [e.strip().lower() for e in args.engines.split(",") if e.strip()]
    prompts = DEFAULT_PROMPTS
    if args.prompts_json:
        prompts = json.loads(args.prompts_json)
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("--prompts-json must be a JSON array of strings")

    results: list[EngineBenchmarkResult] = []
    for engine in engines:
        results.append(await _bench_engine(engine, prompts, max(1, args.iterations)))

    winner = _pick_winner(results)
    _print_summary(results, winner)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "winner": winner,
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return 0


def main() -> int:
    parser = _make_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
