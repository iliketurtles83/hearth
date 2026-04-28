"""
Measure Ollama model cold-swap latency between gemma3:4b and qwen2.5-coder:7b.

Run this BEFORE Phase 10b to establish a baseline. The result determines how
much UX work is worth doing for the loading state in the code_tool node.

Usage (from repo root, with venv active):
    python -m backend.tests.test_swap_latency

Or pass custom models / iterations:
    SWAP_CHAT_MODEL=gemma3:4b SWAP_CODER_MODEL=qwen2.5-coder:7b SWAP_ITERS=10 \
        python -m backend.tests.test_swap_latency

Results are printed as a summary table and a baseline comment ready to paste
into main.py.
"""

import os
import statistics
import time

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("SWAP_CHAT_MODEL", "gemma3:4b")
CODER_MODEL = os.getenv("SWAP_CODER_MODEL", "qwen2.5-coder:7b")
ITERS = int(os.getenv("SWAP_ITERS", "10"))
TIMEOUT = 120  # seconds per call — model load can take a while

TRIVIAL_PROMPT = "Reply with one word: ready"


def _unload(model: str) -> None:
    """Ask Ollama to evict a model by calling generate with keep_alive=0."""
    try:
        httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": 0},
            timeout=30,
        )
    except Exception:
        pass  # best-effort; model may already be unloaded


def _warm_generate(model: str) -> float:
    """
    Send a trivial prompt to ``model`` and return wall-clock seconds until
    the first token arrives (time-to-first-token, TTFT).

    Uses the streaming generate endpoint so we can stop as soon as the first
    token arrives rather than waiting for the full response.
    """
    start = time.perf_counter()
    with httpx.stream(
        "POST",
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": TRIVIAL_PROMPT, "stream": True},
        timeout=TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for _ in resp.iter_lines():
            # First line == first token chunk; stop immediately.
            break
    return time.perf_counter() - start


def measure_swap(from_model: str, to_model: str, iters: int) -> list[float]:
    """
    Measure the wall-clock time for a cold swap from *from_model* to *to_model*.

    Each iteration:
      1. Ensure *from_model* is loaded (trivial prompt, not timed).
      2. Unload *from_model* (keep_alive=0).
      3. Time how long *to_model* takes to produce its first token (TTFT).

    Returns a list of TTFT values in seconds.
    """
    results: list[float] = []

    # Initial warm-up: make sure from_model is cached.
    print(f"  Warming up {from_model!r}…", flush=True)
    _warm_generate(from_model)

    for i in range(iters):
        # Step 1: ensure from_model is resident (re-warm after each iteration).
        if i > 0:
            _warm_generate(from_model)

        # Step 2: evict from_model.
        _unload(from_model)

        # Step 3: time cold load of to_model.
        elapsed = _warm_generate(to_model)
        results.append(elapsed)
        print(f"  iter {i + 1:2d}/{iters}: {elapsed:.2f}s", flush=True)

    return results


def _stats(values: list[float]) -> dict:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    print(f"\nOllama swap latency benchmark")
    print(f"  OLLAMA_URL  : {OLLAMA_URL}")
    print(f"  chat model  : {CHAT_MODEL}")
    print(f"  coder model : {CODER_MODEL}")
    print(f"  iterations  : {ITERS}")
    print()

    # ── Direction 1: chat → coder ──────────────────────────────────────────────
    print(f"[1/2] Measuring {CHAT_MODEL} → {CODER_MODEL} swap …")
    chat_to_coder = measure_swap(CHAT_MODEL, CODER_MODEL, ITERS)
    s1 = _stats(chat_to_coder)

    # ── Direction 2: coder → chat ──────────────────────────────────────────────
    print(f"\n[2/2] Measuring {CODER_MODEL} → {CHAT_MODEL} swap …")
    coder_to_chat = measure_swap(CODER_MODEL, CHAT_MODEL, ITERS)
    s2 = _stats(coder_to_chat)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS (time-to-first-token after cold swap, seconds)")
    print("=" * 60)
    print(f"\n{CHAT_MODEL} → {CODER_MODEL}:")
    print(f"  min={s1['min']:.2f}  max={s1['max']:.2f}  "
          f"mean={s1['mean']:.2f}  median={s1['median']:.2f}  "
          f"stdev={s1['stdev']:.2f}")

    print(f"\n{CODER_MODEL} → {CHAT_MODEL}:")
    print(f"  min={s2['min']:.2f}  max={s2['max']:.2f}  "
          f"mean={s2['mean']:.2f}  median={s2['median']:.2f}  "
          f"stdev={s2['stdev']:.2f}")

    # ── Baseline comment for main.py ───────────────────────────────────────────
    overall_median = statistics.median(chat_to_coder + coder_to_chat)
    print("\n" + "-" * 60)
    print("Paste this comment near the model config in backend/main.py:")
    print("-" * 60)
    print(
        f"# Measured cold-swap latency ({CHAT_MODEL} ↔ {CODER_MODEL}, "
        f"n={ITERS} each, RTX 3060 12 GB):\n"
        f"#   {CHAT_MODEL}→{CODER_MODEL}: "
        f"median={s1['median']:.1f}s  min={s1['min']:.1f}s  max={s1['max']:.1f}s\n"
        f"#   {CODER_MODEL}→{CHAT_MODEL}: "
        f"median={s2['median']:.1f}s  min={s2['min']:.1f}s  max={s2['max']:.1f}s\n"
        f"#   Overall median: {overall_median:.1f}s — "
        + ("barely noticeable; skip loading-state UX."
           if overall_median < 5 else
           "noticeable; add loading-state badge in Phase 10b."
           if overall_median < 15 else
           "significant; loading-state UX is required for Phase 10b.")
    )
    print("-" * 60)


if __name__ == "__main__":
    main()
