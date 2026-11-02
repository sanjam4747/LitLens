"""
Performance and logging utilities for LitLens.
Tracks memory usage, execution times, and formatting.
"""

import os
import time
import psutil
from contextlib import contextmanager
from typing import Generator, Dict, Any


def get_memory_usage_mb() -> float:
    """Return the current process RSS memory consumption in megabytes."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


@contextmanager
def log_stage(stage_name: str) -> Generator[Dict[str, Any], None, None]:
    """Context manager to measure and print execution time and memory delta for a pipeline stage."""
    mem_start = get_memory_usage_mb()
    t_start = time.perf_counter()
    info: Dict[str, Any] = {"stage": stage_name, "start_time": t_start, "mem_start_mb": mem_start}
    
    print(f"\n[START] {stage_name} (Initial RAM: {mem_start:.1f} MB)")
    try:
        yield info
    finally:
        t_end = time.perf_counter()
        mem_end = get_memory_usage_mb()
        duration = t_end - t_start
        mem_delta = mem_end - mem_start
        info["duration_sec"] = duration
        info["mem_end_mb"] = mem_end
        info["mem_delta_mb"] = mem_delta
        print(f"[DONE]  {stage_name} in {duration:.2f}s | RAM: {mem_end:.1f} MB (Delta: {mem_delta:+.1f} MB)")

