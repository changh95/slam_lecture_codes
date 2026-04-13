"""
py_profiler.py - Lightweight Python profiler that emits JSON compatible with
analyze_profiler.py (the same schema as easy_profiler_converter output).

Usage:
    from py_profiler import Profiler, block

    prof = Profiler()
    prof.enable()

    with block("SLAM/FrameProcess"):
        do_work()
        with block("SLAM/SubStep"):
            more_work()

    prof.dump("/output/cuvslam_profiler.json")
"""

import json
import os
import time
import threading
from contextlib import contextmanager


class _Block:
    __slots__ = ("name", "start", "stop", "children")

    def __init__(self, name: str, start_ns: int):
        self.name = name
        self.start = start_ns
        self.stop = 0
        self.children = []


class Profiler:
    """Thread-aware profiler that captures nested blocks and dumps them as JSON."""

    _instance = None

    def __init__(self):
        self._enabled = False
        self._threads = {}  # thread_id -> {"roots": [], "stack": []}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "Profiler":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def _get_thread_state(self):
        tid = threading.get_ident()
        state = self._threads.get(tid)
        if state is None:
            state = {"roots": [], "stack": [], "name": threading.current_thread().name}
            self._threads[tid] = state
        return state

    def _begin(self, name: str) -> _Block:
        if not self._enabled:
            return None
        start_ns = time.perf_counter_ns()
        blk = _Block(name, start_ns)
        with self._lock:
            state = self._get_thread_state()
            if state["stack"]:
                state["stack"][-1].children.append(blk)
            else:
                state["roots"].append(blk)
            state["stack"].append(blk)
        return blk

    def _end(self, blk: _Block):
        if blk is None:
            return
        blk.stop = time.perf_counter_ns()
        with self._lock:
            state = self._get_thread_state()
            if state["stack"] and state["stack"][-1] is blk:
                state["stack"].pop()

    def dump(self, path: str):
        """Write profiler data to a JSON file matching easy_profiler schema."""
        def block_to_dict(b: _Block, idx: int) -> dict:
            return {
                "id": idx,
                "name": b.name,
                "start": b.start,
                "stop": b.stop,
                "descriptor": 0,
                "children": [block_to_dict(c, i) for i, c in enumerate(b.children)],
            }

        threads_out = []
        with self._lock:
            for tid, state in self._threads.items():
                threads_out.append({
                    "threadId": tid,
                    "threadName": state["name"],
                    "children": [block_to_dict(r, i) for i, r in enumerate(state["roots"])],
                })

        data = {
            "version": "py_profiler-1.0",
            "timeUnits": "ns",
            "blockDescriptors": [],
            "threads": threads_out,
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Profiler data saved to {path}")


@contextmanager
def block(name: str):
    """Context manager for profiling a code block."""
    prof = Profiler.instance()
    blk = prof._begin(name)
    try:
        yield
    finally:
        prof._end(blk)


def enable():
    Profiler.instance().enable()


def dump(path: str):
    Profiler.instance().dump(path)
