"""
High-precision profiling utility for PyTorch inference.

Fixes the "Micro-Profiling Inaccuracies" flaw from the original prototype:
`time.perf_counter()` measures host-side (CPU) wall-clock time. Even with
`torch.cuda.synchronize()` called before/after the forward pass -- which
guarantees *correctness* (the measurement won't stop before the CUDA kernel
has actually finished) -- the value itself still includes OS scheduler
jitter, Python/CUDA context-switch overhead, and the limited resolution of
the host clock. For inference in the microsecond/millisecond range on a
compact network like EdgeLSTM, that jitter can be on the same order of
magnitude as the signal being measured.

`torch.cuda.Event(enable_timing=True)` inserts markers directly into the
CUDA stream and measures elapsed time in hardware (on the GPU), via
`elapsed_time()`, isolating the measurement from host-side jitter. It is
the mechanism recommended by the PyTorch documentation for benchmarking
GPU inference latency.

CPU has no equivalent "hardware event" concept (there is no asynchronous
stream to synchronize against), so `time.perf_counter()` remains the best
available option in that case -- and is used only as a fallback.
"""

from __future__ import annotations

import time

import torch


class InferenceTimer:
    """
    Context manager that times a block of code (typically `model(x)`),
    automatically choosing the most accurate profiling mechanism available
    for the given `device`:

        - device.type == "cuda" -> torch.cuda.Event (hardware-level
          measurement, immune to host-side OS jitter).
        - otherwise (CPU)        -> time.perf_counter() (best-effort
          fallback; CPU exposes no asynchronous stream events).

    Usage:
        with InferenceTimer(device) as timer:
            pred = model(x)
        tau_inf = timer.elapsed_s  # seconds, always in this unit
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda_events = device.type == "cuda"
        self.elapsed_s: float = 0.0

        if self.use_cuda_events:
            self._start_evt = torch.cuda.Event(enable_timing=True)
            self._end_evt = torch.cuda.Event(enable_timing=True)
        else:
            self._t0 = 0.0

    def __enter__(self) -> "InferenceTimer":
        if self.use_cuda_events:
            # Drain the stream before marking the start: ensures no
            # previously pending work is included in the measured window.
            torch.cuda.synchronize()
            self._start_evt.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.use_cuda_events:
            self._end_evt.record()
            # Required: elapsed_time() requires both events to have
            # already completed on the device.
            torch.cuda.synchronize()
            elapsed_ms = self._start_evt.elapsed_time(self._end_evt)
            self.elapsed_s = elapsed_ms / 1000.0
        else:
            self.elapsed_s = time.perf_counter() - self._t0
