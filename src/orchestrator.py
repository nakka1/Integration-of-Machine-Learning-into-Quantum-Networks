"""
Component 4 -- Orchestrator (`DigitalTwinOrchestrator`).

Keeps two strictly separate methods, each with its own profiling regime:

    - run_intelligent      : timer isolated around `self.model(x)`,
                               measuring only tau_inf (the forward pass),
                               via `InferenceTimer` (CUDA Events on GPU,
                               `perf_counter` on CPU -- see timing.py).
                               Applies the admission control
                               (HALT_PURIFICATION vs PURIFY).
    - run_blind_baseline    : NEVER invokes the neural network. Classical
                               latency is forced to 0.0 and admission is
                               unconditional.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .quantum_node import QuantumRepeaterNode
from .timing import InferenceTimer


class DigitalTwinOrchestrator:
    """
    Central orchestrator of the Quantum Repeater Digital Twin.

    Keeps two strictly separate simulation loops to eliminate any
    cross-contamination in profiling between the intelligent (predictive)
    approach and the blind/reactive (baseline) approach.
    """

    def __init__(self, model: nn.Module, quantum_node: QuantumRepeaterNode,
                 threshold: float = 0.65, success_rate_cutoff: float = 0.5,
                 device: torch.device = None):
        self.model = model
        self.quantum_node = quantum_node
        self.threshold = threshold
        self.success_rate_cutoff = success_rate_cutoff
        self.device = device if device is not None else torch.device("cpu")
        self.log = []

    def run_intelligent(self, X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
        """
        Simulation loop with predictive admission control (EdgeLSTM + CS_MSELoss).

        `InferenceTimer` wraps STRICTLY the `self.model(x_sample)` call --
        no other operation (scalar extraction via `.item()`, threshold
        comparison, call into the quantum dataplane) enters the timed
        window. On GPU, the measurement is done with `torch.cuda.Event`
        (hardware markers on the CUDA stream), avoiding the host-clock
        jitter inherent to `time.perf_counter()`; on CPU, `InferenceTimer`
        falls back to `perf_counter()`, since there is no asynchronous
        stream to instrument.
        """
        assert self.model is not None, "run_intelligent requires a trained model."
        self.model.eval()

        results = []
        useful_pairs = 0
        halted = 0
        total_forward_latency = 0.0
        total_steps = len(X_test)

        with torch.no_grad():
            for i in range(total_steps):
                x_sample = X_test[i:i + 1]
                true_fidelity = float(y_test[i].item())

                # --- Isolated profiling: times STRICTLY the forward pass ---
                with InferenceTimer(self.device) as timer:
                    pred_tensor = self.model(x_sample)
                tau_inf = timer.elapsed_s
                # --- End of timed window ---

                pred_fidelity = float(pred_tensor.item())
                total_forward_latency += tau_inf

                if pred_fidelity < self.threshold:
                    halted += 1
                    results.append({
                        "step": i, "action": "HALT_PURIFICATION",
                        "pred_fidelity": pred_fidelity, "true_fidelity": true_fidelity,
                        "latency_s": tau_inf,
                    })
                    continue

                # Approved: dispatches the isolated classical latency to the
                # quantum node (memory aging) and runs the purification circuit.
                aged_simulator = self.quantum_node.apply_latency_decay(tau_inf)
                success_rate, _counts = self.quantum_node.run_purification(simulator=aged_simulator)

                is_useful = (success_rate >= self.success_rate_cutoff) and (true_fidelity >= self.threshold)
                if is_useful:
                    useful_pairs += 1

                results.append({
                    "step": i, "action": "PURIFY",
                    "pred_fidelity": pred_fidelity, "true_fidelity": true_fidelity,
                    "latency_s": tau_inf, "purification_success_rate": success_rate,
                    "useful": is_useful,
                })

        self.log = results
        return {
            "mode": "intelligent",
            "total_steps": total_steps,
            "useful_pairs": useful_pairs,
            "halted": halted,
            "attempted": total_steps - halted,
            "avg_classical_latency_s": total_forward_latency / max(total_steps, 1),
        }

    def run_blind_baseline(self, X_test: torch.Tensor, y_test: torch.Tensor) -> dict:
        """
        Simulation loop for the blind/reactive (baseline) approach.

        Admission is UNCONDITIONAL: every window is purified, with no
        consultation of the predictive model whatsoever. The neural network
        is NEVER instantiated nor called in this routine -- there is no
        residual "background call". Classical latency is therefore
        correctly forced and recorded as 0.0 seconds (there is no inference
        wait to time).
        """
        results = []
        useful_pairs = 0
        total_steps = len(X_test)
        forced_latency = 0.0  # No AI inference => no memory wait.

        for i in range(total_steps):
            true_fidelity = float(y_test[i].item())

            aged_simulator = self.quantum_node.apply_latency_decay(forced_latency)
            success_rate, _counts = self.quantum_node.run_purification(simulator=aged_simulator)

            is_useful = (success_rate >= self.success_rate_cutoff) and (true_fidelity >= self.threshold)
            if is_useful:
                useful_pairs += 1

            results.append({
                "step": i, "action": "PURIFY_BLIND",
                "true_fidelity": true_fidelity, "latency_s": forced_latency,
                "purification_success_rate": success_rate, "useful": is_useful,
            })

        self.log = results
        return {
            "mode": "blind",
            "total_steps": total_steps,
            "useful_pairs": useful_pairs,
            "halted": 0,
            "attempted": total_steps,
            "avg_classical_latency_s": forced_latency,
        }
