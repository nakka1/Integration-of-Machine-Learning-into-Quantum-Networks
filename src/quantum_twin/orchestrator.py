"""
Component 4 -- Orchestrator (`DigitalTwinOrchestrator`).

Keeps two strictly separate methods, each with its own profiling regime:

    - run_intelligent      : timer isolated around `self.model(x)`,
                               measuring only tau_inf (the forward pass),
                               via `InferenceTimer` (CUDA Events on GPU,
                               `perf_counter` on CPU -- see timing.py).
                               Applies the admission control
                               (HALT_PURIFICATION vs PURIFY), and tracks
                               the resulting admission CONFUSION MATRIX
                               (TP/FP/TN/FN -- see below).
    - run_blind_baseline    : NEVER invokes the neural network. Classical
                               latency is forced to 0.0 and admission is
                               unconditional (its own degenerate confusion
                               matrix is reported for context: it always
                               "predicts admit", so FN == TN == 0 by
                               construction).

Admission confusion matrix
---------------------------
Ground truth ("good"/"bad" photon) is `true_fidelity >= threshold`; the
predicted label is the admission DECISION itself (`pred_fidelity >=
threshold` => PURIFY/"admit", i.e. NOT halted):

    TP : good photon, correctly admitted.
    FP : a DEAD photon admitted (F_true < threshold <= F_pred) -- wastes a
         purification attempt (QPU time/shots/energy) on a pair that was
         never going to be useful. This is exactly the error
         `CS_MSELoss.lambda_penalty` penalizes severely.
    FN : a GOOD photon discarded (F_pred < threshold <= F_true) -- a
         usable pair is thrown away, directly reducing throughput.
         Penalized moderately by `CS_MSELoss.lambda_fn`.
    TN : bad photon, correctly halted.

Both loops report `confusion_matrix: {"TP", "FP", "TN", "FN"}` in their
returned metrics dict; `metrics.compute_confusion_metrics` derives
precision/recall/FPR/FNR/F1 from it.
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
                 device: torch.device | None = None) -> None:
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

        Every cycle's admission decision (PURIFY/HALT) is compared against
        ground truth (`true_fidelity >= threshold`) and tallied into the
        confusion matrix described in the module docstring -- this happens
        for EVERY step, including halted ones, so `TP + FP + TN + FN ==
        total_steps` always holds.
        """
        assert self.model is not None, "run_intelligent requires a trained model."
        self.model.eval()

        results = []
        useful_pairs = 0
        halted = 0
        total_forward_latency = 0.0
        total_steps = len(X_test)
        confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

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

                is_true_good = true_fidelity >= self.threshold
                is_pred_admit = pred_fidelity >= self.threshold
                if is_pred_admit and is_true_good:
                    confusion["TP"] += 1
                elif is_pred_admit and not is_true_good:
                    confusion["FP"] += 1  # dead photon admitted
                elif (not is_pred_admit) and is_true_good:
                    confusion["FN"] += 1  # good photon discarded
                else:
                    confusion["TN"] += 1

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
            "confusion_matrix": confusion,
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

        Because admission is unconditional, the confusion matrix is
        degenerate by construction: every cycle counts as "predicted
        admit", so `FN == TN == 0` always, `TP` counts truly-good photons
        purified (correctly, by luck of unconditional admission) and `FP`
        counts truly-dead photons purified anyway (wasted QPU time) --
        this is precisely the failure mode a predictive controller with an
        asymmetric (`CS_MSELoss`) penalty is meant to reduce.
        """
        results = []
        useful_pairs = 0
        total_steps = len(X_test)
        forced_latency = 0.0  # No AI inference => no memory wait.
        confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        for i in range(total_steps):
            true_fidelity = float(y_test[i].item())

            aged_simulator = self.quantum_node.apply_latency_decay(forced_latency)
            success_rate, _counts = self.quantum_node.run_purification(simulator=aged_simulator)

            is_useful = (success_rate >= self.success_rate_cutoff) and (true_fidelity >= self.threshold)
            if is_useful:
                useful_pairs += 1

            if true_fidelity >= self.threshold:
                confusion["TP"] += 1
            else:
                confusion["FP"] += 1  # dead photon admitted (unconditional admission)

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
            "confusion_matrix": confusion,
        }
