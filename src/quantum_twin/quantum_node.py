"""
Component 3 -- Virtual Quantum Dataplane (`QuantumRepeaterNode`).

Implements the BBPSSW protocol under NISQ noise (depolarization + T1/T2),
with the "logical latency clock" applied via a thermal-relaxation channel
e^(-latency/T2). The circuit is built and transpiled exactly once per
instance (avoids thousands of redundant recompilations during the Pareto
sweep, since the circuit structure doesn't change across runs -- only the
simulator's noise model does).
"""

from __future__ import annotations

from typing import Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


class QuantumRepeaterNode:
    """
    Virtual quantum dataplane of a repeater node, emulated via Qiskit Aer.

    Implements the BBPSSW entanglement-purification circuit under a custom
    NISQ noise model (depolarization + T1/T2 relaxation), and exposes a
    "logical latency clock" that ages the quantum memory in proportion to
    the isolated classical inference time.
    """

    def __init__(self, T1: float = 50e-6, T2: float = 30e-6,
                 depol_prob: float = 0.01, shots: int = 512, seed: int = 7) -> None:
        assert T2 <= 2 * T1, "Physical constraint: T2 must be <= 2*T1"
        self.T1 = T1
        self.T2 = T2
        self.depol_prob = depol_prob
        self.shots = shots
        self.seed = seed

        self.base_noise_model = self._build_noise_model()
        self.simulator = AerSimulator(noise_model=self.base_noise_model, seed_simulator=seed)

        # The BBPSSW circuit is structural and doesn't change across runs --
        # only the simulator's noise model varies (via apply_latency_decay).
        # For this reason it is built and transpiled ONCE here, and reused
        # by run_purification() on every subsequent call, eliminating the
        # bottleneck of thousands of recompilations in the simulation loop /
        # Pareto sweep.
        self._circuit = self.build_bbpssw_circuit()
        self._compiled_circuit = transpile(self._circuit, self.simulator)

    def _build_noise_model(self, extra_relax_error=None) -> NoiseModel:
        """
        Builds the NISQ noise model: depolarization + T1/T2 relaxation on
        logical gates, and (optionally) an additional relaxation error
        mapped to the 'id' gate, used to represent aging due to classical
        latency.
        """
        noise_model = NoiseModel()

        error_1q = depolarizing_error(self.depol_prob, 1)
        error_2q = depolarizing_error(self.depol_prob * 2, 2)

        gate_time_1q = 50e-9
        gate_time_2q = 300e-9

        thermal_1q = thermal_relaxation_error(self.T1, self.T2, gate_time_1q)
        thermal_2q_single = thermal_relaxation_error(self.T1, self.T2, gate_time_2q)
        thermal_2q = thermal_2q_single.tensor(thermal_2q_single)

        # Composes depolarization + thermal relaxation into a single
        # QuantumError per gate, avoiding multiple calls to
        # add_all_qubit_quantum_error on the same instruction (which would
        # generate redundant composition warnings).
        combined_1q = error_1q.compose(thermal_1q)
        combined_2q = error_2q.compose(thermal_2q)

        noise_model.add_all_qubit_quantum_error(combined_1q, ["u1", "u2", "u3", "x", "h"])
        noise_model.add_all_qubit_quantum_error(combined_2q, ["cx"])

        if extra_relax_error is not None:
            noise_model.add_all_qubit_quantum_error(extra_relax_error, ["id"])

        return noise_model

    def apply_latency_decay(self, latency: float) -> AerSimulator:
        """
        Logical latency clock.

        Takes the computational time (in seconds) measured in isolation
        (strictly the EdgeLSTM forward pass, or 0.0 in the blind baseline)
        and builds a simulator whose noise model includes a thermal
        relaxation channel equivalent to that interval, applied to the
        'id' gate of the circuit (quantum memory sitting idle, waiting on
        the classical AI's decision). With latency=0.0 (the baseline case),
        the aging channel is effectively null, reflecting unconditional
        admission with no additional wait.
        """
        latency = max(latency, 0.0)
        aging_error = thermal_relaxation_error(self.T1, self.T2, latency) if latency > 0.0 else None
        aged_noise_model = self._build_noise_model(extra_relax_error=aging_error)
        aged_simulator = AerSimulator(noise_model=aged_noise_model, seed_simulator=self.seed)
        return aged_simulator

    @staticmethod
    def build_bbpssw_circuit() -> QuantumCircuit:
        """
        BBPSSW entanglement-purification circuit.

        Qubits 0, 1 -> Bell pair "A"; qubits 2, 3 -> Bell pair "B" (sacrificed).
        1) Creates the two Bell pairs.
        2) 'id' gate on all qubits: represents the wait in quantum memory
           (the target of latency-driven aging).
        3) Bilateral CNOTs of the BBPSSW protocol.
        4) Measures the control qubits (sacrificed pair) in the Z basis;
           matching outcomes (00 or 11) indicate successful purification.
        """
        qc = QuantumCircuit(4, 2, name="BBPSSW")

        for a, b in [(0, 1), (2, 3)]:
            qc.h(a)
            qc.cx(a, b)
        qc.barrier()

        for q in range(4):
            qc.id(q)
        qc.barrier()

        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.barrier()

        qc.measure(2, 0)
        qc.measure(3, 1)

        return qc

    def run_purification(self, simulator: AerSimulator | None = None) -> Tuple[float, dict]:
        """
        Runs the pre-compiled BBPSSW circuit on the given simulator (or on
        the base simulator, without aging, if none is provided).
        """
        sim = simulator if simulator is not None else self.simulator
        result = sim.run(self._compiled_circuit, shots=self.shots).result()
        counts = result.get_counts()

        success_counts = counts.get("00", 0) + counts.get("11", 0)
        success_rate = success_counts / self.shots
        return success_rate, counts
