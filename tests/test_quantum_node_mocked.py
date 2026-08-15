"""
Unit tests for `quantum_twin.quantum_node.QuantumRepeaterNode` that isolate
its NOISE-MODEL COMPOSITION and LATENCY-DECAY math from an actual Qiskit
Aer circuit execution, per the project's testing policy: "you don't need
to run a full Aer simulation to verify the fidelity-decay math is
correct" -- `AerSimulator`, `NoiseModel`, `transpile`, and the noise-model
factory functions `quantum_node.py` imports are replaced with
`unittest.mock` doubles, so these tests exercise ONLY this project's own
logic (the T1/T2 physical constraint, the latency -> thermal-relaxation-
channel wiring, the run_purification success-rate arithmetic, and the
"build the circuit once" optimization), at a fraction of a real Aer run's
cost and with zero dependency on Aer's actual noise semantics.

`mocked_quantum_node_deps` patches the NAMES `quantum_node.py` imported
into its own module namespace (`quantum_twin.quantum_node.AerSimulator`,
etc.) -- the standard `unittest.mock.patch.object` pattern -- rather than
faking the `qiskit`/`qiskit_aer` packages themselves, so these tests run
correctly whether or not a real Qiskit installation is present. Every
mock is exposed by NAME in a dict (not by positional decorator-argument
order): stacking five bare `@patch.object` decorators and relying on
their injected-argument order is a well-known footgun -- the order mocks
arrive as positional arguments is the REVERSE of the decorators' visual
top-to-bottom order, and a single typo'd parameter ordering silently
binds the wrong mock to the wrong name with no error, only a confusing
assertion failure. A fixture yielding a `{name: Mock}` dict sidesteps
that class of bug entirely.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from quantum_twin import quantum_node as qn


@pytest.fixture
def mocked_quantum_node_deps():
    """Patches every external (Qiskit) collaborator `quantum_node.py`
    imports, for the duration of one test, and returns them as a
    `{name: MagicMock}` dict. `depolarizing_error`/`thermal_relaxation_error`'s
    return values are pre-wired so `.compose(...)`/`.tensor(...)` (which
    `_build_noise_model` calls on them) return fresh mocks rather than
    relying on `MagicMock`'s auto-speccing alone.
    """
    with ExitStack() as stack:
        mocks = {
            "transpile": stack.enter_context(
                patch.object(qn, "transpile", side_effect=lambda circuit, sim: circuit)
            ),
            "AerSimulator": stack.enter_context(patch.object(qn, "AerSimulator")),
            "NoiseModel": stack.enter_context(patch.object(qn, "NoiseModel")),
            "depolarizing_error": stack.enter_context(patch.object(qn, "depolarizing_error")),
            "thermal_relaxation_error": stack.enter_context(patch.object(qn, "thermal_relaxation_error")),
        }
        mocks["depolarizing_error"].return_value.compose.return_value = MagicMock()
        mocks["thermal_relaxation_error"].return_value.compose.return_value = MagicMock()
        mocks["thermal_relaxation_error"].return_value.tensor.return_value = MagicMock()

        # `_build_noise_model` constructs a NEW `NoiseModel()` on every
        # call (see that method's docstring); track every instance
        # produced so tests can inspect calls across ALL of them, not
        # just `NoiseModel.return_value` (which only ever reflects the
        # LAST one built).
        instances: list = []

        def _noise_model_side_effect(*_args, **_kwargs):
            instance = MagicMock()
            instances.append(instance)
            return instance

        mocks["NoiseModel"].side_effect = _noise_model_side_effect
        mocks["noise_model_instances"] = instances

        yield mocks


def _make_result(counts: dict) -> MagicMock:
    """A mock Aer `Result`-like object whose `.get_counts()` returns `counts`."""
    result = MagicMock()
    result.get_counts.return_value = counts
    return result


def _id_gate_registrations(noise_model_instances: list) -> list:
    """Every `add_all_qubit_quantum_error(error, ["id"])` call recorded
    across ALL `NoiseModel()` instances constructed so far -- the one and
    only call site where an aging/latency error is registered, so its
    presence (or absence) is the ground truth for "was an aging channel
    added?"."""
    registrations = []
    for instance in noise_model_instances:
        for call in instance.add_all_qubit_quantum_error.call_args_list:
            args, _kwargs = call
            if len(args) >= 2 and args[1] == ["id"]:
                registrations.append(call)
    return registrations


# ---------------------------------------------------------------------------
# T1/T2 physical constraint
# ---------------------------------------------------------------------------

def test_t2_greater_than_2_t1_raises(mocked_quantum_node_deps) -> None:
    with pytest.raises(AssertionError):
        qn.QuantumRepeaterNode(T1=10e-6, T2=25e-6)  # T2 > 2*T1: physically invalid


def test_t2_exactly_2_t1_is_accepted(mocked_quantum_node_deps) -> None:
    node = qn.QuantumRepeaterNode(T1=10e-6, T2=20e-6)  # boundary: T2 == 2*T1
    assert node.T1 == 10e-6 and node.T2 == 20e-6


# ---------------------------------------------------------------------------
# apply_latency_decay: does latency correctly gate the aging channel?
# ---------------------------------------------------------------------------

def test_zero_latency_registers_no_id_gate_error(mocked_quantum_node_deps) -> None:
    node = qn.QuantumRepeaterNode(T1=50e-6, T2=30e-6)
    node.apply_latency_decay(0.0)

    assert _id_gate_registrations(mocked_quantum_node_deps["noise_model_instances"]) == []


def test_positive_latency_registers_an_id_gate_error(mocked_quantum_node_deps) -> None:
    node = qn.QuantumRepeaterNode(T1=50e-6, T2=30e-6)
    node.apply_latency_decay(1e-5)

    assert len(_id_gate_registrations(mocked_quantum_node_deps["noise_model_instances"])) == 1


def test_negative_latency_is_clamped_like_zero(mocked_quantum_node_deps) -> None:
    node = qn.QuantumRepeaterNode(T1=50e-6, T2=30e-6)
    node.apply_latency_decay(-5e-6)  # negative latency: must behave like 0.0

    assert _id_gate_registrations(mocked_quantum_node_deps["noise_model_instances"]) == []


def test_aging_error_built_from_this_node_t1_t2_and_the_given_latency(mocked_quantum_node_deps) -> None:
    node = qn.QuantumRepeaterNode(T1=50e-6, T2=30e-6)
    mock_thermal = mocked_quantum_node_deps["thermal_relaxation_error"]
    mock_thermal.reset_mock()  # ignore the two fixed-gate calls made during __init__

    node.apply_latency_decay(7.5e-6)

    # Among every thermal_relaxation_error(...) call apply_latency_decay
    # triggers (the two rebuilt fixed-gate terms plus the new aging
    # term), exactly one must carry THIS node's own T1/T2 and the
    # requested latency -- the aging channel itself.
    aging_calls = [call for call in mock_thermal.call_args_list if call.args[:3] == (50e-6, 30e-6, 7.5e-6)]
    assert len(aging_calls) == 1


# ---------------------------------------------------------------------------
# run_purification: success-rate arithmetic
# ---------------------------------------------------------------------------

def test_run_purification_success_rate_matches_counts(mocked_quantum_node_deps) -> None:
    node = qn.QuantumRepeaterNode(T1=50e-6, T2=30e-6, shots=524)

    mock_sim_instance = mocked_quantum_node_deps["AerSimulator"].return_value
    mock_sim_instance.run.return_value.result.return_value = _make_result(
        {"00": 300, "11": 212, "01": 8, "10": 4}
    )

    success_rate, counts = node.run_purification()

    # success_rate = (counts["00"] + counts["11"]) / shots -- this
    # project's own arithmetic, verified independently of whatever counts
    # a real Aer run would have produced.
    assert success_rate == pytest.approx((300 + 212) / 524)
    assert counts == {"00": 300, "11": 212, "01": 8, "10": 4}


def test_run_purification_zero_success_counts(mocked_quantum_node_deps) -> None:
    node = qn.QuantumRepeaterNode(T1=50e-6, T2=30e-6, shots=512)

    mock_sim_instance = mocked_quantum_node_deps["AerSimulator"].return_value
    mock_sim_instance.run.return_value.result.return_value = _make_result({"01": 512})

    success_rate, _counts = node.run_purification()

    assert success_rate == pytest.approx(0.0)


def test_run_purification_uses_provided_simulator_over_base(mocked_quantum_node_deps) -> None:
    """`run_purification(simulator=aged_sim)` must dispatch to `aged_sim`,
    NOT to `self.simulator` (the un-aged base simulator built at
    construction time) -- this is exactly the mechanism
    `DigitalTwinOrchestrator.run_intelligent` relies on to apply the
    latency-dependent aging channel per admitted cycle."""
    node = qn.QuantumRepeaterNode(T1=50e-6, T2=30e-6, shots=100)

    base_sim = node.simulator
    base_sim.run.return_value.result.return_value = _make_result({"00": 999})  # should NOT be used

    aged_sim = MagicMock()
    aged_sim.run.return_value.result.return_value = _make_result({"00": 50, "11": 50})

    success_rate, _counts = node.run_purification(simulator=aged_sim)

    assert success_rate == pytest.approx(100 / 100)
    aged_sim.run.assert_called_once()
    base_sim.run.assert_not_called()


# ---------------------------------------------------------------------------
# The circuit is built/transpiled exactly ONCE per node instance
# ---------------------------------------------------------------------------

def test_transpile_called_exactly_once_regardless_of_purification_calls(mocked_quantum_node_deps) -> None:
    """The BBPSSW circuit's structure never changes between calls (only
    the noise model does, via `apply_latency_decay`) -- `transpile` must
    therefore run exactly once at construction, never again, even after
    many `apply_latency_decay` + `run_purification` cycles. This is the
    exact optimization documented in `QuantumRepeaterNode.__init__`'s
    docstring (avoiding thousands of redundant recompilations across a
    Pareto sweep)."""
    node = qn.QuantumRepeaterNode(T1=50e-6, T2=30e-6, shots=10)
    mock_transpile = mocked_quantum_node_deps["transpile"]
    assert mock_transpile.call_count == 1

    for latency in (0.0, 1e-6, 2e-6, 0.0, 5e-6):
        aged_sim = node.apply_latency_decay(latency)
        aged_sim.run.return_value.result.return_value = _make_result({"00": 5, "11": 5})
        node.run_purification(simulator=aged_sim)

    assert mock_transpile.call_count == 1  # still exactly one, after 5 full cycles
