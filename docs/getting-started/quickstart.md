# Quickstart

This walks through the smallest complete example: generate the synthetic
channel, train `EdgeLSTM + CS_MSELoss`, and evaluate it with the Digital
Twin's blind-vs-intelligent admission comparison.

```python
from quantum_twin.channel_simulator import WDMChannelSimulator
from quantum_twin.cli import get_device
from quantum_twin.config import QuantumConfig, SimConfig, TrainConfig
from quantum_twin.models import EdgeLSTM, train_edge_lstm
from quantum_twin.orchestrator import DigitalTwinOrchestrator
from quantum_twin.quantum_node import QuantumRepeaterNode
from quantum_twin.reproducibility import seed_everything

device = get_device()
sim_cfg = SimConfig()
train_cfg = TrainConfig()
quantum_cfg = QuantumConfig()

# 1. Generate the synthetic WDM channel and windowed train/test tensors.
wdm_sim = WDMChannelSimulator(n_steps=sim_cfg.n_steps, dt=sim_cfg.dt, seed=sim_cfg.seed)
df = wdm_sim.generate_dataset()
X_train, y_train, X_test, y_test, _scaler = wdm_sim.preprocess(
    df, window_size=sim_cfg.window_size, test_size=sim_cfg.test_size,
)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# 2. Train EdgeLSTM with the cost-sensitive loss.
seed_everything(42)
model = EdgeLSTM(input_size=2, hidden_size=train_cfg.hidden_size).to(device)
model = train_edge_lstm(
    model, X_train, y_train, threshold=train_cfg.threshold,
    lambda_penalty=10.0, lambda_fn=train_cfg.lambda_fn,
    discard_penalty_weight=train_cfg.discard_penalty_weight,
    max_discard_rate=train_cfg.max_discard_rate,
    epochs=train_cfg.epochs, lr=train_cfg.lr, device=device, seed=42,
)

# 3. Evaluate: does the predictive controller beat blind/unconditional purification?
quantum_node = QuantumRepeaterNode(T1=quantum_cfg.T1, T2=quantum_cfg.T2,
                                    depol_prob=quantum_cfg.depol_prob,
                                    shots=quantum_cfg.shots, seed=quantum_cfg.seed)
orchestrator = DigitalTwinOrchestrator(model=model, quantum_node=quantum_node,
                                         threshold=train_cfg.threshold, device=device)
metrics = orchestrator.run_intelligent(X_test, y_test)

baseline_node = QuantumRepeaterNode(T1=quantum_cfg.T1, T2=quantum_cfg.T2,
                                     depol_prob=quantum_cfg.depol_prob,
                                     shots=quantum_cfg.shots, seed=quantum_cfg.seed)
baseline_orchestrator = DigitalTwinOrchestrator(model=None, quantum_node=baseline_node,
                                                  threshold=train_cfg.threshold, device=device)
baseline_metrics = baseline_orchestrator.run_blind_baseline(X_test, y_test)

print(f"Intelligent: {metrics['useful_pairs']} useful pairs from {metrics['attempted']} attempts")
print(f"Blind:       {baseline_metrics['useful_pairs']} useful pairs from {baseline_metrics['attempted']} attempts")
```

## Running full experiments

The full multi-seed Pareto sweep, cross-architecture comparison, ablation
study, and walk-forward validation are each one function call away (see
[Running experiments](../guides/experiments.md)) or one script away:

```bash
python experiments/run_pareto_sweep.py --epochs 150
python experiments/run_model_comparison.py --epochs 150
python experiments/run_ablation.py --epochs 150
python experiments/run_walk_forward.py --n-splits 5
```

Or, for the CLI-driven all-in-one entry point:

```bash
quantum-twin --epochs 150 --compare-baselines --run-ablation
```
