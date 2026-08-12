import torch

from qrepeater_twin import CS_MSELoss, EdgeLSTM, InferenceTimer, StandardLSTM, train_edge_lstm


def test_edge_lstm_output_shape_and_range():
    torch.manual_seed(0)
    model = EdgeLSTM(input_size=2, hidden_size=8, num_layers=1)
    x = torch.rand(4, 20, 2)
    out = model(x)
    assert out.shape == (4, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_standard_lstm_output_shape_and_range():
    torch.manual_seed(0)
    model = StandardLSTM(input_size=2, hidden_size=32, num_layers=2, dropout=0.1)
    x = torch.rand(4, 20, 2)
    out = model(x)
    assert out.shape == (4, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_standard_lstm_single_layer_ignores_inter_layer_dropout():
    # nn.LSTM only accepts dropout > 0 when num_layers > 1; StandardLSTM
    # must silently zero it out for num_layers == 1 rather than raising.
    torch.manual_seed(0)
    model = StandardLSTM(input_size=2, hidden_size=16, num_layers=1, dropout=0.5)
    x = torch.rand(2, 20, 2)
    out = model(x)
    assert out.shape == (2, 1)


def test_standard_lstm_has_more_parameters_than_edge_lstm_by_default():
    # The whole point of StandardLSTM as the ablation's architecture
    # counterpart is that it is NOT edge-optimized (larger capacity).
    edge = EdgeLSTM(input_size=2, hidden_size=16, num_layers=1)
    standard = StandardLSTM(input_size=2, hidden_size=64, num_layers=2, dropout=0.1)
    n_params_edge = sum(p.numel() for p in edge.parameters())
    n_params_standard = sum(p.numel() for p in standard.parameters())
    assert n_params_standard > n_params_edge


def test_cs_mse_loss_penalizes_false_positive_more_than_false_negative():
    threshold = 0.65
    loss = CS_MSELoss(threshold=threshold, lambda_penalty=10.0, lambda_fn=2.0,
                       discard_penalty_weight=0.0, max_discard_rate=1.0)

    # False positive: ground truth below threshold, prediction above.
    y_true_fp = torch.tensor([[0.5]])
    y_pred_fp = torch.tensor([[0.9]])

    # False negative: ground truth above threshold, prediction below (same absolute error).
    y_true_fn = torch.tensor([[0.9]])
    y_pred_fn = torch.tensor([[0.5]])

    loss_fp = loss(y_pred_fp, y_true_fp).item()
    loss_fn = loss(y_pred_fn, y_true_fn).item()

    assert loss_fp > loss_fn


def test_train_edge_lstm_reduces_loss():
    torch.manual_seed(0)
    X = torch.rand(16, 20, 2)
    y = torch.rand(16, 1)

    model = EdgeLSTM(input_size=2, hidden_size=8, num_layers=1)
    criterion = CS_MSELoss()
    with torch.no_grad():
        loss_before = criterion(model(X), y).item()

    model = train_edge_lstm(model, X, y, epochs=50, lr=1e-2, seed=0)

    with torch.no_grad():
        loss_after = criterion(model(X), y).item()

    assert loss_after < loss_before


def test_inference_timer_cpu_returns_nonnegative_elapsed():
    device = torch.device("cpu")
    model = EdgeLSTM(input_size=2, hidden_size=8, num_layers=1)
    x = torch.rand(1, 20, 2)

    with InferenceTimer(device) as timer:
        _ = model(x)

    assert timer.elapsed_s >= 0.0
    assert timer.use_cuda_events is False
