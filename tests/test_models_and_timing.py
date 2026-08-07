import torch

from qrepeater_twin import CS_MSELoss, EdgeLSTM, InferenceTimer, train_edge_lstm


def test_edge_lstm_output_shape_and_range():
    torch.manual_seed(0)
    model = EdgeLSTM(input_size=2, hidden_size=8, num_layers=1)
    x = torch.rand(4, 20, 2)
    out = model(x)
    assert out.shape == (4, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


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
