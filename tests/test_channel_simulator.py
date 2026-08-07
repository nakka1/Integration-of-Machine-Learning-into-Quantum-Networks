import numpy as np

from qrepeater_twin import WDMChannelSimulator


def test_generate_dataset_shape_and_bounds():
    sim = WDMChannelSimulator(n_steps=500, dt=0.01, seed=1)
    df = sim.generate_dataset()

    assert len(df) == 500
    assert set(df.columns) == {"phase_deviation", "temp_gradient", "fidelity"}
    assert (df["phase_deviation"] >= 0).all()
    assert (df["temp_gradient"] >= 0).all()
    assert (df["fidelity"] >= 0).all() and (df["fidelity"] <= 1).all()


def test_generate_dataset_is_deterministic_given_seed():
    df_a = WDMChannelSimulator(n_steps=200, seed=7).generate_dataset()
    df_b = WDMChannelSimulator(n_steps=200, seed=7).generate_dataset()
    assert np.allclose(df_a.values, df_b.values)


def test_preprocess_shapes():
    sim = WDMChannelSimulator(n_steps=500, dt=0.01, seed=1)
    df = sim.generate_dataset()
    X_train, y_train, X_test, y_test, scaler = sim.preprocess(df, window_size=20, test_size=0.2)

    n_windows = len(df) - 20
    expected_train = int(n_windows * 0.8)
    expected_test = n_windows - expected_train

    assert X_train.shape == (expected_train, 20, 2)
    assert y_train.shape == (expected_train, 1)
    assert X_test.shape == (expected_test, 20, 2)
    assert y_test.shape == (expected_test, 1)
