import numpy as np
import tempfile
import os

from app.model import LogisticRegression, StandardScaler, sigmoid


def test_sigmoid_stability():
    # Extreme values should not overflow
    large_pos = sigmoid(np.array([1000000.0]))
    large_neg = sigmoid(np.array([-1000000.0]))
    assert np.isfinite(large_pos).all()
    assert np.isfinite(large_neg).all()
    assert 0.0 < large_pos < 1.0
    assert 0.0 < large_neg < 1.0


def test_training_reduces_loss():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 5))
    true_w = rng.normal(size=5)
    y = ((X @ true_w) > 0).astype(np.int64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Initial model: same init, no optimisation steps
    model0 = LogisticRegression(lr=0.1, n_iters=0).fit(Xs, y)
    initial_loss = model0.loss(Xs, y)

    # Trained model
    model = LogisticRegression(lr=0.1, n_iters=2000).fit(Xs, y)
    final_loss = model.loss(Xs, y)

    assert np.isfinite(initial_loss)
    assert np.isfinite(final_loss)
    assert final_loss < initial_loss
    assert final_loss < 0.7


def test_predict_output_shape_and_bounds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.integers(0, 2, size=50)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(lr=0.1, n_iters=500)
    model.fit(Xs, y)

    probs = model.predict_proba(Xs)

    assert probs.shape == (50,)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    preds = model.predict(Xs)
    assert preds.shape == (50,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_save_and_load_consistency():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 4))
    y = rng.integers(0, 2, size=100)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(Xs, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.npz") 
        
        model.save_npz(path, scaler, ["f1", "f2", "f3", "f4"], {"a":0, "b":1}) 
 
        loaded_model, loaded_scaler, _ = LogisticRegression.load_npz(path) 
 
        original_preds = model.predict_proba(Xs) 
        loaded_preds = loaded_model.predict_proba(loaded_scaler.transform(X)) 

        assert np.allclose(original_preds, loaded_preds, atol=1e-8)