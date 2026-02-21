import numpy as np
import tempfile
import os

from app.model import LogisticRegressionScratch, StandardScaler, sigmoid


def test_sigmoid_stability():
    # Extreme values should not overflow
    large_pos = sigmoid(np.array([1000.0]))
    large_neg = sigmoid(np.array([-1000.0]))

    assert np.isfinite(large_pos).all()
    assert np.isfinite(large_neg).all()
    assert 0.0 < large_pos < 1.0
    assert 0.0 < large_neg < 1.0


def test_training_reduces_loss():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 5))
    true_w = rng.normal(size=5)
    logits = X @ true_w
    y = (logits > 0).astype(np.int64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegressionScratch(lr=0.1, n_iters=2000)

    initial_loss = model.loss(Xs, y) if model.w is not None else None
    model.fit(Xs, y)
    final_loss = model.loss(Xs, y)

    # After training loss should be finite and small
    assert np.isfinite(final_loss)
    assert final_loss < 0.7  # random baseline ~0.69


def test_predict_output_shape_and_bounds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.integers(0, 2, size=50)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegressionScratch(lr=0.1, n_iters=500)
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

    model = LogisticRegressionScratch(lr=0.1, n_iters=1000)
    model.fit(Xs, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.npz") 
        
        model.save_npz(path, scaler, ["f1", "f2", "f3", "f4"], {"a":0, "b":1}) 
 
        loaded_model, loaded_scaler, _ = LogisticRegressionScratch.load_npz(path) 
 
        original_preds = model.predict_proba(Xs) 
        loaded_preds = loaded_model.predict_proba(loaded_scaler.transform(X)) 

        assert np.allclose(original_preds, loaded_preds, atol=1e-8)