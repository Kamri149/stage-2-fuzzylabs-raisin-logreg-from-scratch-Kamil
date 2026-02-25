import numpy as np
from pathlib import Path


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the logistic sigmoid function in a numerically stable way.

    The sigmoid maps real-valued inputs to probabilities in (0, 1):

        sigmoid(z) = 1 / (1 + exp(-z))

    This implementation clips the input values to avoid numerical
    overflow or underflow when computing exp(-z).

    Parameters
    ----------
    z : np.ndarray
        Input array (logits), typically w^T x + b.

    Returns
    -------
    np.ndarray
        Element-wise sigmoid probabilities in the range (0, 1).
    """

    # Clip logits to prevent overflow in exp(-z).
    # Values beyond ±35 are already effectively saturated.
    z = np.clip(z, -35.0, 35.0)

    # Apply logistic transformation
    return 1.0 / (1.0 + np.exp(-z))


class StandardScaler:
    """
    Feature standardisation utility.

    Transforms features to have zero mean and unit variance:

        X_scaled = (X - mean) / std

    This improves optimisation stability and convergence speed
    for gradient-based models such as logistic regression.

    Notes
    -----
    - Mean and standard deviation are computed per feature (column-wise).
    - Statistics must be computed on training data only to prevent data leakage.
    - Zero-variance features are handled by replacing std=0 with std=1.
    """

    def __init__(self) -> None:
        # Per-feature mean and standard deviation
        # Stored after fitting
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Compute per-feature mean and standard deviation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.

        Returns
        -------
        StandardScaler
            Fitted scaler instance.
        """

        # Compute column-wise mean and standard deviation
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        # Prevent division by zero for constant features
        # If a feature has zero variance, scaling would fail.
        # Replacing 0 with 1 leaves the feature unchanged after centering.
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardisation using previously computed statistics.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Standardised feature matrix.
        """

        # Ensure scaler has been fitted
        assert self.mean_ is not None and self.std_ is not None, \
            "StandardScaler must be fitted before calling transform()."

        # Apply feature-wise standardisation
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Convenience method: fit scaler and immediately transform data.

        Equivalent to:
            scaler.fit(X)
            scaler.transform(X)

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix.

        Returns
        -------
        np.ndarray
            Standardised feature matrix.
        """
        return self.fit(X).transform(X)


class LogisticRegression:
    """
    Binary Logistic Regression implemented from first principles.

    This model estimates:

        p(y=1 | x) = sigmoid(w^T x + b)

    using batch gradient descent to minimise binary cross-entropy loss.

    Design principles:
    - Deterministic training
    - Numerical stability
    - Explicit state storage
    - Minimal dependencies (NumPy only)
    """

    def __init__(self, lr: float = 0.1, n_iters: int = 2000) -> None:
        """
        Initialise model hyperparameters.

        Parameters
        ----------
        lr : float
            Learning rate for gradient descent.
        n_iters : int
            Number of optimisation iterations.
        """
        self.lr = float(lr)
        self.n_iters = int(n_iters)

        # Model parameters (initialised during fit)
        self.w: np.ndarray | None = None  # weight vector (d,)
        self.b: float = 0.0               # bias term


    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train logistic regression using batch gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix (typically scaled).
        y : np.ndarray of shape (n_samples,)
            Binary target vector in {0, 1}.

        Returns
        -------
        LogisticRegression
            Fitted model instance.
        """

        # X: (n, d)
        # y: (n,)
        n, d = X.shape

        # Initialise parameters
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0

        for _ in range(self.n_iters):

            # Linear combination (logits)
            z = X @ self.w + self.b  # shape: (n,)

            # Convert logits to probabilities
            p = sigmoid(z)           # shape: (n,)

            # Compute prediction error 
            # L=−n1​i=1∑n​[yi​logpi​+(1−yi​)log(1−pi​)]
            err = (p - y)

            # Gradient of binary cross-entropy:
            # dL/dw = (1/n) X^T (p - y)
            # dL/db = (1/n) sum(p - y)
            dw = (X.T @ err) / n
            db = float(err.mean())

            # Gradient descent update
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predicted probabilities for class 1.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Probability vector in range (0, 1).
        """
        assert self.w is not None, "Model must be fitted before prediction."
        return sigmoid(X @ self.w + self.b)


    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert probabilities into binary class predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        threshold : float
            Classification threshold.

        Returns
        -------
        np.ndarray
            Binary predictions in {0, 1}.
        """
        return (self.predict_proba(X) >= threshold).astype(np.int64)


    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Binary target vector.

        Returns
        -------
        float
            Mean binary cross-entropy.
        """

        p = self.predict_proba(X)

        # Clip probabilities to prevent log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)

        # Binary cross-entropy
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


    def save_npz(
        self,
        path: str,
        scaler: StandardScaler,
        feature_names: list[str],
        class_mapping: dict[str, int],
    ) -> None:
        """
        Save model, preprocessing state, and label mapping to a compressed NumPy archive.

        Ensures:
        - Fully self-contained inference artifact
        - No dependency on training code
        - Protection against silent label-order changes
        - Safe directory creation if path does not exist
        """

        assert self.w is not None, "Model must be fitted before saving."
        assert scaler.mean_ is not None and scaler.std_ is not None, \
            "Scaler must be fitted before saving."

        # Ensure parent directory exists
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Store mapping as arrays (avoids nested objects / pickle)
        keys = np.array(list(class_mapping.keys()), dtype=object)
        vals = np.array(list(class_mapping.values()), dtype=np.int64)

        # Save primitives only (no pickled objects)
        np.savez_compressed(
            path_obj,
            w=self.w,
            b=np.array([self.b], dtype=np.float64),
            mean=scaler.mean_,
            std=scaler.std_,
            feature_names=np.array(feature_names, dtype=object),
            class_keys=keys,
            class_vals=vals,
        )


    @staticmethod
    def load_npz(path: str) -> tuple["LogisticRegression", StandardScaler, list[str]]:
        """
        Load model and preprocessing state from a saved artifact.

        Parameters
        ----------
        path : str
            Path to .npz file.

        Returns
        -------
        tuple
            (model, scaler, feature_names)
        """

        data = np.load(path, allow_pickle=True)

        # Reconstruct model
        model = LogisticRegression()
        model.w = data["w"].astype(np.float64)
        model.b = float(data["b"][0])

        # Reconstruct scaler
        scaler = StandardScaler()
        scaler.mean_ = data["mean"].astype(np.float64)
        scaler.std_ = data["std"].astype(np.float64)

        feature_names = [str(x) for x in data["feature_names"].tolist()]

        return model, scaler, feature_names
