import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.1, n_iters: int = 2000) -> None:
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        # X: (n, d), y: (n,) in {0,1}
        n, d = X.shape
        self.w = np.zeros(d, dtype=np.float64)
        self.b = 0.0

        for _ in range(self.n_iters):
            z = X @ self.w + self.b                 # (n,)
            p = sigmoid(z)                          # (n,)

            # Gradients (batch)
            # dL/dw = (1/n) X^T (p - y)
            # dL/db = (1/n) sum(p - y)
            err = (p - y)
            dw = (X.T @ err) / n
            db = float(err.mean())

            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return sigmoid(X @ self.w + self.b)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(X)
        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    # def save_npz(self, path: str, scaler: StandardScaler, feature_names: list[str]) -> None:
    #     assert self.w is not None
    #     assert scaler.mean_ is not None and scaler.std_ is not None
    #     np.savez_compressed(
    #         path,
    #         w=self.w,
    #         b=np.array([self.b], dtype=np.float64),
    #         mean=scaler.mean_,
    #         std=scaler.std_,
    #         feature_names=np.array(feature_names, dtype=object),
    #     )

    def save_npz(
        self,
        path: str,
        scaler: StandardScaler,
        feature_names: list[str],
        class_mapping: dict[str, int],
    ) -> None:
        assert self.w is not None
        assert scaler.mean_ is not None and scaler.std_ is not None

        # Store class_mapping inside the artifact so inference can:
        # 1) Return human-readable class labels (not just 0/1)
        # 2) Remain self-contained and reproducible even if class order changes
        #    during retraining (prevents silent label flipping)
        keys = np.array(list(class_mapping.keys()), dtype=object)
        vals = np.array(list(class_mapping.values()), dtype=np.int64)

        np.savez_compressed(
            path,
            w=self.w,
            b=np.array([self.b], dtype=np.float64),
            mean=scaler.mean_,
            std=scaler.std_,
            feature_names=np.array(feature_names, dtype=object),
            class_keys=keys,
            class_vals=vals,
        )


    @staticmethod
    def load_npz(path: str) -> tuple["LogisticRegressionScratch", StandardScaler, list[str]]:
        data = np.load(path, allow_pickle=True)
        
        model = LogisticRegressionScratch()
        model.w = data["w"].astype(np.float64)
        model.b = float(data["b"][0])

        scaler = StandardScaler()
        scaler.mean_ = data["mean"].astype(np.float64)
        scaler.std_ = data["std"].astype(np.float64)

        feature_names = [str(x) for x in data["feature_names"].tolist()]
        
        return model, scaler, feature_names
