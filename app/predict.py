import json
import numpy as np
from app.model import LogisticRegressionScratch


if __name__ == "__main__":
    model, scaler, feature_names = LogisticRegressionScratch.load_npz("model.npz")

    # Example input: {"Area": 875.0, "Perimeter": 110.1, ...}
    raw = json.loads(input().strip())

    x = np.array([float(raw[name]) for name in feature_names], dtype=np.float64)[None, :]
    x_s = scaler.transform(x)

    p1 = float(model.predict_proba(x_s)[0])
    pred = int(p1 >= 0.5)

    print(json.dumps({"p_class1": p1, "pred": pred}))

