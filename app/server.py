import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
from model import LogisticRegressionScratch

MODEL_PATH = "model.npz"

model, scaler, feature_names = LogisticRegressionScratch.load_npz(MODEL_PATH)


def predict_one(payload: dict) -> dict:
    # Enforce feature order exactly as training
    x = np.array([float(payload[name]) for name in feature_names], dtype=np.float64)[None, :]
    x_s = scaler.transform(x)

    p1 = float(model.predict_proba(x_s)[0])
    pred = int(p1 >= 0.5)

    # Lightweight explainability: per-feature contribution in logit space
    # logit = w^T x_scaled + b ; contribution_i = w_i * x_i
    logit = float((x_s @ model.w)[0] + model.b)
    contrib = {name: float(model.w[i] * x_s[0, i]) for i, name in enumerate(feature_names)}

    return {
        "pred": pred,
        "p_class1": p1,
        "logit": logit,
        "feature_contrib_logit": contrib,
        "feature_names": feature_names,
    }


# predict batch???
def predict_batch(payload: None):
    pass


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict):
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/healthz":
            return self._send(200, {"status": "ok"})
        if path == "/model":
            return self._send(200, {"model_path": MODEL_PATH, "features": feature_names})
        return self._send(404, {"error": "not_found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/predict":
            return self._send(404, {"error": "not_found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(length).decode("utf-8")
            payload = json.loads(data)

            missing = [f for f in feature_names if f not in payload]
            if missing:
                return self._send(400, {"error": "missing_features", "missing": missing})

            out = predict_one(payload)
            return self._send(200, out)
        except Exception as e:
            return self._send(400, {"error": "bad_request", "detail": str(e)})


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000
    print(f"Serving on http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()
