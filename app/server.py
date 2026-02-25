import json
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import numpy as np
from app.model import LogisticRegression

MODEL_PATH = "models/model.npz"
model, scaler, feature_names = LogisticRegression.load_npz(MODEL_PATH)


# ----------------------------
# Domain logic (pure functions)
# ----------------------------

def _vectorize_one(payload: dict) -> np.ndarray:
    # Enforce feature order exactly as training
    return np.array([float(payload[name]) for name in feature_names], dtype=np.float64)[None, :]


def validate_payload(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a JSON object (dict)")
    missing = [f for f in feature_names if f not in payload]
    if missing:
        raise ValueError(f"missing_features: {missing}")


def predict_single(payload: dict) -> dict:
    """
    Perform inference for a single observation.

    Parameters
    ----------
    payload : dict
        Mapping of feature name to numeric value. Must include all required
        feature names exactly as used during training.

    Returns
    -------
    dict
        Prediction output including class, probability, logit, and per-feature
        logit contributions.
    """
    validate_payload(payload)

    x = _vectorize_one(payload)
    x_s = scaler.transform(x)

    p1 = float(model.predict_proba(x_s)[0])
    pred = int(p1 >= 0.5)

    logit = float((x_s @ model.w)[0] + model.b)
    contrib = {name: float(model.w[i] * x_s[0, i]) for i, name in enumerate(feature_names)}

    return {
        "pred": pred,
        "p_class1": p1,
        "logit": logit,
        "feature_contrib_logit": contrib,
        "feature_names": feature_names,
    }


# ----------------------------
# HTTP glue (request/response)
# ----------------------------

@dataclass(frozen=True)
class Response:
    status: int
    body: dict


def ok(body: dict) -> Response:
    return Response(200, body)


def err(status: int, code: str, **extra) -> Response:
    return Response(status, {"error": code, **extra})


def parse_json_body(handler: BaseHTTPRequestHandler) -> object:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length).decode("utf-8") if length else ""
    if not raw:
        raise ValueError("empty_body")
    return json.loads(raw)


# ----------------------------
# Routing (separate from Handler)
# ----------------------------

def route(method: str, path: str, handler: BaseHTTPRequestHandler) -> Response:
    """
    Route an HTTP request to an endpoint handler.

    Keep this function tiny and explicit: method+path => response.
    """
    if method == "GET" and path == "/healthz":
        return ok({"status": "ok"})

    if method == "GET" and path == "/model":
        return ok({"model_path": MODEL_PATH, "features": feature_names})

    if method == "POST" and path == "/predict":
        payload = parse_json_body(handler)

        if not isinstance(payload, dict):
            return err(400, "bad_request", detail="payload must be an object (single record)")

        try:
            out = predict_single(payload)
            return ok(out)
        except ValueError as e:
            msg = str(e)
            if msg.startswith("missing_features:"):
                # keep a structured error
                missing = msg.split("missing_features:", 1)[1].strip()
                return err(400, "missing_features", missing=missing)
            return err(400, "bad_request", detail=msg)
        except Exception as e:
            return err(400, "bad_request", detail=str(e))

    return err(404, "not_found")


# ----------------------------
# Thin Handler (only HTTP I/O)
# ----------------------------

class Handler(BaseHTTPRequestHandler):
    """
    Thin HTTP handler: delegates routing/logic to `route` and only handles I/O.
    """

    def _send(self, resp: Response):
        raw = json.dumps(resp.body).encode("utf-8")
        self.send_response(resp.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        resp = route("GET", path, self)
        return self._send(resp)

    def do_POST(self):
        path = urlparse(self.path).path
        resp = route("POST", path, self)
        return self._send(resp)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000
    print(f"Serving on http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()
