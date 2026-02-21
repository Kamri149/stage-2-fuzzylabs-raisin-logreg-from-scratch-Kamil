import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_URL = "http://127.0.0.1:8000"


def _http_get(path: str, timeout: float = 2.0) -> tuple[int, dict]:
    req = urllib.request.Request(SERVER_URL + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


def _http_post_raw(path: str, raw_bytes: bytes, timeout: float = 2.0) -> tuple[int, dict]:
    req = urllib.request.Request(
        SERVER_URL + path,
        data=raw_bytes,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        # Read error response body if server returns JSON
        try:
            body = e.read().decode("utf-8")
            return e.code, json.loads(body) if body else {"error": "http_error"}
        except Exception:
            return e.code, {"error": "http_error"}


def _http_post_json(path: str, payload: dict, timeout: float = 2.0) -> tuple[int, dict]:
    return _http_post_raw(path, json.dumps(payload).encode("utf-8"), timeout=timeout)


def _wait_until_healthy(deadline_s: float = 10.0) -> None:
    start = time.time()
    while time.time() - start < deadline_s:
        try:
            status, body = _http_get("/healthz", timeout=1.0)
            if status == 200 and body.get("status") == "ok":
                return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("Server did not become healthy in time")


def _start_server() -> subprocess.Popen:
    server_py = ROOT / "app/server.py"
    model_npz = ROOT / "model.npz"

    if not server_py.exists():
        raise RuntimeError(f"Missing {server_py}")
    if not model_npz.exists():
        raise RuntimeError("Missing model.npz. Run `python train.py` once to generate it.")

    # Start server as a subprocess from repo root
    proc = subprocess.Popen(
        [sys.executable, str(server_py)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="module")
def server():
    """
    Module-scoped server fixture:
      - starts server once
      - waits for readiness
      - tears down at the end
    """
    proc = _start_server()
    try:
        _wait_until_healthy(deadline_s=10.0)
        yield proc
    finally:
        _stop_server(proc)

        # If the server died unexpectedly, print logs for debugging
        if proc.stdout:
            out = proc.stdout.read()
            if out.strip():
                print("\n--- server stdout ---\n", out)
        if proc.stderr:
            err = proc.stderr.read()
            if err.strip():
                print("\n--- server stderr ---\n", err)


def _valid_payload() -> dict:
    # Values are representative; correctness is “contract” not specific output.
    return {
        "Area": 87524,
        "MajorAxisLength": 442.2460114,
        "MinorAxisLength": 253.291155,
        "Eccentricity": 0.819738392,
        "ConvexArea": 90546,
        "Extent": 0.758650579,
        "Perimeter": 1184.04,
    }


def test_healthz(server):
    status, body = _http_get("/healthz")
    assert status == 200
    assert body.get("status") == "ok"


def test_predict_happy_path(server):
    status, out = _http_post_json("/predict", _valid_payload())
    assert status == 200

    # Contract assertions
    assert "pred" in out
    assert "p_class1" in out
    assert 0.0 <= float(out["p_class1"]) <= 1.0
    assert int(out["pred"]) in (0, 1)

    # If you return explainability, validate shape
    if "feature_contrib_logit" in out:
        contrib = out["feature_contrib_logit"]
        assert isinstance(contrib, dict)
        for k in _valid_payload().keys():
            assert k in contrib


def test_predict_missing_features_returns_400(server):
    payload = _valid_payload()
    payload.pop("Area")

    status, out = _http_post_json("/predict", payload)
    assert status == 400
    # Flexible: accept any error schema, but enforce we got an error response
    assert "error" in out


def test_predict_invalid_json_returns_400(server):
    status, out = _http_post_raw("/predict", b'{"Area": 123,', timeout=2.0)
    assert status == 400
    assert "error" in out


def test_concurrent_requests(server):
    """
    Basic concurrency/load sanity:
      - multiple requests in parallel
      - all return valid outputs
    """
    payload = _valid_payload()
    n = 20
    max_workers = 8

    def call():
        return _http_post_json("/predict", payload, timeout=5.0)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(call) for _ in range(n)]
        for fut in as_completed(futures):
            results.append(fut.result())

    assert len(results) == n
    for status, out in results:
        assert status == 200
        assert 0.0 <= float(out["p_class1"]) <= 1.0
        assert int(out["pred"]) in (0, 1)