"""
KServe Model Inference Adapter
================================
Adapts kserve/kserve into the HFT execution arm as a production
model serving layer for all ML signal engines.

Source repo: /home/user/kserve
Key SDK:     python/kserve/  (KServe Python SDK)

KServe role in the platform:
  When any ML model (FinRL agent, NVIDIA TFT, Stock-Prediction-Models)
  is promoted to production, it can be deployed as a KServe InferenceService
  on Kubernetes. The KServeAdapter provides a unified client that:
    1. Routes inference requests to the correct endpoint (REST v2 protocol)
    2. Falls back to the local in-process model if no endpoint is configured
    3. Supports batch inference for DailyUniverseScanner scoring
    4. Provides health-check and latency monitoring

KServe InferenceService protocol (v2):
  POST /v2/models/{model_name}/infer
  {
    "inputs": [{"name": "input-0", "shape": [1, N], "datatype": "FP32",
                "data": [...features...]}]
  }

Architecture:
  KServeAdapter
    ├── register(name, endpoint_url)  — register a deployed model endpoint
    ├── infer(name, features)          — call endpoint or local fallback
    ├── health(name)                   — check liveness
    └── batch_infer(name, batch)       — batch inference for scanner

Supported model names (matching deployed InferenceServices):
  "stock_prediction_es"  — Stock-Prediction-Models Evolution Strategy
  "finrl_ppo"            — FinRL PPO agent
  "nvidia_tft"           — NVIDIA Temporal Fusion Transformer
  "deep_trading_vol"     — Deep-Trading volatility LSTM

Usage
-----
adapter = KServeAdapter()
adapter.register("nvidia_tft", "http://nvidia-tft.kserve.svc.cluster.local/v2")
result = adapter.infer("nvidia_tft", features=[0.1, 0.2, ...])
"""

import os
import json
import time
import threading
import numpy as np
from datetime import datetime
from typing import Optional, Callable


# ===========================================================================
# KServe v2 REST Protocol helpers
# ===========================================================================

def _build_v2_payload(features: list, model_name: str,
                      input_name: str = "input-0") -> dict:
    """Build a KServe v2 inference request payload."""
    flat = np.array(features).flatten().tolist()
    return {
        "model_name": model_name,
        "inputs": [{
            "name":     input_name,
            "shape":    [1, len(flat)],
            "datatype": "FP32",
            "data":     flat,
        }]
    }


def _parse_v2_response(resp: dict) -> Optional[list]:
    """Extract output data from a KServe v2 response."""
    try:
        return resp["outputs"][0]["data"]
    except (KeyError, IndexError):
        return None


# ===========================================================================
# ModelEndpoint — per-model endpoint config + latency tracking
# ===========================================================================

class _ModelEndpoint:
    def __init__(self, name: str, url: str,
                 local_fallback: Optional[Callable] = None):
        self.name            = name
        self.url             = url
        self.local_fallback  = local_fallback
        self.healthy         = True
        self.last_latency_ms = 0.0
        self.call_count      = 0
        self.error_count     = 0
        self._lock           = threading.Lock()


# ===========================================================================
# KServeAdapter — Public API
# ===========================================================================

class KServeAdapter:
    """
    Unified inference client for KServe-deployed models.

    In development (no Kubernetes cluster): all calls fall back to
    local_fallback functions — same behaviour, zero infra required.

    In production: register endpoint URLs and the adapter routes
    inference over HTTP to the deployed InferenceService pods.

    Timeout / retry policy (mirrors KServe production recommendations):
      - Timeout:       100ms per request (HFT budget)
      - Max retries:   1 (fail fast, use local fallback on 2nd error)
      - Circuit breaker: opens after 3 consecutive errors
    """

    TIMEOUT_MS      = 100    # Hard timeout per inference call
    CIRCUIT_OPEN_N  = 3      # Open circuit after N consecutive errors

    def __init__(self):
        self._endpoints: dict[str, _ModelEndpoint] = {}
        self._circuit_errors: dict[str, int] = {}

        # Try importing requests (optional — not required for fallback mode)
        try:
            import requests as _req
            self._requests = _req
        except ImportError:
            self._requests = None

    # -----------------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------------

    def register(self, model_name: str, endpoint_url: str,
                 local_fallback: Optional[Callable] = None) -> None:
        """
        Register a KServe InferenceService endpoint.

        Parameters
        ----------
        model_name    : str  Logical model name (e.g. "nvidia_tft")
        endpoint_url  : str  KServe v2 base URL
                             e.g. "http://nvidia-tft.kserve-ns.svc/v2"
        local_fallback: callable(features) → list  Used when endpoint unreachable
        """
        ep = _ModelEndpoint(model_name, endpoint_url, local_fallback)
        self._endpoints[model_name] = ep
        self._circuit_errors[model_name] = 0
        print(f"[KServe] Registered: {model_name} → {endpoint_url}")

    def register_local(self, model_name: str,
                       local_fallback: Callable) -> None:
        """Register a local-only model (no remote endpoint)."""
        self.register(model_name, "local://", local_fallback)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    def infer(self, model_name: str, features: list,
              use_fallback_on_error: bool = True) -> Optional[list]:
        """
        Run inference. Returns output list or None on failure.

        Routing logic:
          1. If circuit open → local fallback immediately
          2. If URL is "local://" → local fallback
          3. Otherwise → HTTP POST to KServe endpoint
          4. On error → increment circuit counter → fallback if enabled
        """
        ep = self._endpoints.get(model_name)
        if ep is None:
            return None

        # Circuit breaker check
        if self._circuit_errors.get(model_name, 0) >= self.CIRCUIT_OPEN_N:
            return self._run_fallback(ep, features)

        # Local-only model
        if ep.url == "local://":
            return self._run_fallback(ep, features)

        # Remote KServe call
        if self._requests is None:
            return self._run_fallback(ep, features)

        try:
            t0 = time.monotonic()
            payload = _build_v2_payload(features, model_name)
            resp = self._requests.post(
                f"{ep.url}/models/{model_name}/infer",
                json=payload,
                timeout=self.TIMEOUT_MS / 1000,
            )
            latency = (time.monotonic() - t0) * 1000

            with ep._lock:
                ep.last_latency_ms = round(latency, 2)
                ep.call_count      += 1

            if resp.status_code == 200:
                self._circuit_errors[model_name] = 0   # reset circuit
                return _parse_v2_response(resp.json())
            else:
                return self._handle_error(ep, model_name, features,
                                          use_fallback_on_error)
        except Exception as exc:
            return self._handle_error(ep, model_name, features,
                                      use_fallback_on_error, exc)

    def batch_infer(self, model_name: str,
                    batch: list) -> list:
        """
        Batch inference: list of feature vectors → list of outputs.
        Used by DailyUniverseScanner for scoring multiple tickers.
        """
        return [self.infer(model_name, f) or [] for f in batch]

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------

    def health(self, model_name: str) -> dict:
        """Check liveness of a registered endpoint."""
        ep = self._endpoints.get(model_name)
        if ep is None:
            return {"model": model_name, "status": "NOT_REGISTERED"}

        if ep.url == "local://":
            return {"model": model_name, "status": "LOCAL", "latency_ms": 0}

        if self._requests is None:
            return {"model": model_name, "status": "NO_REQUESTS_LIB"}

        try:
            r = self._requests.get(
                f"{ep.url}/v2/health/live",
                timeout=0.5,
            )
            status = "HEALTHY" if r.status_code == 200 else f"HTTP_{r.status_code}"
        except Exception as e:
            status = f"UNREACHABLE ({type(e).__name__})"

        return {
            "model":         model_name,
            "status":        status,
            "url":           ep.url,
            "latency_ms":    ep.last_latency_ms,
            "calls":         ep.call_count,
            "errors":        ep.error_count,
            "circuit_errors": self._circuit_errors.get(model_name, 0),
        }

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "source":     "kserve/kserve",
            "models":     list(self._endpoints.keys()),
            "health":     {n: self.health(n) for n in self._endpoints},
            "timestamp":  datetime.now().isoformat(),
        }

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _run_fallback(self, ep: _ModelEndpoint, features: list) -> Optional[list]:
        if ep.local_fallback is not None:
            try:
                return ep.local_fallback(features)
            except Exception:
                pass
        return None

    def _handle_error(self, ep: _ModelEndpoint, model_name: str,
                      features: list, use_fallback: bool,
                      exc: Optional[Exception] = None) -> Optional[list]:
        with ep._lock:
            ep.error_count += 1
        self._circuit_errors[model_name] = self._circuit_errors.get(model_name, 0) + 1

        if self._circuit_errors[model_name] == self.CIRCUIT_OPEN_N:
            print(f"[KServe] ⚡ Circuit opened for {model_name} "
                  f"after {self.CIRCUIT_OPEN_N} errors")

        if use_fallback:
            return self._run_fallback(ep, features)
        return None
