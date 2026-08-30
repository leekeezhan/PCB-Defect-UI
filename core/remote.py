"""
remote.py
=========
Detection over HTTP: a drop-in replacement for :class:`core.detector.DefectDetector`
that forwards each image to a Module 3 inference service instead of loading the
checkpoint into this process.

Why this exists
---------------
Loading a checkpoint in-process is the simplest arrangement and stays the
default. A separate inference service is nonetheless worth supporting, for three
reasons:

* it lets the interface be deployed somewhere that cannot hold a multi-gigabyte
  PyTorch install (Streamlit Community Cloud, for instance);
* it lets the detection module be updated, restarted or moved to a GPU host
  without touching the interface;
* the assignment's *System Implementation* criterion explicitly rewards
  integration of system components such as databases and APIs.

The class deliberately mirrors ``DefectDetector``'s public surface — ``available``,
``load_error``, ``class_names``, ``model_name``, ``status()`` and ``predict()`` —
so ``app.py`` treats the two interchangeably and no page needs to know which one
it is holding.

The wire format this client speaks is specified in ``docs/API_CONTRACT.md``.
If the service is unreachable, slow or returns something unexpected, the failure
is reported through ``DetectionResult.error`` exactly as a local model failure
would be, and the interface keeps working.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import requests

from .detector import DEFAULT_CLASS_NAMES, Detection, DetectionResult
from .viz import encode_jpeg

BACKEND_REMOTE = "remote-api"

#: Default endpoint paths, appended to the configured base URL.
HEALTH_PATH = "/health"
PREDICT_PATH = "/predict"


class RemoteDetector:
    """
    HTTP client for a Module 3 inference service.

    The constructor performs one health check so the interface can report the
    service's state immediately rather than waiting for the first inspection. It
    never raises: a service that is down leaves ``available`` False and
    ``load_error`` set.

    Args:
        base_url: root URL of the service, e.g. ``http://127.0.0.1:8000``.
        api_key: optional bearer token, sent as ``Authorization: Bearer …``.
        timeout: per-request timeout in seconds.
        verify_tls: set False only for a self-signed development certificate.
        class_names: fallback class order when the service reports none.
    """

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None = None,
        timeout: float = 30.0,
        verify_tls: bool = True,
        class_names: Iterable[str] | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.timeout = float(timeout)
        self.verify_tls = bool(verify_tls)
        self.class_names: tuple[str, ...] = tuple(class_names) if class_names else DEFAULT_CLASS_NAMES

        self.backend = BACKEND_REMOTE
        self.load_error: str | None = None
        self.remote_model: str | None = None
        self.health: dict[str, Any] = {}
        self._ready = False

        if not self.base_url:
            self.load_error = "No inference service URL was configured."
        elif not self.base_url.startswith(("http://", "https://")):
            self.load_error = f"The service URL must start with http:// or https:// — got {self.base_url!r}."
        else:
            self._check_health()

    # -- session ----------------------------------------------------------- #
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _check_health(self) -> None:
        """
        Call ``GET /health`` and adopt the model name and class order it reports.

        A service that answers but omits these fields is still considered
        healthy — the client simply keeps its configured defaults.
        """
        url = f"{self.base_url}{HEALTH_PATH}"
        try:
            response = requests.get(
                url, headers=self._headers(), timeout=min(self.timeout, 10.0),
                verify=self.verify_tls,
            )
        except requests.exceptions.ConnectionError:
            self.load_error = (
                f"Could not connect to the inference service at {self.base_url}. "
                "Check that it is running and reachable from this machine."
            )
            return
        except requests.exceptions.Timeout:
            self.load_error = f"The inference service at {self.base_url} did not answer in time."
            return
        except Exception as exc:                             # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            return

        if response.status_code == 401 or response.status_code == 403:
            self.load_error = (
                f"The inference service rejected the credentials (HTTP {response.status_code}). "
                "Check the API key."
            )
            return
        if response.status_code >= 400:
            self.load_error = (
                f"The inference service answered HTTP {response.status_code} at {HEALTH_PATH}. "
                "Check that the URL points at the service root, not at an endpoint."
            )
            return

        try:
            payload = response.json()
        except ValueError:
            self.load_error = (
                f"{HEALTH_PATH} did not return JSON. The URL may be pointing at a web page "
                "rather than at the inference service."
            )
            return

        self.health = payload if isinstance(payload, dict) else {}
        self.remote_model = self.health.get("model") or self.health.get("model_name")

        classes = self.health.get("classes") or self.health.get("class_names")
        if isinstance(classes, dict) and classes:
            try:
                self.class_names = tuple(classes[k] for k in sorted(classes, key=lambda s: int(s)))
            except (ValueError, TypeError):
                pass
        elif isinstance(classes, (list, tuple)) and classes:
            self.class_names = tuple(str(name) for name in classes)

        self._ready = True

    # -- state ------------------------------------------------------------- #
    @property
    def available(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self.remote_model or (self.base_url or "remote")

    def status(self) -> dict[str, Any]:
        """Summary shown on the System status page, matching DefectDetector."""
        return {
            "available": self.available,
            "weights": self.base_url or None,
            "backend": BACKEND_REMOTE,
            "classes": list(self.class_names),
            "num_classes": len(self.class_names),
            "error": self.load_error,
            "remote_model": self.remote_model,
            "remote_health": self.health,
        }

    # -- inference --------------------------------------------------------- #
    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.25,
        iou: float = 0.45,
        max_detections: int = 300,
    ) -> DetectionResult:
        """
        Send one image to the service and parse the detections it returns.

        The image is JPEG-encoded before transmission — a 640x640 board costs
        roughly 60 kB that way, against 1.2 MB as a raw array, which matters over
        anything slower than a loopback connection.

        Args:
            image: BGR array.
            confidence: minimum score, passed to the service.
            iou: NMS IoU threshold, passed to the service.
            max_detections: cap applied locally to whatever the service returns.

        Returns:
            A :class:`core.detector.DetectionResult`, with ``error`` set when the
            call failed for any reason.
        """
        height, width = image.shape[:2]
        if not self.available:
            return DetectionResult([], 0.0, (height, width), self.model_name, self.load_error)

        encoded = encode_jpeg(image, quality=92)
        if encoded is None:
            return DetectionResult([], 0.0, (height, width), self.model_name,
                                   "The image could not be JPEG-encoded for transmission.")

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}{PREDICT_PATH}",
                headers=self._headers(),
                files={"file": ("board.jpg", encoded, "image/jpeg")},
                data={"confidence": str(confidence), "iou": str(iou)},
                timeout=self.timeout,
                verify=self.verify_tls,
            )
        except requests.exceptions.Timeout:
            return DetectionResult(
                [], 0.0, (height, width), self.model_name,
                f"The inference service did not answer within {self.timeout:.0f} s. "
                "Raise the timeout in the sidebar, or use a faster model on the service.",
            )
        except requests.exceptions.ConnectionError:
            self._ready = False
            self.load_error = f"Lost the connection to {self.base_url}."
            return DetectionResult([], 0.0, (height, width), self.model_name, self.load_error)
        except Exception as exc:                             # noqa: BLE001
            return DetectionResult([], 0.0, (height, width), self.model_name,
                                   f"{type(exc).__name__}: {exc}")

        round_trip_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code >= 400:
            return DetectionResult(
                [], 0.0, (height, width), self.model_name,
                f"The inference service answered HTTP {response.status_code}: "
                f"{response.text[:200]}",
            )

        try:
            payload = response.json()
        except ValueError:
            return DetectionResult([], 0.0, (height, width), self.model_name,
                                   "The inference service did not return JSON.")

        detections, parse_error = self._parse(payload, max_detections)
        if parse_error:
            return DetectionResult([], round_trip_ms, (height, width), self.model_name, parse_error)

        # The service reports its own inference time; the round trip is reported
        # instead when it does not, so the figure on screen is never blank.
        reported = payload.get("inference_ms")
        inference_ms = float(reported) if isinstance(reported, (int, float)) else round_trip_ms

        model = payload.get("model") or self.model_name
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionResult(detections, inference_ms, (height, width), str(model))

    def _parse(
        self, payload: Any, max_detections: int
    ) -> tuple[list[Detection], str | None]:
        """
        Convert the service's JSON into :class:`Detection` objects.

        Two bounding-box spellings are accepted, because both are natural to
        write on the server side: a ``bbox`` list of four numbers, or separate
        ``x1``/``y1``/``x2``/``y2`` keys. Boxes given as ``[x, y, w, h]`` are
        detected by an explicit ``"format": "xywh"`` field and converted.
        """
        if not isinstance(payload, dict):
            return [], "The inference service returned JSON that is not an object."

        raw = payload.get("detections")
        if raw is None:
            return [], "The response contained no 'detections' field."
        if not isinstance(raw, list):
            return [], "The 'detections' field was not a list."

        box_format = str(payload.get("format", "xyxy")).lower()
        detections: list[Detection] = []

        for item in raw[:max_detections]:
            if not isinstance(item, dict):
                continue
            box = item.get("bbox") or item.get("box")
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                a, b, c, d = (float(v) for v in box[:4])
            else:
                try:
                    a = float(item["x1"]); b = float(item["y1"])
                    c = float(item["x2"]); d = float(item["y2"])
                except (KeyError, TypeError, ValueError):
                    continue

            if box_format in ("xywh", "cxcywh"):
                if box_format == "cxcywh":
                    a, b = a - c / 2.0, b - d / 2.0
                x1, y1, x2, y2 = a, b, a + c, b + d
            else:
                x1, y1, x2, y2 = a, b, c, d

            try:
                class_id = int(item.get("class_id", item.get("cls", -1)))
            except (TypeError, ValueError):
                class_id = -1
            try:
                score = float(item.get("confidence", item.get("score", 0.0)))
            except (TypeError, ValueError):
                score = 0.0

            name = item.get("class_name") or item.get("label") or self._name_for(class_id)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=str(name),
                    confidence=score,
                    x1=min(x1, x2), y1=min(y1, y2),
                    x2=max(x1, x2), y2=max(y1, y2),
                )
            )

        return detections, None

    def _name_for(self, class_id: int) -> str:
        """Map a class index to its label, tolerating an out-of-range index."""
        if 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return f"class_{class_id}"
