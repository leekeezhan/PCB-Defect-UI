"""
local_service/serve.py
=======================
A real, runnable stand-in for the Modules 1, 2 & 3 service — so Module 4 can be
developed and demoed today, without waiting on a teammate's deployment.

WHAT THIS IS
------------
It implements the two contracts this repository already defines:

    docs/API_CONTRACT_PIPELINE.md   GET /health, POST /process   (Modules 1 & 2)
    docs/API_CONTRACT.md            GET /health, POST /predict   (Module 3)

...but instead of a teammate's server, it runs the actual code already sitting
on this machine:

    Modules 1 & 2   ``image_pipeline.preprocess_image`` / ``align_image``,
                     imported straight from a checkout of the shared
                     Image-Processing repository — read only, nothing in that
                     repository is modified or committed to by running this.
    Module 3        this repository's own ``core.detector.DefectDetector``,
                     loading a checkpoint exactly as the sidebar's
                     "Local checkpoint" option would.

WHAT THIS IS NOT
----------------
Not a replacement for a teammate's service, and not part of Module 4's own
graded deliverable — it is a development convenience. Once a teammate stands up
the real thing, point the sidebar at their URL instead and stop running this.
It is safe to delete at any time; nothing else in this repository imports it.

Detection quality depends entirely on which checkpoint is configured below. If
Module 3 has not yet delivered a trained checkpoint, the only ``.pt`` files
available are the stock, pre-trained-on-COCO downloads Ultralytics ships
(``yolov10n.pt``, ``rtdetr-l.pt``) — this server will load one of those without
complaint, but it will detect COCO objects ("person", "car", ...), not PCB
defects. That is expected, not a bug: it proves the wire format end-to-end
while the real weights are still pending. Swap ``WEIGHTS`` below for
``Student3-Defect Detection/runs/<run>/weights/best.pt`` the moment it exists.

RUNNING IT
----------
From the repository root, with the shared repository checked out beside it
(the default layout ``core.workspace`` already looks for)::

    pip install fastapi uvicorn python-multipart ultralytics torch
    uvicorn local_service.serve:app --host 127.0.0.1 --port 8000

Then in the interface sidebar, point BOTH fields at the same URL — this one
process answers both contracts:

    Processing pipeline -> Remote API -> http://127.0.0.1:8000
    Detection model      -> Remote API -> http://127.0.0.1:8000

Configuration is by environment variable, all optional:

    PCB_WORKSPACE   folder holding image_pipeline.py (a checkout of the shared
                    repository). Defaults to the same resolution
                    core/workspace.py uses — a sibling ``Image-Processing``
                    folder is found automatically.
    PCB_WEIGHTS     path to a .pt/.pth checkpoint for Module 3. Defaults to
                    whatever core.detector.discover_weights() finds under the
                    workspace, which currently means a stock checkpoint — see
                    the warning above.
    PCB_DEVICE      "cpu" or "cuda". Defaults to automatic.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

# Make ``core`` importable regardless of the working directory this is
# launched from, matching the trick app.py itself uses.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.detector import DefectDetector, discover_weights, is_pretrained_stock  # noqa: E402
from core.workspace import workspace_root  # noqa: E402

WORKSPACE = workspace_root(os.environ.get("PCB_WORKSPACE"))
DEVICE = os.environ.get("PCB_DEVICE") or None

_configured_weights = os.environ.get("PCB_WEIGHTS")
if _configured_weights:
    WEIGHTS: Path | None = Path(_configured_weights)
else:
    _candidates = discover_weights(WORKSPACE)
    WEIGHTS = _candidates[0] if _candidates else None

_STOCK_WARNING = (
    f"{WEIGHTS.name} is a stock pre-trained download, not a checkpoint trained "
    "on the PCB dataset — it will report COCO object classes, not defect types. "
    "Set PCB_WEIGHTS once Module 3 delivers a trained best.pt."
    if WEIGHTS is not None and is_pretrained_stock(WEIGHTS) else None
)

_CONTRACT = None
_CONTRACT_ERROR: str | None = None
if WORKSPACE is not None and (WORKSPACE / "image_pipeline.py").is_file():
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    try:
        import image_pipeline as _CONTRACT                          # noqa: N812
    except Exception as exc:                                        # noqa: BLE001
        _CONTRACT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _CONTRACT_ERROR = (
        "image_pipeline.py was not found under the workspace "
        f"({WORKSPACE or 'not resolved'}). Set PCB_WORKSPACE to the shared "
        "repository's folder."
    )

_detector = DefectDetector(str(WEIGHTS) if WEIGHTS else None, device=DEVICE)

print(f"[local_service] workspace         : {WORKSPACE}")
print(f"[local_service] image_pipeline.py  : "
      f"{'loaded' if _CONTRACT else f'unavailable — {_CONTRACT_ERROR}'}")
print(f"[local_service] weights            : {WEIGHTS or 'none found'}")
print(f"[local_service] detector           : "
      f"{'loaded — ' + _detector.backend if _detector.available else _detector.load_error}")
if _STOCK_WARNING:
    print(f"[local_service] WARNING            : {_STOCK_WARNING}")

app = FastAPI(title="PCB inspection — local stand-in for Modules 1, 2 & 3")


def _decode_upload(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _encode(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else None


# --------------------------------------------------------------------------- #
# GET /health — read by both clients; each reads the half it cares about.
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        # -- read by core/pipeline_remote.py -------------------------------- #
        "service": "local-stand-in-modules-1-2-3",
        "modules": {"preprocess": _CONTRACT is not None, "align": _CONTRACT is not None},
        "modes": ["module"] if _CONTRACT is not None else [],
        "workspace": str(WORKSPACE) if WORKSPACE else None,
        "pipeline_error": _CONTRACT_ERROR,
        # -- read by core/remote.py ------------------------------------------ #
        "model": _detector.model_name if _detector.available else None,
        "classes": list(_detector.class_names),
        "device": _detector.status().get("device") or DEVICE or "auto",
        "detector_error": _detector.load_error,
        "stock_weights_warning": _STOCK_WARNING,
    }


# --------------------------------------------------------------------------- #
# POST /process — Modules 1 & 2, per docs/API_CONTRACT_PIPELINE.md
# --------------------------------------------------------------------------- #
@app.post("/process")
async def process(
    file: UploadFile = File(...),
    preprocess: str = Form("true"),
    align: str = Form("true"),
    mode: str = Form("module"),
) -> dict[str, Any]:
    original = _decode_upload(await file.read())
    if original is None:
        return {"notes": ["The uploaded image could not be decoded."]}

    if _CONTRACT is None:
        return {"notes": [f"image_pipeline.py is unavailable — {_CONTRACT_ERROR}"]}

    started = time.perf_counter()
    notes: list[str] = []
    working = original
    pre_image: np.ndarray | None = None
    aligned_image: np.ndarray | None = None

    if preprocess.lower() == "true":
        pre_image = _CONTRACT.preprocess_image(original)
        if pre_image is None:
            notes.append("Module 1 returned nothing; the original image was kept.")
        else:
            working = pre_image

    if align.lower() == "true":
        aligned_image = _CONTRACT.align_image(working)
        if aligned_image is None:
            notes.append("No four-corner board boundary was found.")
        else:
            working = aligned_image

    return {
        "preprocessed": _encode(pre_image),
        "aligned": _encode(aligned_image),
        "final": _encode(working),
        "notes": notes,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "mode": "module",
    }


# --------------------------------------------------------------------------- #
# POST /predict — Module 3, per docs/API_CONTRACT.md
# --------------------------------------------------------------------------- #
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
) -> dict[str, Any]:
    image = _decode_upload(await file.read())
    if image is None:
        return {"detections": [], "model": _detector.model_name}

    result = _detector.predict(image, confidence=confidence, iou=iou)
    return {
        "model": result.model_name,
        "inference_ms": result.inference_ms,
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox": [d.x1, d.y1, d.x2, d.y2],
            }
            for d in result.detections
        ],
    }
