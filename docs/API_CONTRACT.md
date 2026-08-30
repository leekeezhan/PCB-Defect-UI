# Module 3 inference service — API contract

**Audience:** whoever implements the detection API.
**Client:** `Student4-UI/core/remote.py`, selected from the sidebar as
*Detector source → Remote API*.

The interface already works by loading the checkpoint in-process, so this
service is an addition rather than a prerequisite. Implement the two endpoints
below and the interface will drive the service instead, with no change to the
interface code.

Two endpoints. That is the whole contract.

---

## `GET /health`

Called once when the interface connects, so it can display the service state and
adopt the model's class order.

**Request:** no body. `Authorization: Bearer <key>` is sent when a key is
configured in the sidebar.

**Response:** `200 OK`, JSON.

```json
{
  "status": "ok",
  "model": "rtdetr-l-best",
  "classes": ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"],
  "device": "cuda"
}
```

| Field | Required | Notes |
|---|---|---|
| `status` | no | Ignored by the client; any `2xx` counts as healthy. |
| `model` | no | Displayed in the interface and printed in the PDF report. |
| `classes` | no | Class order. Accepts a list, or an object keyed by index (`{"0": "missing_hole", …}`). If omitted, the client keeps its own six-class order. |
| `device` | no | Informational. |

Anything `4xx`/`5xx`, a connection failure, or a non-JSON body leaves the
interface in its "detector unavailable" state with the reason shown on screen.
`401`/`403` produce a specific "check the API key" message.

---

## `POST /predict`

Called once per inspected image — per board on the single and batch pages, per
sampled frame on the video and live pages.

**Request:** `multipart/form-data`

| Part | Type | Notes |
|---|---|---|
| `file` | file | The image, JPEG-encoded, filename `board.jpg`, type `image/jpeg`. This is the image **after** Modules 1 and 2 have run — do not pre-process it again. |
| `confidence` | form field (string) | Minimum score, e.g. `"0.25"`. |
| `iou` | form field (string) | NMS IoU threshold, e.g. `"0.45"`. |

**Response:** `200 OK`, JSON.

```json
{
  "model": "rtdetr-l-best",
  "inference_ms": 41.2,
  "detections": [
    { "class_id": 0, "class_name": "missing_hole", "confidence": 0.91, "bbox": [100, 120, 128, 148] },
    { "class_id": 2, "class_name": "open_circuit", "confidence": 0.77, "bbox": [320, 260, 352, 292] }
  ]
}
```

| Field | Required | Notes |
|---|---|---|
| `detections` | **yes** | A list. An empty list means a clean board — that is a valid, successful response. |
| `inference_ms` | no | The service's own timing. When omitted the client reports the round-trip time instead. |
| `model` | no | Overrides the name shown for this result. |
| `format` | no | `"xyxy"` (default), `"xywh"`, or `"cxcywh"`. |

Per detection:

| Field | Required | Notes |
|---|---|---|
| `bbox` | **yes*** | Four numbers, **absolute pixels** in the submitted image. Also accepted as `box`, or as separate `x1`/`y1`/`x2`/`y2` fields. |
| `confidence` | **yes** | `0.0`–`1.0`. Also accepted as `score`. |
| `class_id` | recommended | Integer index. Also accepted as `cls`. |
| `class_name` | recommended | Defect name. Also accepted as `label`. Falls back to the class order from `/health` when absent. |

\* Either `bbox`/`box`, or the four separate coordinate fields.

**Coordinates must be absolute pixels, not normalised.** YOLO's native output is
normalised `cxcywh`; multiply by the image dimensions before returning, or set
`"format": "xywh"` after scaling. This is the single most likely thing to go
wrong, and it shows up as boxes clustered in the top-left corner.

Errors: return a non-`2xx` with a short plain-text or JSON body. The interface
shows the status code and the first 200 characters, marks that image as
un-inspected, and carries on with the rest of the batch.

---

## Reference implementation

Complete and runnable. Drop the trained checkpoint beside it and start.

```python
# serve.py  —  pip install fastapi uvicorn python-multipart ultralytics
import io, time
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from ultralytics import YOLO

WEIGHTS = "runs/detect/train/weights/best.pt"

app = FastAPI(title="PCB defect detection service")
model = YOLO(WEIGHTS)
CLASSES = [model.names[i] for i in sorted(model.names)]


@app.get("/health")
def health():
    return {"status": "ok", "model": WEIGHTS.split("/")[-1], "classes": CLASSES}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
):
    image = np.array(Image.open(io.BytesIO(await file.read())).convert("RGB"))[:, :, ::-1]

    started = time.perf_counter()
    results = model.predict(image, conf=confidence, iou=iou, verbose=False)
    elapsed_ms = (time.perf_counter() - started) * 1000

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box, score, cls in zip(
            result.boxes.xyxy.cpu().numpy(),      # absolute pixels — what the client expects
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            detections.append({
                "class_id": int(cls),
                "class_name": CLASSES[int(cls)] if int(cls) < len(CLASSES) else f"class_{cls}",
                "confidence": float(score),
                "bbox": [float(v) for v in box],
            })

    return {"model": WEIGHTS.split("/")[-1], "inference_ms": elapsed_ms, "detections": detections}
```

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
```

Then in the interface sidebar: **Detector source → Remote API**, base URL
`http://127.0.0.1:8000`. The System status page confirms the connection and
lists the class order the service reported.

---

## Notes for the implementer

**One service or two.** Modules 1, 2 and 3 live in one repository, so this
endpoint can sit in the same FastAPI app as the `/process` endpoint of
[`API_CONTRACT_PIPELINE.md`](API_CONTRACT_PIPELINE.md) — one file, one process,
one port. The interface supports that with no changes: the same base URL goes
into both sidebar fields. The one thing to get right is `/health`, which both
clients call; return the union of the two contracts' fields and each client
reads the half it cares about. That file's *One service or two* note has the
details.

**Authentication.** If a bearer token is required, read the `Authorization`
header and reject with `401`. Leave it open for a local demonstration.

**CORS is not needed.** The client is Python, not a browser.

**Model choice.** The measured trade-off on this dataset (`Student3-Defect
Detection/results/comparison_summary.md`, CPU):

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS |
|---|---|---|---|---|---|
| YOLOv10n | 0.566 | 0.249 | 0.565 | 0.527 | 19.2 |
| RT-DETR-L | 0.669 | 0.293 | 0.701 | 0.637 | 2.1 |

RT-DETR-L is more accurate on both axes and is the right default for
single-board and batch inspection. YOLOv10n is roughly nine times faster and is
the right choice for the video and live pages. Serving both — say at
`/predict?model=fast` or on two ports — lets the interface use each where it
belongs; the interface's own sidebar already supports exactly this split for
locally loaded checkpoints.

**Timeouts.** The client's timeout is configurable in the sidebar and defaults
to 30 s. A batch of 100 boards makes 100 sequential requests, so cold-start
loading belongs at service start-up, not inside the request handler.
