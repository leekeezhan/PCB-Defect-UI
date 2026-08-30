# Modules 1 & 2 processing service — API contract

**Audience:** whoever implements the pre-processing and alignment API.
**Client:** `core/pipeline_remote.py`, selected from the sidebar as
*Processing pipeline → Remote API*.

Module 4 lives in its own repository, so it reaches Modules 1 and 2 the same way
it reaches Module 3: over HTTP. You keep your notebooks and your algorithms in
your own repository, wrap them in the service below, and the interface drives
them without either side importing the other's code.

Two endpoints. That is the whole contract.

---

## `GET /health`

Called once when the interface connects, so it can display the service state and
know which stages you offer.

**Request:** no body. `Authorization: Bearer <key>` is sent when a key is
configured in the sidebar.

**Response:** `200 OK`, JSON.

```json
{
  "status": "ok",
  "service": "modules-1-2",
  "modules": { "preprocess": true, "align": true },
  "modes": ["module", "notebook"]
}
```

| Field | Required | Notes |
|---|---|---|
| `status` | no | Ignored by the client; any `2xx` counts as healthy. |
| `service` | no | Displayed on the System status page. |
| `modules` | no | Which stages you implement. Omitting it means "both", which is the right reading of a minimal implementation. |
| `modes` | no | Execution modes you honour. Include `"notebook"` only if you really run the original notebook cells; the interface reports the mode as unavailable otherwise, which is accurate rather than a fault. |

Anything `4xx`/`5xx`, a connection failure, or a non-JSON body leaves the
interface in its "pipeline unavailable" state, with the reason shown on screen
and detection continuing on the raw image. `401`/`403` produce a specific
"check the API key" message.

---

## `POST /process`

Called once per inspected image — per board on the single and batch pages, per
sampled frame on the video and live pages.

Both stages travel in **one** request. The video page calls this per frame; a
second round trip per frame would double the latency for nothing.

**Request:** `multipart/form-data`

| Part | Type | Notes |
|---|---|---|
| `file` | file | The board as the operator supplied it, JPEG-encoded at quality 95, filename `board.jpg`, type `image/jpeg`. **This is the raw input** — nothing has been done to it yet. |
| `preprocess` | form field (string) | `"true"` or `"false"`. When `"false"`, skip Module 1 and run alignment on the image as received. |
| `align` | form field (string) | `"true"` or `"false"`. When `"false"`, skip Module 2. |
| `mode` | form field (string) | `"module"` or `"notebook"`. Honour it if you can; ignoring it is acceptable, and reporting what you actually did in the response's `mode` field is better still. |

**Response:** `200 OK`, JSON. Images are **base64-encoded JPEG or PNG**.

```json
{
  "preprocessed": "<base64 image>",
  "aligned": "<base64 image>",
  "final": "<base64 image>",
  "notes": ["No four-corner boundary found; returned the pre-processed image."],
  "elapsed_ms": 18.4,
  "mode": "module"
}
```

| Field | Required | Notes |
|---|---|---|
| `final` | **strongly recommended** | The image Module 3 should detect on. When absent the client falls back to `aligned`, then `preprocessed`, then the original — so a service that returns only the two stages still works, but say what you mean. |
| `preprocessed` | no | Module 1's output. `null` when the stage was skipped or produced nothing. Also accepted as `preprocessed_image` or `module1`. |
| `aligned` | no | Module 2's output. `null` is the correct answer when no board boundary was found — that is a normal outcome, not an error. Also accepted as `aligned_image` or `module2`. |
| `notes` | no | Strings shown to the operator alongside the result and printed in the PDF report. This is where "CLAHE clip limit raised for a dark capture" or "board boundary found at 3 corners only" belongs. |
| `elapsed_ms` | no | Your own timing. The client reports the round trip when it is absent. |
| `mode` | no | What you actually ran. |

A `data:image/jpeg;base64,…` prefix on any image field is accepted and stripped.

**The two intermediate images are for the operator, not for the pipeline.** The
interface shows the full chain — original, pre-processed, aligned, detected — on
the single-board page, which is how a marker sees that four modules really ran.
Returning only `final` is valid; the display then has less to show.

Errors: return a non-`2xx` with a short plain-text or JSON body. The interface
shows the status code and the first 200 characters, notes that the board was
inspected without pre-processing, and carries on with the rest of the batch.

---

## Reference implementation

Complete and runnable. Import your own two functions at the top and start.

```python
# serve_pipeline.py  —  pip install fastapi uvicorn python-multipart opencv-python numpy
import base64
import io
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

# --- your code -------------------------------------------------------------- #
# Both take a BGR numpy array and return one. align_image returns None when it
# cannot find a four-corner board boundary — that is a valid answer.
from image_pipeline import align_image, preprocess_image

app = FastAPI(title="PCB pipeline service — Modules 1 & 2")


def _encode(image):
    """BGR array -> base64 JPEG, or None."""
    if image is None:
        return None
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return base64.b64encode(buffer.tobytes()).decode("ascii") if ok else None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "modules-1-2",
        "modules": {"preprocess": True, "align": True},
        "modes": ["module"],
    }


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    preprocess: str = Form("true"),
    align: str = Form("true"),
    mode: str = Form("module"),
):
    data = np.frombuffer(await file.read(), np.uint8)
    original = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if original is None:
        return {"notes": ["The uploaded image could not be decoded."]}

    started = time.perf_counter()
    notes = []
    working = original
    preprocessed = aligned = None

    if preprocess.lower() == "true":
        preprocessed = preprocess_image(original)          # Module 1
        if preprocessed is None:
            notes.append("Module 1 returned nothing; the original image was kept.")
        else:
            working = preprocessed

    if align.lower() == "true":
        aligned = align_image(working)                     # Module 2
        if aligned is None:
            notes.append("No four-corner board boundary was found.")
        else:
            working = aligned

    return {
        "preprocessed": _encode(preprocessed),
        "aligned": _encode(aligned),
        "final": _encode(working),
        "notes": notes,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "mode": "module",
    }
```

```bash
uvicorn serve_pipeline:app --host 0.0.0.0 --port 8100
```

Then in the interface sidebar: **Processing pipeline → Remote API**, base URL
`http://127.0.0.1:8100`. The System status page confirms the connection and
lists what the service reported.

---

## Notes for the implementer

**Colour order.** The image arrives as JPEG bytes; `cv2.imdecode` gives you BGR,
which is what OpenCV code expects and what the client decodes back into. If your
notebook works in RGB, convert on both edges of your own code and leave the wire
format alone.

**Return `null`, not an error, for a stage that had nothing to say.** An
unalignable board is an ordinary outcome — a photo taken at an angle the
homography cannot resolve. The interface displays that as a degradation with the
note you supplied and inspects the pre-processed image instead. A `500` in that
situation loses the board entirely.

**Size.** Whatever dimensions your modules produce are what the interface shows
and what Module 3 receives. If Module 1 resizes to 512x512, the boxes Module 3
returns will be in 512x512 coordinates, which is consistent and correct — the
two services do not need to agree on a size, only to hand each other whole
images.

**One service or two.** Modules 1, 2 and 3 live in one repository, so serving
all three from a single FastAPI app is the least work — one file, one process,
one port. The interface supports that with no changes: put the same base URL in
both sidebar fields and the two clients call `/process` and `/predict` on the
same host.

The only thing to get right is `/health`, which both clients call. Return the
union of the two contracts' fields and each client reads the half it cares
about, ignoring the rest:

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        # read by the pipeline client
        "service": "modules-1-2-3",
        "modules": {"preprocess": True, "align": True},
        "modes": ["module"],
        # read by the detection client
        "model": "rtdetr-l-best",
        "classes": CLASSES,
        "device": "cuda",
    }
```

Two separate services still work and are worth splitting out when detection
wants a GPU host that the pre-processing does not, or when the two are being
worked on by different people who would rather not restart each other's process.
`8100` for this service and `8000` for detection is then the convention, so
nobody has to guess which sidebar field takes which URL.

**Timeouts.** The client's timeout is configurable in the sidebar and defaults to
30 s. A batch of 100 boards makes 100 sequential requests, and the video page
calls once per sampled frame, so anything expensive — loading a reference
template, building a SIFT index — belongs at service start-up rather than inside
the request handler.

**Authentication.** If a bearer token is required, read the `Authorization`
header and reject with `401`. Leave it open for a local demonstration.

**CORS is not needed.** The client is Python, not a browser.
