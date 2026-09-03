"""
pipeline_remote.py
==================
Modules 1 and 2 over HTTP.

This is the counterpart of :mod:`core.remote` — that one puts Module 3 behind a
service, this one does the same for the two upstream stages:

    Module 1 : Image Acquisition & Pre-processing  (resize, normalisation, CLAHE)
    Module 2 : Image Alignment & Calibration       (board boundary, perspective warp)

Why over HTTP
-------------
Module 4 is developed in its own repository. Importing the teammates' code would
mean either vendoring a copy of it (which goes stale the moment they improve an
algorithm) or nesting this repository inside theirs (which is what having a
separate repository was meant to avoid). A service boundary removes both
problems: they own their code and deploy it, this interface owns the wire format
and calls it, and either side can be worked on without touching the other.

The wire format is specified in ``docs/API_CONTRACT_PIPELINE.md``, which also
contains a complete, runnable FastAPI reference implementation. One request per
image carries both stages, because the video and live pages call this per frame
and a second round trip per frame is a cost with nothing to show for it.

Design rules, unchanged from the local adapter this replaces:

1. **No hard failure.** A service that is down, slow or answering nonsense leaves
   the interface usable: the stage is marked as not run, a human-readable note is
   recorded, and detection continues on the best image available.
2. **Nothing is written to disk.** Images travel as JPEG bytes and stay in memory.
3. **``StageResult.final`` is never ``None``**, so the pages always have
   something to display and to detect on.
"""

from __future__ import annotations

import base64
import binascii
import time
from typing import Any

import numpy as np
import requests

from .pipeline_bridge import MODE_MODULE, StageResult, decode_image
from .viz import encode_jpeg
from .workspace import dataset_folders, workspace_root

SOURCE_REMOTE = "remote"

#: Endpoint paths, appended to the configured base URL.
HEALTH_PATH = "/health"
PROCESS_PATH = "/process"

#: JPEG quality used for the image sent upstream. The image is about to be
#: filtered and warped, so a visually lossless setting is the right trade —
#: a 512x512 board costs roughly 60 kB at this quality against 786 kB raw.
_UPLOAD_QUALITY = 95


class RemotePipeline:
    """
    HTTP client for a Modules 1 & 2 processing service.

    The public surface deliberately matches
    :class:`core.pipeline_bridge.PipelineBridge` — ``available``, ``status()``,
    ``run()``, ``preprocess()``, ``align()``, ``dataset_folders()`` — so
    ``app.py``, ``core.video`` and ``core.live`` hold either one without knowing
    which.

    The constructor performs a single health check so the System status page can
    report the service state before the first board is inspected. It never
    raises.

    Args:
        base_url: service root, e.g. ``http://127.0.0.1:8100``.
        api_key: optional bearer token, sent as ``Authorization: Bearer …``.
        timeout: per-request timeout in seconds.
        verify_tls: set False only for a self-signed development certificate.
        workspace: optional data folder, used purely for the sample pickers.
    """

    source = SOURCE_REMOTE

    def __init__(
        self,
        base_url: str | None,
        api_key: str | None = None,
        timeout: float = 30.0,
        verify_tls: bool = True,
        workspace: str | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.timeout = float(timeout)
        self.verify_tls = bool(verify_tls)
        self._workspace = workspace_root(workspace)

        self.load_error: str | None = None
        self.health: dict[str, Any] = {}
        self.service_name: str | None = None
        self._module1 = False
        self._module2 = False
        self._ready = False

        if not self.base_url:
            self.load_error = (
                "No processing service URL was configured. Enter one under "
                "**Processing pipeline** in the sidebar."
            )
        elif not self.base_url.startswith(("http://", "https://")):
            self.load_error = (
                f"The service URL must start with http:// or https:// — got {self.base_url!r}."
            )
        else:
            self._check_health()

    # -- session ------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _check_health(self) -> None:
        """
        Call ``GET /health`` and record which of the two stages the service says
        it can run.

        A service that answers ``2xx`` but omits the ``modules`` field is taken
        at its word and assumed to offer both stages — the alternative would be
        to refuse to work with a minimal implementation for no good reason.
        """
        url = f"{self.base_url}{HEALTH_PATH}"
        try:
            response = requests.get(
                url,
                headers=self._headers(),
                timeout=min(self.timeout, 10.0),
                verify=self.verify_tls,
            )
        except requests.exceptions.ConnectionError:
            self.load_error = (
                f"Could not connect to the processing service at {self.base_url}. "
                "Check that it is running and reachable from this machine."
            )
            return
        except requests.exceptions.Timeout:
            self.load_error = (
                f"The processing service at {self.base_url} did not answer in time."
            )
            return
        except Exception as exc:                                 # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            return

        if response.status_code in (401, 403):
            self.load_error = (
                f"The processing service rejected the credentials "
                f"(HTTP {response.status_code}). Check the API key."
            )
            return
        if response.status_code >= 400:
            self.load_error = (
                f"The processing service answered HTTP {response.status_code} at "
                f"{HEALTH_PATH}. Check that the URL points at the service root, "
                "not at an endpoint."
            )
            return

        try:
            payload = response.json()
        except ValueError:
            self.load_error = (
                f"{HEALTH_PATH} did not return JSON. The URL may be pointing at a "
                "web page rather than at the processing service."
            )
            return

        self.health = payload if isinstance(payload, dict) else {}
        self.service_name = self.health.get("service") or self.health.get("model")

        modules = self.health.get("modules")
        if isinstance(modules, dict):
            self._module1 = bool(modules.get("preprocess", modules.get("module1", True)))
            self._module2 = bool(modules.get("align", modules.get("module2", True)))
        else:
            self._module1 = True
            self._module2 = True

        self._ready = True

    # -- state -------------------------------------------------------------- #
    @property
    def available(self) -> bool:
        return self._ready

    @property
    def import_error(self) -> str | None:
        """Alias kept so the two adapters report failures through one name."""
        return self.load_error

    @property
    def project_root(self):
        """The data workspace, or ``None``. Used only by the sample pickers."""
        return self._workspace

    def notebook_mode_ready(self) -> bool:
        """
        Whether the service offers the teammates' original notebook functions.

        Reported by ``/health`` as ``"modes": ["module", "notebook"]``. The
        interface treats an absent field as "module only", which is the correct
        reading of a minimal implementation.
        """
        modes = self.health.get("modes")
        if isinstance(modes, (list, tuple)):
            return "notebook" in {str(m).lower() for m in modes}
        return False

    def status(self) -> dict[str, Any]:
        """Summary shown on the System status page."""
        return {
            "source": SOURCE_REMOTE,
            "available": self.available,
            "endpoint": self.base_url or None,
            "error": self.load_error,
            "module1_ready": self.available and self._module1,
            "module2_ready": self.available and self._module2,
            "notebook_mode_ready": self.notebook_mode_ready(),
            "notebook_error": None if self.notebook_mode_ready()
            else "The service did not advertise a notebook execution mode.",
            "service": self.service_name,
            "health": self.health,
            "workspace": str(self._workspace) if self._workspace else None,
        }

    # -- dataset discovery --------------------------------------------------- #
    def dataset_folders(self) -> dict[str, "Any"]:
        """Dataset folders found in the configured workspace, for the pickers."""
        return dataset_folders(self._workspace)

    # -- individual stages --------------------------------------------------- #
    def preprocess(self, image: np.ndarray) -> np.ndarray | None:
        """Run Module 1 alone. Returns ``None`` when the stage did not produce output."""
        return self.run(image, do_preprocess=True, do_align=False).preprocessed

    def align(self, image: np.ndarray) -> np.ndarray | None:
        """Run Module 2 alone. Returns ``None`` when no board boundary was found."""
        return self.run(image, do_preprocess=False, do_align=True).aligned

    # -- wired execution ------------------------------------------------------ #
    def run(
        self,
        image: np.ndarray,
        mode: str = MODE_MODULE,
        do_preprocess: bool = True,
        do_align: bool = True,
    ) -> StageResult:
        """
        Send one image to the service and unpack every stage it returns.

        Args:
            image: input image in OpenCV BGR order.
            mode: ``"module"`` or ``"notebook"``, forwarded to the service.
            do_preprocess: set False to bypass Module 1.
            do_align: set False to bypass Module 2.

        Returns:
            A :class:`core.pipeline_bridge.StageResult` whose ``final`` is never
            ``None``. Any failure is described in ``notes`` rather than raised.
        """
        result = StageResult(original=image, mode=mode)
        result.final = image

        if not do_preprocess and not do_align:
            result.notes.append("Modules 1 and 2 were both bypassed at the operator's request.")
            return result

        if not self.available:
            result.notes.append(
                f"The processing service is unavailable — {self.load_error} "
                "Detection ran on the raw image."
            )
            return result

        encoded = encode_jpeg(image, quality=_UPLOAD_QUALITY)
        if encoded is None:
            result.notes.append(
                "The image could not be JPEG-encoded for transmission — "
                "detection ran on the raw image."
            )
            return result

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}{PROCESS_PATH}",
                headers=self._headers(),
                files={"file": ("board.jpg", encoded, "image/jpeg")},
                data={
                    "preprocess": "true" if do_preprocess else "false",
                    "align": "true" if do_align else "false",
                    "mode": mode,
                },
                timeout=self.timeout,
                verify=self.verify_tls,
            )
        except requests.exceptions.Timeout:
            result.notes.append(
                f"The processing service did not answer within {self.timeout:.0f} s — "
                "detection ran on the raw image. Raise the timeout in the sidebar."
            )
            return result
        except requests.exceptions.ConnectionError:
            self._ready = False
            self.load_error = f"Lost the connection to {self.base_url}."
            result.notes.append(f"{self.load_error} Detection ran on the raw image.")
            return result
        except Exception as exc:                                 # noqa: BLE001
            result.notes.append(
                f"The processing service call failed ({type(exc).__name__}: {exc}) — "
                "detection ran on the raw image."
            )
            return result

        round_trip_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code >= 400:
            result.notes.append(
                f"The processing service answered HTTP {response.status_code}: "
                f"{response.text[:200]} Detection ran on the raw image."
            )
            return result

        try:
            payload = response.json()
        except ValueError:
            result.notes.append(
                "The processing service did not return JSON — detection ran on "
                "the raw image."
            )
            return result

        self._unpack(payload, result, do_preprocess, do_align, round_trip_ms)
        return result

    # -- response parsing ------------------------------------------------------ #
    def _unpack(
        self,
        payload: Any,
        result: StageResult,
        do_preprocess: bool,
        do_align: bool,
        round_trip_ms: float,
    ) -> None:
        """
        Fill ``result`` from the service's JSON body.

        Every field is optional except that *something* usable must come back.
        A response carrying only ``final`` is valid and common — the two
        intermediate images exist so the interface can show the operator the
        whole chain, not because the detection path needs them.
        """
        if not isinstance(payload, dict):
            result.notes.append(
                "The processing service returned JSON that is not an object — "
                "detection ran on the raw image."
            )
            return

        preprocessed = self._decode_field(payload, ("preprocessed", "preprocessed_image", "module1"))
        aligned = self._decode_field(payload, ("aligned", "aligned_image", "module2"))
        final = self._decode_field(payload, ("final", "final_image", "output", "image"))

        if do_preprocess:
            if preprocessed is not None:
                result.preprocessed = preprocessed
            else:
                result.notes.append("Module 1 returned no image — the original was kept.")
        else:
            result.notes.append("Module 1 was bypassed at the operator's request.")

        if do_align:
            if aligned is not None:
                result.aligned = aligned
            else:
                result.notes.append(
                    "Module 2 could not resolve a four-corner board boundary — "
                    "detection continued on the pre-processed image."
                )
        else:
            result.notes.append("Module 2 was bypassed at the operator's request.")

        # Prefer the service's own choice of final image; otherwise take the last
        # stage that produced one, so a minimal implementation still works.
        # Written as explicit None checks rather than ``or``, because ``or`` on a
        # numpy array raises rather than testing for emptiness.
        for candidate in (final, result.aligned, result.preprocessed, result.original):
            if candidate is not None:
                result.final = candidate
                break

        if final is None and (result.aligned is not None or result.preprocessed is not None):
            result.notes.append(
                "The service returned no 'final' image; the last successful stage "
                "was used instead."
            )
        if final is None and result.aligned is None and result.preprocessed is None:
            result.notes.append(
                "The service returned no usable image — detection ran on the raw image."
            )

        # Optional board rectangles (x, y, w, h), so detection marks can be
        # anchored to the boards even when they move between frames.
        boards = payload.get("boards")
        if isinstance(boards, (list, tuple)):
            for entry in boards:
                try:
                    bx, by, bw, bh = (int(v) for v in entry)
                except (TypeError, ValueError):
                    continue
                if bw > 0 and bh > 0:
                    result.board_boxes.append((bx, by, bw, bh))

        for note in payload.get("notes") or []:
            text = str(note).strip()
            if text:
                result.notes.append(text)

        # ``pcb_valid`` is only ever a real bool when the service ran Student 1's
        # validator — see ``local_service/serve.py``. Anything else (missing
        # field, or an older service that predates this check) leaves it
        # ``None``, meaning "unknown", never "invalid".
        pcb_valid = payload.get("pcb_valid")
        if isinstance(pcb_valid, bool):
            result.pcb_valid = pcb_valid
            message = payload.get("pcb_message")
            result.pcb_message = str(message) if message else None

        elapsed = payload.get("elapsed_ms")
        result.elapsed_ms = float(elapsed) if isinstance(elapsed, (int, float)) else round_trip_ms

        reported_mode = payload.get("mode")
        if isinstance(reported_mode, str) and reported_mode:
            result.mode = reported_mode

    @staticmethod
    def _decode_field(payload: dict[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
        """
        Decode the first present field among ``names`` from base64 into a BGR array.

        A ``data:image/jpeg;base64,…`` prefix is tolerated, because that is what a
        server written with a browser in mind tends to emit, and rejecting it
        would be a pointless obstacle.
        """
        for name in names:
            raw = payload.get(name)
            if not raw or not isinstance(raw, str):
                continue
            text = raw.split(",", 1)[1] if raw.startswith("data:") else raw
            try:
                blob = base64.b64decode(text, validate=False)
            except (binascii.Error, ValueError):
                continue
            image = decode_image(blob)
            if image is not None:
                return image
        return None
