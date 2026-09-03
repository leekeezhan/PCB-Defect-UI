"""
pipeline_bridge.py
==================
Adapter between the user interface (Module 4) and the upstream image-processing
modules:

    Module 1 : Image Acquisition & Pre-processing  (bicubic resize, CLAHE, ...)
    Module 2 : Image Alignment & Calibration       (SIFT / perspective homography)

Module 4 lives in its own repository and normally reaches those two modules over
HTTP — that client is :class:`core.pipeline_remote.RemotePipeline`, and the wire
format is specified in ``docs/API_CONTRACT_PIPELINE.md``. This file holds three
things:

* :class:`StageResult`, the container both adapters fill and every page reads;
* :class:`PipelineBridge`, the *local* adapter, kept for the case where a
  checkout of the shared repository happens to be on the same machine and the
  operator would rather not start a service to demonstrate the interface;
* :func:`create_pipeline`, the factory the application calls, which returns one
  or the other according to the sidebar.

The local adapter no longer requires ``image_pipeline.py`` to be findable. It
looks for it in the configured workspace and reports its absence as an ordinary
state, because in this repository its absence is the normal case.

Design rules
------------
1. **No hard failure.** Every upstream call is guarded. If a stage is missing or
   returns ``None``, the adapter degrades to the best result it has and records
   a human-readable note instead of raising. The interface stays usable and
   reports the degradation to the operator.
2. **Nothing is written to disk.** All stages exchange numpy arrays in memory.
3. **Two execution modes.** ``module`` calls the plain functions exported by
   ``image_pipeline``; ``notebook`` calls the teammates' original notebook
   functions. The former is faster and more robust for a live demonstration, the
   latter proves the interface really drives the teammates' own code. Both names
   are forwarded to the service when the pipeline runs remotely.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .workspace import dataset_folders as _dataset_folders
from .workspace import repo_root, workspace_root

# Marker file that identifies a checkout of the shared project.
_CONTRACT_FILENAME = "image_pipeline.py"

#: Execution modes offered in the sidebar.
MODE_MODULE = "module"
MODE_NOTEBOOK = "notebook"
MODES = (MODE_MODULE, MODE_NOTEBOOK)

#: Where Modules 1 and 2 are executed.
SOURCE_LOCAL = "local"
SOURCE_REMOTE = "remote"

#: Mean absolute pixel difference (0-255) below which Module 2's output is
#: treated as geometrically identical to its input — see
#: :attr:`StageResult.align_effective`. Measured separation on the conveyor
#: footage is ~3 for a no-op and ~30-50 for a real perspective correction, so
#: the threshold sits well clear of both.
_ALIGN_NOOP_TOLERANCE = 8.0


# --------------------------------------------------------------------------- #
# Locating and importing the shared contract
# --------------------------------------------------------------------------- #
def find_contract_root(workspace: str | os.PathLike | None = None) -> Path | None:
    """
    Locate a directory holding ``image_pipeline.py``.

    Three places are tried, in order: the configured workspace, a sibling
    checkout of the shared repository, and the parents of this file (which
    covers the historical layout where Module 4 sat inside the shared
    repository).

    Args:
        workspace: the data folder configured in the sidebar, if any.

    Returns:
        The directory holding the contract file, or ``None`` when there is none
        — the ordinary case for a standalone checkout of this repository.
    """
    candidates: list[Path] = []
    resolved = workspace_root(workspace)
    if resolved is not None:
        candidates.append(resolved)

    current = repo_root()
    for _ in range(8):
        candidates.append(current)
        if current.parent == current:      # filesystem root reached
            break
        current = current.parent

    for candidate in candidates:
        if (candidate / _CONTRACT_FILENAME).is_file():
            return candidate
    return None


#: Kept under its original name so older call sites keep working.
find_project_root = find_contract_root


def _import_contract(workspace: str | os.PathLike | None = None) -> tuple[Any | None, str | None]:
    """
    Import ``image_pipeline`` from a checkout of the shared repository.

    Returns:
        ``(module, None)`` on success, or ``(None, error_message)`` on failure.
    """
    root = find_contract_root(workspace)
    if root is None:
        return None, (
            f"{_CONTRACT_FILENAME} was not found. Local execution needs a checkout "
            "of the shared repository — set its folder as the workspace in the "
            "sidebar, or switch the processing pipeline to Remote API."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("image_pipeline"), None
    except Exception as exc:               # noqa: BLE001 - report, never crash
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class StageResult:
    """
    Output of one pass through Modules 1 and 2.

    Attributes:
        original: the image as supplied by the operator (BGR).
        preprocessed: output of Module 1, or ``None`` if the stage was skipped.
        aligned: output of Module 2, or ``None`` if alignment could not resolve
            a board boundary.
        final: the image handed on to Module 3. This is the aligned image when
            alignment succeeded, otherwise the preprocessed image, otherwise the
            original — the interface always has something to detect on.
        notes: human-readable notes describing any degradation that occurred.
        mode: which execution mode produced this result.
        elapsed_ms: how long Modules 1 and 2 took. Set by the remote adapter —
            the service's own timing when it reports one, the round trip
            otherwise — and left at zero by the local adapter.
    """

    original: np.ndarray
    preprocessed: np.ndarray | None = None
    aligned: np.ndarray | None = None
    final: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)
    mode: str = MODE_MODULE
    elapsed_ms: float = 0.0
    #: Board rectangles ``(x, y, w, h)`` detected in this frame, so detection
    #: marks can be anchored to the boards (the boards move along the conveyor).
    board_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def preprocess_ok(self) -> bool:
        return self.preprocessed is not None

    @property
    def align_ok(self) -> bool:
        return self.aligned is not None

    @property
    def align_effective(self) -> bool:
        """
        True when Module 2's output is geometrically *different* from its input.

        Module 2 can report success and still hand back what is only a rescaled
        copy of what it was given. Its corner detector takes the largest
        contour in the frame and reduces it to four points; when the strongest
        rectangle in the picture is already square to the camera — the pale
        carrier card the boards sit on in the conveyor footage, for instance,
        rather than the tilted board resting on it — the four corners it finds
        are that rectangle's, and warping an upright rectangle to an upright
        rectangle changes nothing but the resolution.

        ``align_ok`` cannot tell the two apart: both are a non-``None``
        ``aligned``. This compares the two arrays instead, so the interface can
        say honestly that a board was rescaled rather than straightened.
        """
        if self.aligned is None:
            return False
        source = self.preprocessed if self.preprocessed is not None else self.original
        if source is None:
            return False

        import cv2

        height, width = self.aligned.shape[:2]
        rescaled = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
        if rescaled.shape != self.aligned.shape:
            return True
        difference = np.mean(
            np.abs(rescaled.astype(np.int16) - self.aligned.astype(np.int16))
        )
        return float(difference) > _ALIGN_NOOP_TOLERANCE


# --------------------------------------------------------------------------- #
# The bridge
# --------------------------------------------------------------------------- #
class PipelineBridge:
    """
    Thin, fault-tolerant facade over a *local* checkout of Modules 1 and 2.

    A single instance is created once per Streamlit session and cached, so the
    (relatively slow) notebook parsing performed by ``image_pipeline`` happens at
    most once.

    Args:
        workspace: folder to look in for ``image_pipeline.py`` and for the
            datasets offered by the sample pickers. ``None`` falls back to the
            resolution order described in :mod:`core.workspace`.
    """

    source = SOURCE_LOCAL

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace_root(workspace)
        self._contract_root = find_contract_root(workspace)
        self._module, self._import_error = _import_contract(workspace)
        self._notebook_error: str | None = None
        self._notebook_fns: dict[str, Callable] | None = None

    # -- availability ------------------------------------------------------ #
    @property
    def available(self) -> bool:
        """True when ``image_pipeline`` was imported successfully."""
        return self._module is not None

    @property
    def import_error(self) -> str | None:
        return self._import_error

    @property
    def load_error(self) -> str | None:
        """Alias matching :class:`core.pipeline_remote.RemotePipeline`."""
        return self._import_error

    @property
    def project_root(self) -> Path | None:
        """The data workspace — where the sample pickers look for images."""
        return self._workspace

    def status(self) -> dict[str, Any]:
        """
        Summary used by the 'System status' page.

        The keys are the same ones :class:`core.pipeline_remote.RemotePipeline`
        reports, so the status page renders either adapter without branching.
        """
        return {
            "source": SOURCE_LOCAL,
            "available": self.available,
            "endpoint": str(self._contract_root / _CONTRACT_FILENAME)
            if self._contract_root else None,
            "error": self._import_error,
            "module1_ready": self._has("preprocess_image"),
            "module2_ready": self._has("align_image"),
            "notebook_mode_ready": self.notebook_mode_ready(),
            "notebook_error": self._notebook_error,
            "service": None,
            "health": {},
            "workspace": str(self._workspace) if self._workspace else None,
        }

    def _has(self, name: str) -> bool:
        return self._module is not None and callable(getattr(self._module, name, None))

    def notebook_mode_ready(self) -> bool:
        """
        True when the teammates' original notebook functions can be loaded.

        The check is performed lazily and its outcome cached, because parsing the
        Module 1 notebook is expensive.
        """
        if not self._has("load_student1_functions"):
            self._notebook_error = "image_pipeline.load_student1_functions is unavailable."
            return False
        if self._notebook_fns is not None:
            return True
        if self._notebook_error is not None:
            return False
        try:
            self._notebook_fns = self._module.load_student1_functions()
            return True
        except Exception as exc:           # noqa: BLE001
            self._notebook_error = f"{type(exc).__name__}: {exc}"
            return False

    # -- individual stages ------------------------------------------------- #
    def preprocess(self, image: np.ndarray) -> np.ndarray | None:
        """Run Module 1 on a BGR array. Returns ``None`` if unavailable."""
        if not self._has("preprocess_image"):
            return None
        try:
            return self._module.preprocess_image(image)
        except Exception:                  # noqa: BLE001
            return None

    def align(self, image: np.ndarray) -> np.ndarray | None:
        """Run Module 2 on a BGR array. Returns ``None`` if no board was found."""
        if not self._has("align_image"):
            return None
        try:
            return self._module.align_image(image)
        except Exception:                  # noqa: BLE001
            return None

    def find_boards(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Locate every PCB board in a frame — multi-board conveyor frames
        included. Returns ``(x, y, w, h)`` boxes, left to right, and an empty
        list when the module is unavailable or no board is present.

        Used by the video and live pages so each board is straightened in place
        instead of aligning the whole multi-board frame.
        """
        if not self._has("find_boards"):
            return []
        try:
            return list(self._module.find_boards(image))
        except Exception:                  # noqa: BLE001
            return []

    def rectify_frame(self, image: np.ndarray) -> np.ndarray | None:
        """
        Rectify EVERY board of a multi-board frame IN PLACE (Module 2): each
        board is warped onto the axis-aligned rectangle that bounds it, so the
        frame keeps its original layout with every board at its correct angle.

        Returns ``None`` when the module is unavailable or the frame has zero
        or one board — the caller then uses :meth:`align` as usual.
        """
        if not self._has("rectify_frame"):
            return None
        try:
            rectified, num_boards, _notes = self._module.rectify_frame(image)
            return rectified if num_boards > 1 else None
        except Exception:                  # noqa: BLE001
            return None

    def _scale_to_align_target(self, image: np.ndarray) -> np.ndarray:
        """
        Resize a whole frame to the same longest-side target a successful
        Module 2 alignment would produce (``image_pipeline.ALIGN_TARGET``).

        Used when alignment fails to resolve a boundary: without this, the
        un-aligned frame reaches the detector at its own native resolution,
        which can be a very different scale from what the detector was
        trained on. Mirrors the fallback branch of
        ``image_pipeline.detection_input()``, the function Module 3 documents
        as the one true source of detector input for both training and
        inference.
        """
        import cv2

        target = getattr(self._module, "ALIGN_TARGET", 1280) if self._module else 1280
        height, width = image.shape[:2]
        scale = target / max(width, height)
        out_w, out_h = int(round(width * scale)), int(round(height * scale))
        return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # -- wired execution --------------------------------------------------- #
    def run(
        self,
        image: np.ndarray,
        mode: str = MODE_MODULE,
        do_preprocess: bool = True,
        do_align: bool = True,
    ) -> StageResult:
        """
        Execute Modules 1 and 2 over ``image`` and return every intermediate
        result so the interface can display the full chain.

        Args:
            image: input image in OpenCV BGR order.
            mode: ``"module"`` to call ``image_pipeline``'s own functions,
                ``"notebook"`` to call the teammates' notebook functions.
            do_preprocess: set False to bypass Module 1 (useful when the operator
                uploads an image already taken from ``Preprocessed_Dataset``).
            do_align: set False to bypass Module 2.

        Returns:
            A :class:`StageResult`. ``result.final`` is never ``None``.
        """
        result = StageResult(original=image, mode=mode)

        if not self.available:
            result.notes.append(
                "Modules 1 and 2 are unavailable — detection ran on the raw image."
            )
            result.final = image
            return result

        if mode == MODE_NOTEBOOK and do_preprocess and do_align:
            aligned = self._run_notebook(image)
            if aligned is not None:
                result.aligned = aligned
                result.final = aligned
                result.notes.append(
                    "Modules 1 and 2 executed through the teammates' original notebook functions."
                )
                return result
            result.notes.append(
                "Notebook mode failed; fell back to the shared module functions."
            )

        # -- Module 1 ------------------------------------------------------ #
        working = image
        if do_preprocess:
            preprocessed = self.preprocess(image)
            if preprocessed is None:
                result.notes.append("Module 1 returned no output — the original image was kept.")
            else:
                result.preprocessed = preprocessed
                working = preprocessed
        else:
            result.notes.append("Module 1 was bypassed at the operator's request.")

        # -- Module 2 ------------------------------------------------------ #
        if do_align:
            aligned = self.align(working)
            if aligned is None:
                result.notes.append(
                    "Module 2 could not resolve a four-corner board boundary — "
                    "the frame was scaled to the detector's expected size "
                    "instead of being aligned."
                )
                working = self._scale_to_align_target(working)
            else:
                result.aligned = aligned
                working = aligned
        else:
            result.notes.append("Module 2 was bypassed at the operator's request.")

        result.final = working
        return result

    def _run_notebook(self, image: np.ndarray) -> np.ndarray | None:
        """Call ``student2_from_student1_array`` (teammates' notebook functions)."""
        if not self._has("student2_from_student1_array"):
            self._notebook_error = "image_pipeline.student2_from_student1_array is unavailable."
            return None
        if not self.notebook_mode_ready():
            return None
        try:
            return self._module.student2_from_student1_array(image)
        except Exception as exc:           # noqa: BLE001
            self._notebook_error = f"{type(exc).__name__}: {exc}"
            return None

    # -- dataset discovery -------------------------------------------------- #
    def dataset_folders(self) -> dict[str, Path]:
        """
        Locate the dataset folders produced by Modules 1 and 2, so the batch page
        can offer them as a source without the operator typing a path.

        Returns:
            Mapping of label to existing directory. Missing folders are omitted.
        """
        return _dataset_folders(self._workspace)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def create_pipeline(
    source: str = SOURCE_REMOTE,
    base_url: str = "",
    api_key: str = "",
    timeout: float = 30.0,
    verify_tls: bool = True,
    workspace: str | None = None,
):
    """
    Build the Modules 1 & 2 adapter the sidebar asked for.

    Args:
        source: ``"remote"`` for the HTTP client, ``"local"`` for a checkout of
            the shared repository on this machine.
        base_url: service root, used when ``source`` is ``"remote"``.
        api_key: optional bearer token for the service.
        timeout: per-request timeout in seconds.
        verify_tls: set False only for a self-signed development certificate.
        workspace: data folder for the sample pickers, and the place the local
            adapter looks for ``image_pipeline.py``.

    Returns:
        A :class:`PipelineBridge` or a :class:`core.pipeline_remote.RemotePipeline`.
        Both expose the same surface, so the caller does not need to know which.
    """
    if source == SOURCE_LOCAL:
        return PipelineBridge(workspace=workspace)

    from .pipeline_remote import RemotePipeline      # deferred: avoids a cycle

    return RemotePipeline(
        base_url,
        api_key=api_key,
        timeout=timeout,
        verify_tls=verify_tls,
        workspace=workspace,
    )


# --------------------------------------------------------------------------- #
# Image decoding helpers used by the interface
# --------------------------------------------------------------------------- #
def decode_image(data: bytes) -> np.ndarray | None:
    """
    Decode uploaded image bytes into an OpenCV BGR array.

    Args:
        data: raw bytes of a JPEG/PNG/BMP file.

    Returns:
        BGR array, or ``None`` if the bytes could not be decoded.
    """
    import cv2

    if not data:
        return None
    buffer = np.frombuffer(data, np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def read_image(path: str | os.PathLike) -> np.ndarray | None:
    """Read an image from disk into an OpenCV BGR array, or ``None`` on failure."""
    import cv2

    return cv2.imread(str(path))


#: Image extensions accepted by the batch page.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(folder: str | os.PathLike, recursive: bool = True, limit: int | None = None) -> list[Path]:
    """
    Collect image files from ``folder``.

    Args:
        folder: directory to scan.
        recursive: include sub-directories (the datasets are organised per class).
        limit: stop after this many files. ``None`` collects everything.

    Returns:
        A sorted list of paths. An unreadable folder yields an empty list.
    """
    base = Path(folder)
    if not base.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    found = sorted(
        p for p in base.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return found[:limit] if limit else found
