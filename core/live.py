"""
live.py
=======
Real-time inspection from a camera.

This covers the live-stream half of the *Video Processing* extra-effort
requirement: where ``video.py`` ingests a recorded file, this module drives a
camera attached to the machine running the interface, so a board held under the
lens is inspected as it is seen.

The Streamlit constraint
------------------------
Streamlit executes the page script top to bottom and only redraws when that
script finishes or updates a placeholder. A naive ``while True`` capture loop
would therefore never return, and the *Stop* button — which needs its own script
run to register — could never be clicked.

The loop is consequently run in short **chunks**: each script run captures for a
couple of seconds, updates the placeholders as it goes, saves its accumulated
statistics in the session, and asks Streamlit to re-run. The controls become
live again between chunks, and the camera handle is cached so it is opened once
rather than once per chunk.

Everything here is deliberately free of Streamlit imports so the logic can be
tested without a server.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .analysis import InspectionCriteria, InspectionSummary, rejected, summarise
from .detector import DetectionResult
from .pipeline_bridge import PipelineBridge  # noqa: F401  (documents the contract)
from . import pcb_check
from .video import BoardAnchoredConsensus, BoardRecord, stage_frame

#: Either pipeline adapter — the HTTP client or the local bridge.
Pipeline = Any
from .viz import draw_board_labels, draw_detections, draw_verdict_banner

#: How many camera indices to probe when looking for attached devices.
MAX_CAMERA_INDEX = 4

#: How many times to ask a freshly opened device for a frame before giving up.
#: The first read routinely fails while the camera is still starting its stream.
_READ_ATTEMPTS = 3

#: Which capture backend actually worked for each index, filled in by
#: :func:`list_cameras` so :func:`open_camera` does not have to rediscover it.
_BACKEND_FOR_INDEX: dict[int, int] = {}

#: Rolling timeline length. Older entries are discarded so a long session does
#: not grow without bound.
TIMELINE_LIMIT = 400


def _silence_opencv() -> None:
    """
    Suppress OpenCV's console warnings while probing for cameras.

    Probing an index with no device attached is expected to fail, and OpenCV
    logs each failure at C level where Python's ``contextlib.redirect_stderr``
    cannot reach it.
    """
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:                                        # noqa: BLE001
        pass


def capture_backends() -> tuple[int, ...]:
    """
    Capture backends to try, best first, for the platform this is running on.

    OpenCV's default choice is not always the working one. On Windows it
    defaults to Media Foundation (MSMF), which routinely fails to open
    integrated laptop webcams and several USB cameras that DirectShow opens
    without trouble — the device then looks *absent* to this module even
    though the browser can use the very same camera for Snapshot mode. Trying
    DirectShow first there, and keeping the library default last everywhere,
    means the camera is found whichever backend happens to work.
    """
    if sys.platform.startswith("win"):
        return (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
    if sys.platform == "darwin":
        return (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY)
    return (cv2.CAP_V4L2, cv2.CAP_ANY)


def backend_name(backend: int) -> str:
    """Human-readable name for a capture backend constant."""
    try:
        return cv2.videoio_registry.getBackendName(backend)
    except Exception:                                        # noqa: BLE001
        return str(backend)


def _read_a_frame(capture: cv2.VideoCapture) -> bool:
    """
    Whether ``capture`` delivers a real frame, allowing for a slow start.

    A camera that has just been opened often fails its first read or two while
    the driver starts the stream, so a single failed read is not evidence that
    the device does not work.
    """
    for attempt in range(_READ_ATTEMPTS):
        ok, frame = capture.read()
        if ok and frame is not None:
            return True
        time.sleep(0.08)
    return False


def _try_open(index: int, backend: int) -> cv2.VideoCapture | None:
    """Open one index with one backend, returning it only if it delivers a frame."""
    capture = None
    try:
        capture = cv2.VideoCapture(int(index), int(backend))
        if capture.isOpened() and _read_a_frame(capture):
            return capture
    except Exception:                                        # noqa: BLE001
        pass
    if capture is not None:
        capture.release()
    return None


def probe_report(max_index: int = MAX_CAMERA_INDEX) -> list[dict[str, Any]]:
    """
    Probe every index with every backend and report what each attempt did.

    "No camera was detected" has several very different causes — no device,
    a device held by another program (the browser's own camera preview is the
    usual culprit), or a backend that cannot drive this particular camera.
    The interface shows this table so the operator can tell them apart instead
    of guessing.

    Returns:
        One row per (index, backend) attempt, with ``opened`` and ``read``
        recording how far it got.
    """
    _silence_opencv()
    rows: list[dict[str, Any]] = []
    for index in range(max(0, int(max_index))):
        for backend in capture_backends():
            capture = None
            opened = read = False
            try:
                capture = cv2.VideoCapture(int(index), int(backend))
                opened = bool(capture.isOpened())
                if opened:
                    read = _read_a_frame(capture)
            except Exception:                                # noqa: BLE001
                pass
            finally:
                if capture is not None:
                    capture.release()
            rows.append({
                "camera": index,
                "backend": backend_name(backend),
                "opened": opened,
                "delivered a frame": read,
            })
            if read:                     # this index works; no need for the rest
                break
    return rows


def list_cameras(max_index: int = MAX_CAMERA_INDEX) -> list[int]:
    """
    Find camera indices that can actually deliver a frame.

    ``isOpened()`` alone is not sufficient — on several platforms a virtual or
    busy device opens and then fails to read — so each candidate must produce
    one real frame before it is reported. Each index is tried against every
    backend in :func:`capture_backends`, and the one that worked is remembered
    for :func:`open_camera`.

    Args:
        max_index: probe indices ``0`` to ``max_index - 1``.

    Returns:
        The working indices. An empty list simply means no camera is available,
        which is a normal state, not an error.
    """
    _silence_opencv()
    working: list[int] = []
    for index in range(max(0, int(max_index))):
        for backend in capture_backends():
            capture = _try_open(index, backend)
            if capture is not None:
                capture.release()
                _BACKEND_FOR_INDEX[index] = backend
                working.append(index)
                break
    return working


def open_camera(
    index: int = 0,
    width: int | None = 1280,
    height: int | None = 720,
) -> cv2.VideoCapture | None:
    """
    Open a camera and request a capture resolution.

    The backend that :func:`list_cameras` found working for this index is tried
    first; the remaining ones follow, so opening still succeeds if the probe
    never ran (the camera was plugged in afterwards, say).

    Args:
        index: device index, as returned by :func:`list_cameras`.
        width: requested frame width. The driver may ignore it.
        height: requested frame height.

    Returns:
        An open ``VideoCapture``, or ``None`` when the device is unavailable.
    """
    _silence_opencv()
    index = int(index)

    remembered = _BACKEND_FOR_INDEX.get(index)
    order = list(capture_backends())
    if remembered is not None:
        order = [remembered] + [b for b in order if b != remembered]

    for backend in order:
        capture = _try_open(index, backend)
        if capture is None:
            continue
        _BACKEND_FOR_INDEX[index] = backend
        if width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        # A small internal buffer keeps the displayed frame close to the present;
        # a large one shows the operator what the camera saw seconds ago.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:                                    # noqa: BLE001
            pass
        return capture
    return None


def close_camera(capture: cv2.VideoCapture | None) -> None:
    """Release a camera handle, tolerating one that is already released."""
    if capture is None:
        return
    try:
        capture.release()
    except Exception:                                        # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Session statistics
# --------------------------------------------------------------------------- #
@dataclass
class LiveStats:
    """
    Running totals for one live session, accumulated across chunks.

    Held in Streamlit's session state, so the figures survive the re-runs that
    the chunked loop depends on.
    """

    frames: int = 0
    defects: int = 0
    inference_ms_total: float = 0.0
    verdict_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    worst_score: float = 101.0
    worst_frame: np.ndarray | None = None
    worst_caption: str = ""

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.time() - self.started_at)

    @property
    def effective_fps(self) -> float:
        """Frames actually inspected per second of wall-clock time."""
        elapsed = self.elapsed_s
        return self.frames / elapsed if elapsed > 0 else 0.0

    @property
    def mean_inference_ms(self) -> float:
        return self.inference_ms_total / self.frames if self.frames else 0.0

    @property
    def pass_rate(self) -> float:
        """Percentage of inspected frames that passed."""
        if not self.frames:
            return 0.0
        return 100.0 * self.verdict_counts.get("PASS", 0) / self.frames

    def record(self, summary: InspectionSummary, annotated: np.ndarray | None) -> None:
        """Fold one inspected frame into the totals."""
        self.frames += 1
        self.defects += summary.total_defects
        self.inference_ms_total += summary.inference_ms
        self.verdict_counts[summary.verdict] = self.verdict_counts.get(summary.verdict, 0) + 1
        for name, count in summary.class_counts.items():
            self.class_counts[name] = self.class_counts.get(name, 0) + count

        self.timeline.append(
            {
                "frame": self.frames,
                "time_s": round(self.elapsed_s, 2),
                "verdict": summary.verdict,
                "defects": summary.total_defects,
                "quality_score": round(summary.quality_score, 1),
            }
        )
        if len(self.timeline) > TIMELINE_LIMIT:
            del self.timeline[: len(self.timeline) - TIMELINE_LIMIT]

        if summary.quality_score < self.worst_score and annotated is not None:
            self.worst_score = summary.quality_score
            self.worst_frame = annotated
            self.worst_caption = (
                f"Frame {self.frames} (t = {self.elapsed_s:.1f} s) — "
                f"{summary.verdict}, {summary.total_defects} defect(s)"
            )


@dataclass
class ChunkResult:
    """What one chunk of the capture loop produced."""

    frames_processed: int
    last_annotated: np.ndarray | None
    last_summary: InspectionSummary | None
    last_boards: list[BoardRecord] = field(default_factory=list)
    error: str | None = None


@dataclass
class CaptureResult:
    """What one chunk of plain recording produced."""

    frames: int
    last_frame: np.ndarray | None = None
    error: str | None = None


def capture_chunk(
    capture: cv2.VideoCapture,
    seconds: float = 2.0,
    max_fps: float = 12.0,
    on_frame: Callable[[np.ndarray], None] | None = None,
) -> CaptureResult:
    """
    Record from the camera for a bounded slice of time, without inspecting.

    This is the viewfinder half of the Live page. Detection takes seconds per
    frame, so a preview drawn from inspected frames lags far behind what the
    camera is actually pointing at — useless for aiming it. Capturing on its
    own runs at camera speed, so the operator sees the real scene while it is
    being recorded; the whole recording is inspected board by board once they
    stop.

    Args:
        capture: an open camera handle.
        seconds: how long this chunk may run before returning control to
            Streamlit, so the Stop button stays responsive.
        max_fps: ceiling on the preview/record rate. The limit is the browser
            round trip for each previewed frame, not the camera.
        on_frame: called with every frame as it arrives, for display and
            recording.

    Returns:
        A :class:`CaptureResult`. A camera that stops delivering frames is
        reported through ``error`` rather than raising.
    """
    deadline = time.time() + max(0.2, float(seconds))
    interval = 1.0 / max(1.0, float(max_fps))
    frames = 0
    last_frame: np.ndarray | None = None
    error: str | None = None

    while time.time() < deadline:
        started = time.time()

        ok, frame = capture.read()
        if not ok or frame is None:
            error = "The camera stopped delivering frames."
            break

        frames += 1
        last_frame = frame
        if on_frame is not None:
            on_frame(frame)

        remaining = interval - (time.time() - started)
        if remaining > 0:
            time.sleep(min(remaining, max(0.0, deadline - time.time())))

    return CaptureResult(frames, last_frame, error)


# --------------------------------------------------------------------------- #
# The chunked capture loop
# --------------------------------------------------------------------------- #
def run_chunk(
    capture: cv2.VideoCapture,
    bridge: Pipeline,
    detector: Any,
    stats: LiveStats,
    criteria: InspectionCriteria | None = None,
    seconds: float = 2.0,
    target_fps: float = 4.0,
    mode: str = "module",
    do_preprocess: bool = True,
    do_align: bool = True,
    confirm_frames: int = 2,
    confidence: float = 0.25,
    iou: float = 0.45,
    show_labels: bool = True,
    show_confidence: bool = True,
    consensus: BoardAnchoredConsensus | None = None,
    on_frame: Callable[[np.ndarray, InspectionSummary], None] | None = None,
    on_boards: Callable[[list[BoardRecord]], None] | None = None,
    on_capture: Callable[[np.ndarray], None] | None = None,
) -> ChunkResult:
    """
    Capture and inspect frames for a bounded slice of time.

    Args:
        capture: an open camera handle.
        bridge: Modules 1 and 2 adapter.
        detector: Module 3 adapter — either a local ``DefectDetector`` or a
            ``RemoteDetector``; only ``predict`` is required.
        stats: the running session totals, updated in place.
        criteria: acceptance rules applied to every frame.
        seconds: how long this chunk may run before returning control to
            Streamlit so the Stop button becomes clickable again.
        target_fps: how many frames per second to *inspect*. Detection is far
            slower than capture, so the loop paces itself rather than inspecting
            every frame the camera offers.
        mode: pipeline execution mode.
        do_preprocess: run Module 1 on each frame.
        do_align: run Module 2 on each frame. Each detected board is
            straightened in place before detection.
        confirm_frames: temporal smoothing — a defect is only reported once it
            has been seen in this many analyses of the same board. ``1``
            disables it.
        confidence: detector confidence threshold.
        iou: detector NMS IoU threshold.
        show_labels: label each bounding box.
        show_confidence: append the score to each label.
        consensus: the board-anchored consensus shared across chunks (held in
            session state, like stats); created on the fly when ``None``.
        on_frame: called with ``(annotated, summary)`` after each inspected
            frame, so the interface can update its placeholders mid-chunk.
        on_boards: called with the per-board records of each frame, so the
            interface can show one result per board.
        on_capture: called with every frame as it comes off the camera, before
            any processing. The Live page uses it to record the session so the
            whole run can be inspected board by board once it is stopped.

    Returns:
        A :class:`ChunkResult`. A camera that stops delivering frames is
        reported through ``error`` rather than raising.
    """
    criteria = criteria or InspectionCriteria()
    deadline = time.time() + max(0.2, float(seconds))
    interval = 1.0 / max(0.5, float(target_fps))

    if consensus is None:
        consensus = BoardAnchoredConsensus(confirm=max(1, int(confirm_frames)))

    processed = 0
    last_annotated: np.ndarray | None = None
    last_summary: InspectionSummary | None = None
    last_boards: list[BoardRecord] = []
    error: str | None = None

    while time.time() < deadline:
        frame_started = time.time()

        ok, frame = capture.read()
        if not ok or frame is None:
            error = "The camera stopped delivering frames."
            break

        if on_capture is not None:
            on_capture(frame)

        stages = stage_frame(bridge, frame, mode=mode,
                             do_preprocess=do_preprocess, do_align=do_align)
        if len(stages.board_boxes) > 1:
            # A genuine multi-board conveyor frame — find_boards() already
            # gives a stronger answer than either single-board PCB check is
            # designed to provide, so neither one runs.
            pcb_valid, pcb_message = True, None
        else:
            # Student 1's own check (StageResult.pcb_valid) plus this
            # repository's independent second opinion (hue concentration +
            # texture) — either one rejecting is enough. Only reached when
            # stage_frame() fell back to bridge.run(): a frame stage_frame()
            # already resolved to 2+ real boards never goes through this
            # branch at all.
            pcb_valid, pcb_message = pcb_check.evaluate(
                stages.pcb_valid, stages.pcb_message, stages.original
            )
        if pcb_valid is False:
            # Rejected before Modules 1-3 did any work on it (camera pointed
            # at the bench, a hand, nothing at all, ...).
            image_shape = stages.final.shape[:2] if stages.final is not None else (0, 0)
            detection_result = DetectionResult(
                detections=[], inference_ms=0.0, image_shape=image_shape,
                model_name=detector.model_name, error=None,
            )
            summary = rejected(
                pcb_message or "Frame does not appear to contain a PCB.",
                image_shape=image_shape,
            )
            raw_result = detection_result
            board_boxes: list[tuple[int, int, int, int]] = []
            real_boxes = False
            per_board: list[tuple[int, list]] = []
            stable = []
        else:
            raw_result = detector.predict(stages.final, confidence=confidence, iou=iou)

            # Board-anchored temporal consensus — same stabilisation the video page
            # uses: defects are matched per board and only reported after
            # `confirm_frames` analyses, and marks are re-projected onto the
            # board's current rectangle so they stay glued to the moving board.
            board_boxes = list(stages.board_boxes)
            real_boxes = bool(board_boxes)
            if not real_boxes and stages.final is not None:
                fh, fw = stages.final.shape[:2]
                board_boxes = [(0, 0, fw, fh)]
            frame_index = stats.frames
            consensus.update(board_boxes, raw_result.detections, frame_index)
            per_board = consensus.render_by_board(board_boxes, frame_index)
            stable = [d for _bid, dets in per_board for d in dets]
            detection_result = DetectionResult(
                stable, raw_result.inference_ms, raw_result.image_shape,
                raw_result.model_name, error=raw_result.error,
            )
            summary = summarise(detection_result, criteria)

        boards: list[BoardRecord] = []
        for (board_id, dets), board_box in zip(per_board, board_boxes):
            board_summary = summarise(
                DetectionResult(dets, raw_result.inference_ms,
                                raw_result.image_shape,
                                raw_result.model_name, raw_result.error),
                criteria,
            )
            boards.append(BoardRecord(
                label=f"Board {board_id + 1}",
                verdict=board_summary.verdict,
                defect_count=board_summary.total_defects,
                quality_score=board_summary.quality_score,
                confidence=max((d.confidence for d in dets), default=0.0),
                class_counts=dict(board_summary.class_counts),
            ))

        annotated = draw_detections(
            stages.final,
            stable,
            show_labels=show_labels,
            show_confidence=show_confidence,
        )
        if per_board and real_boxes:
            annotated = draw_board_labels(
                annotated,
                [(box, f"Board {board_id + 1}")
                 for (board_id, _dets), box in zip(per_board, board_boxes)],
            )
        annotated = draw_verdict_banner(
            annotated,
            summary.verdict,
            f"live · {summary.total_defects} defect(s) · {summary.inference_ms:.0f} ms",
        )

        stats.record(summary, annotated)
        processed += 1
        last_annotated, last_summary, last_boards = annotated, summary, boards

        if on_frame is not None:
            on_frame(annotated, summary)
        if on_boards is not None:
            on_boards(boards)

        # Pace the loop to the requested inspection rate. When detection already
        # took longer than the interval, continue immediately.
        remaining = interval - (time.time() - frame_started)
        if remaining > 0:
            time.sleep(min(remaining, max(0.0, deadline - time.time())))

    return ChunkResult(processed, last_annotated, last_summary, last_boards, error)
