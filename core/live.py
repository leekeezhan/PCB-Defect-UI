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

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .analysis import InspectionCriteria, InspectionSummary, summarise
from .pipeline_bridge import PipelineBridge  # noqa: F401  (documents the contract)

#: Either pipeline adapter — the HTTP client or the local bridge.
Pipeline = Any
from .viz import draw_detections, draw_verdict_banner

#: How many camera indices to probe when looking for attached devices.
MAX_CAMERA_INDEX = 4

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


def list_cameras(max_index: int = MAX_CAMERA_INDEX) -> list[int]:
    """
    Find camera indices that can actually deliver a frame.

    ``isOpened()`` alone is not sufficient — on several platforms a virtual or
    busy device opens and then fails to read — so each candidate must produce
    one real frame before it is reported.

    Args:
        max_index: probe indices ``0`` to ``max_index - 1``.

    Returns:
        The working indices. An empty list simply means no camera is attached,
        which is a normal state, not an error.
    """
    _silence_opencv()
    working: list[int] = []
    for index in range(max(0, int(max_index))):
        capture = None
        try:
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None:
                    working.append(index)
        except Exception:                                    # noqa: BLE001
            pass
        finally:
            if capture is not None:
                capture.release()
    return working


def open_camera(
    index: int = 0,
    width: int | None = 1280,
    height: int | None = 720,
) -> cv2.VideoCapture | None:
    """
    Open a camera and request a capture resolution.

    Args:
        index: device index, as returned by :func:`list_cameras`.
        width: requested frame width. The driver may ignore it.
        height: requested frame height.

    Returns:
        An open ``VideoCapture``, or ``None`` when the device is unavailable.
    """
    _silence_opencv()
    try:
        capture = cv2.VideoCapture(int(index))
        if not capture.isOpened():
            capture.release()
            return None
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
    except Exception:                                        # noqa: BLE001
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
    error: str | None = None


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
    do_align: bool = False,
    confidence: float = 0.25,
    iou: float = 0.45,
    show_labels: bool = True,
    show_confidence: bool = True,
    on_frame: Callable[[np.ndarray, InspectionSummary], None] | None = None,
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
        do_align: run Module 2 on each frame. Off by default — a hand-held board
            rarely presents a clean four-corner boundary, and a failed alignment
            on every frame only costs time.
        confidence: detector confidence threshold.
        iou: detector NMS IoU threshold.
        show_labels: label each bounding box.
        show_confidence: append the score to each label.
        on_frame: called with ``(annotated, summary)`` after each inspected
            frame, so the interface can update its placeholders mid-chunk.

    Returns:
        A :class:`ChunkResult`. A camera that stops delivering frames is
        reported through ``error`` rather than raising.
    """
    criteria = criteria or InspectionCriteria()
    deadline = time.time() + max(0.2, float(seconds))
    interval = 1.0 / max(0.5, float(target_fps))

    processed = 0
    last_annotated: np.ndarray | None = None
    last_summary: InspectionSummary | None = None
    error: str | None = None

    while time.time() < deadline:
        frame_started = time.time()

        ok, frame = capture.read()
        if not ok or frame is None:
            error = "The camera stopped delivering frames."
            break

        stages = bridge.run(frame, mode=mode, do_preprocess=do_preprocess, do_align=do_align)
        detection_result = detector.predict(stages.final, confidence=confidence, iou=iou)
        summary = summarise(detection_result, criteria)

        annotated = draw_detections(
            stages.final,
            detection_result.detections,
            show_labels=show_labels,
            show_confidence=show_confidence,
        )
        annotated = draw_verdict_banner(
            annotated,
            summary.verdict,
            f"live · {summary.total_defects} defect(s) · {summary.inference_ms:.0f} ms",
        )

        stats.record(summary, annotated)
        processed += 1
        last_annotated, last_summary = annotated, summary

        if on_frame is not None:
            on_frame(annotated, summary)

        # Pace the loop to the requested inspection rate. When detection already
        # took longer than the interval, continue immediately.
        remaining = interval - (time.time() - frame_started)
        if remaining > 0:
            time.sleep(min(remaining, max(0.0, deadline - time.time())))

    return ChunkResult(processed, last_annotated, last_summary, error)
