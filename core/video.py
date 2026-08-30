"""
video.py
========
Frame-by-frame execution of the full inspection pipeline over a video stream.

This covers the *Video Processing* extra-effort requirement in the assignment
specification: "enable ingestion of video streams as input data and execute
real-time algorithmic analysis across individual frames".

Each sampled frame travels the same path as a still image — Module 1
pre-processing, Module 2 alignment, Module 3 detection, then the Module 4
verdict — so the video page reports exactly what the single-image page would
report for any given frame.

Two practical constraints shape the implementation:

* OpenCV's ``VideoWriter`` needs a path and a fixed frame size, so a temporary
  file is used internally and every annotated frame is resized to the size of
  the first one. The temporary files are always removed.
* Module 2 crops to the detected board, which changes the frame size from one
  frame to the next; the resize above absorbs that.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .analysis import InspectionCriteria, InspectionSummary, summarise
from .detector import DefectDetector
from .pipeline_bridge import PipelineBridge  # noqa: F401  (documents the contract)

#: Either pipeline adapter — the HTTP client or the local bridge.
Pipeline = Any
from .viz import draw_detections, draw_verdict_banner

_FOURCC = "mp4v"


@dataclass
class FrameRecord:
    """Per-frame outcome, used for the timeline table and the defect chart."""

    frame_index: int
    timestamp_s: float
    verdict: str
    defect_count: int
    quality_score: float
    class_counts: dict[str, int] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "frame": self.frame_index,
            "time_s": round(self.timestamp_s, 2),
            "verdict": self.verdict,
            "defects": self.defect_count,
            "quality_score": round(self.quality_score, 1),
        }


@dataclass
class VideoResult:
    """Outcome of one video inspection run."""

    video_bytes: bytes | None
    records: list[FrameRecord]
    summaries: list[InspectionSummary]
    frames_read: int
    frames_analysed: int
    fps: float
    duration_s: float
    worst_frame: np.ndarray | None = None
    worst_caption: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.video_bytes is not None

    @property
    def total_defects(self) -> int:
        return sum(record.defect_count for record in self.records)

    @property
    def failed_frames(self) -> int:
        return sum(1 for record in self.records if record.verdict == "FAIL")


def probe(video_bytes: bytes) -> dict[str, Any]:
    """
    Read a video's metadata without processing it, so the interface can show the
    operator what they uploaded and estimate the run time before starting.

    Returns:
        A dictionary with ``frame_count``, ``fps``, ``width``, ``height`` and
        ``duration_s``. Values are ``0`` when the video could not be opened.
    """
    empty = {"frame_count": 0, "fps": 0.0, "width": 0, "height": 0, "duration_s": 0.0}
    if not video_bytes:
        return empty

    path = _write_temp(video_bytes)
    try:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return empty
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        return {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_s": frame_count / fps if fps else 0.0,
        }
    finally:
        _remove(path)


def process_video(
    video_bytes: bytes,
    bridge: Pipeline,
    detector: DefectDetector,
    criteria: InspectionCriteria | None = None,
    mode: str = "module",
    do_preprocess: bool = True,
    do_align: bool = False,
    frame_stride: int = 5,
    max_frames: int | None = 150,
    confidence: float = 0.25,
    iou: float = 0.45,
    progress: Callable[[float, str], None] | None = None,
) -> VideoResult:
    """
    Run the inspection pipeline across a video and produce an annotated copy.

    Args:
        video_bytes: the uploaded MP4 as bytes.
        bridge: adapter for Modules 1 and 2.
        detector: adapter for Module 3.
        criteria: acceptance rules applied to every analysed frame.
        mode: pipeline execution mode, passed through to the bridge.
        do_preprocess: run Module 1 on each frame.
        do_align: run Module 2 on each frame. Off by default — a conveyor stream
            rarely presents a clean four-corner board boundary, and a failed
            alignment on every frame only slows the run down.
        frame_stride: analyse every *n*-th frame. Frames in between are written
            to the output carrying the most recent annotation, which keeps the
            output playing at the original speed without paying for detection on
            every frame.
        max_frames: stop after this many *analysed* frames, so a long clip cannot
            hang the interface. ``None`` processes the whole video.
        confidence: detector confidence threshold.
        iou: detector NMS IoU threshold.
        progress: optional callback receiving ``(fraction, message)`` for the
            Streamlit progress bar.

    Returns:
        A :class:`VideoResult`. On failure, ``error`` explains what went wrong
        and ``video_bytes`` is ``None``.
    """
    criteria = criteria or InspectionCriteria()
    if not video_bytes:
        return VideoResult(None, [], [], 0, 0, 0.0, 0.0, error="No video data was supplied.")

    input_path = _write_temp(video_bytes)
    output_path: str | None = None
    writer: cv2.VideoWriter | None = None

    records: list[FrameRecord] = []
    summaries: list[InspectionSummary] = []
    worst_frame: np.ndarray | None = None
    worst_score = 101.0
    worst_caption = ""

    try:
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            return VideoResult(None, [], [], 0, 0, 0.0, 0.0,
                               error="The video file could not be opened by OpenCV.")

        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = max(1, int(frame_stride))
        budget = max_frames if max_frames and max_frames > 0 else None

        output_path = tempfile.mktemp(suffix="_inspected.mp4")
        output_size: tuple[int, int] | None = None

        frames_read = 0
        frames_analysed = 0
        last_annotated: np.ndarray | None = None

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1

            if frames_read % stride == 1 or stride == 1:
                stages = bridge.run(frame, mode=mode,
                                    do_preprocess=do_preprocess, do_align=do_align)
                detection_result = detector.predict(
                    stages.final, confidence=confidence, iou=iou
                )
                summary = summarise(detection_result, criteria)
                summaries.append(summary)

                timestamp = (frames_read - 1) / fps if fps else 0.0
                records.append(
                    FrameRecord(
                        frame_index=frames_read,
                        timestamp_s=timestamp,
                        verdict=summary.verdict,
                        defect_count=summary.total_defects,
                        quality_score=summary.quality_score,
                        class_counts=dict(summary.class_counts),
                    )
                )

                annotated = draw_detections(stages.final, detection_result.detections)
                annotated = draw_verdict_banner(
                    annotated,
                    summary.verdict,
                    f"frame {frames_read} · {summary.total_defects} defect(s) · t={timestamp:.1f}s",
                )
                last_annotated = annotated
                frames_analysed += 1

                if summary.quality_score < worst_score:
                    worst_score = summary.quality_score
                    worst_frame = annotated
                    worst_caption = (
                        f"Frame {frames_read} (t = {timestamp:.1f} s) — "
                        f"{summary.verdict}, {summary.total_defects} defect(s)"
                    )

                if progress is not None:
                    fraction = min(1.0, frames_read / total_frames) if total_frames else 0.0
                    progress(fraction, f"Analysed frame {frames_read} — "
                                       f"{summary.total_defects} defect(s)")

            # Write either the freshly annotated frame or the most recent one, so
            # the output keeps the source frame rate even when detection is
            # sampled every n-th frame.
            frame_to_write = last_annotated if last_annotated is not None else frame
            if writer is None:
                height, width = frame_to_write.shape[:2]
                output_size = (width, height)
                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*_FOURCC), fps, output_size
                )
                if not writer.isOpened():
                    capture.release()
                    return VideoResult(
                        None, records, summaries, frames_read, frames_analysed, fps, 0.0,
                        error="OpenCV could not open a video writer for the output file.",
                    )
            _write_frame(writer, frame_to_write, output_size)

            if budget is not None and frames_analysed >= budget:
                break

        capture.release()
        if writer is not None:
            writer.release()

        if output_path and os.path.exists(output_path):
            with open(output_path, "rb") as handle:
                encoded = handle.read()
        else:
            encoded = None

        return VideoResult(
            video_bytes=encoded,
            records=records,
            summaries=summaries,
            frames_read=frames_read,
            frames_analysed=frames_analysed,
            fps=fps,
            duration_s=frames_read / fps if fps else 0.0,
            worst_frame=worst_frame,
            worst_caption=worst_caption,
            error=None if encoded else "The annotated video could not be written.",
        )

    except Exception as exc:                                  # noqa: BLE001
        return VideoResult(
            None, records, summaries, 0, 0, 0.0, 0.0,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if writer is not None:
            writer.release()
        _remove(input_path)
        _remove(output_path)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_frame(writer: cv2.VideoWriter | None, frame: np.ndarray | None,
                 size: tuple[int, int] | None) -> None:
    """Write one frame, resizing it to the writer's fixed frame size."""
    if writer is None or frame is None or size is None:
        return
    height, width = frame.shape[:2]
    if (width, height) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    writer.write(frame)


def _write_temp(data: bytes, suffix: str = ".mp4") -> str:
    """Persist bytes to a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        return handle.name


def _remove(path: str | None) -> None:
    """Delete a temporary file, ignoring the case where it is already gone."""
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
