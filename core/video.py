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

from . import pcb_check
from .analysis import InspectionCriteria, InspectionSummary, rejected, summarise
from .detector import DefectDetector, DetectionResult
from .pipeline_bridge import PipelineBridge  # noqa: F401  (documents the contract)
from .roi import crop_with_padding, find_pcb_regions

#: Either pipeline adapter — the HTTP client or the local bridge.
Pipeline = Any
from .viz import draw_detections, draw_verdict_banner

# Codecs to try, in order, when opening the output video writer. ``avc1``
# (H.264) is what Chrome/Edge/Firefox can all play inline in a <video> tag,
# but a plain ``pip install opencv-python`` build on Windows often ships
# without a working H.264 encoder (it's licensed, and OpenCV's official
# wheels leave it out). When that happens ``VideoWriter.isOpened()`` comes
# back ``False`` for that codec, so each candidate is tried in turn and the
# first one that actually opens is used. ``mp4v`` is last because it always
# opens (it ships in every OpenCV build) but is not natively playable in
# Chrome's inline player — it's the safety net that guarantees a video is
# still produced even on a machine with no H.264 encoder available.
_FOURCC_CANDIDATES = ("avc1", "H264", "mp4v")


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


def scan_boards(
    video_bytes: bytes,
    every_n_frames: int = 3,
    match_tolerance: int = 120,
    drop_after: int = 4,
    max_boards: int = 50,
    progress: Callable[[float, str], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Capture every board in the clip exactly once, and return one raw crop per
    physical board.

    A board stays visible for hundreds of frames as it travels the belt, so
    sampling frames blindly returns the same board over and over. The obvious
    fix — capture a board as it crosses the mid-line — silently loses any board
    that never crosses it: one already past the middle when the clip starts,
    and one still arriving when the clip ends. This tracks each board across
    frames instead, by following its centre from one sampled frame to the next,
    and captures each track once at its *best* moment: the frame where the
    board is completely inside the picture and closest to the centre, where it
    is least distorted.

    A board that is never completely inside the picture — it enters as the clip
    ends, say — is still returned, marked ``fully_visible=False``, so the
    caller can report it rather than drop it silently.

    The crops are returned raw: Modules 1, 2 and 3 are the caller's job, so
    each board goes through exactly the same path as a still image on any
    other page.

    Args:
        video_bytes: the uploaded video as bytes.
        every_n_frames: only test every *n*-th frame, since a board moves a
            few pixels per frame and testing all of them buys nothing.
        match_tolerance: how far (in pixels) a board's centre may move between
            two sampled frames and still be recognised as the same board.
        drop_after: close a track once it has gone unmatched for this many
            sampled frames — the board has left the picture.
        max_boards: cap on how many boards are returned.
        progress: optional ``(fraction, message)`` callback for a progress bar.

    Returns:
        One dictionary per board, in the order the boards first appear, with
        ``frame``, ``timestamp_s``, ``box`` (``(x, y, w, h)`` in the full
        frame), ``crop`` (BGR array), ``fully_visible`` and ``views`` (how many
        sampled frames the board was seen in).
    """
    if not video_bytes:
        return []

    path = _write_temp(video_bytes)
    try:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return []

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        centre_x = width / 2.0
        step = max(1, int(every_n_frames))

        tracks: list[dict[str, Any]] = []
        active: list[dict[str, Any]] = []
        frame_id = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_id % step == 0:
                boxes = find_pcb_regions(frame)
                claimed: set[int] = set()

                # Continue the tracks that are still on screen: each takes the
                # nearest unclaimed box, which is unambiguous here because
                # boards are far further apart than they move between samples.
                for track in active:
                    nearest_index, nearest_distance = None, None
                    for index, (x, _, w, _) in enumerate(boxes):
                        if index in claimed:
                            continue
                        distance = abs(x + w / 2.0 - track["last_centre"])
                        if distance <= match_tolerance and (
                            nearest_distance is None or distance < nearest_distance
                        ):
                            nearest_index, nearest_distance = index, distance
                    if nearest_index is not None:
                        claimed.add(nearest_index)
                        _record_view(track, boxes[nearest_index], frame, frame_id,
                                     width, centre_x)

                # Anything left is a board that has just come into view.
                for index, box in enumerate(boxes):
                    if index in claimed:
                        continue
                    track = {"first_frame": frame_id, "views": 0,
                             "best": None, "last_centre": 0.0, "last_seen": frame_id}
                    _record_view(track, box, frame, frame_id, width, centre_x)
                    tracks.append(track)
                    active.append(track)

                active = [
                    track for track in active
                    if frame_id - track["last_seen"] <= drop_after * step
                ]

            if progress is not None and total_frames:
                progress(min(1.0, frame_id / total_frames),
                         f"Scanned frame {frame_id} — {len(tracks)} board(s) found")
            frame_id += 1

        capture.release()

        boards: list[dict[str, Any]] = []
        for track in sorted(tracks, key=lambda t: t["first_frame"])[:max_boards]:
            best = track["best"]
            if best is None:
                continue
            boards.append({
                "frame": best["frame"],
                "timestamp_s": best["frame"] / fps if fps else 0.0,
                "box": best["box"],
                "crop": best["crop"],
                "fully_visible": best["fully_visible"],
                "views": track["views"],
            })
        return boards
    finally:
        _remove(path)


def _record_view(
    track: dict[str, Any],
    box: tuple[int, int, int, int],
    frame: np.ndarray,
    frame_id: int,
    width: int,
    centre_x: float,
) -> None:
    """
    Add one sighting to a track, keeping only the best view of the board.

    "Best" is a board that is completely inside the picture, and among those
    the one sitting closest to the centre of the frame. A board touching
    either side edge is cut off, so it is only ever kept while nothing better
    has been seen. The winning crop is held as it is met, which avoids a
    second pass over the video to fetch it back.
    """
    x, _, w, _ = box
    fully_visible = x > 1 and (x + w) < width - 1
    offset = abs(x + w / 2.0 - centre_x)

    track["last_centre"] = x + w / 2.0
    track["last_seen"] = frame_id
    track["views"] += 1

    current = track["best"]
    better = (
        current is None
        # A complete view always beats a cut-off one; between two of a kind,
        # the one nearer the middle of the frame wins.
        or (fully_visible and not current["fully_visible"])
        or (fully_visible == current["fully_visible"] and offset < current["offset"])
    )
    if better:
        track["best"] = {
            "frame": frame_id, "box": box, "crop": crop_with_padding(frame, box),
            "fully_visible": fully_visible, "offset": offset,
        }


def extract_frame(video_bytes: bytes, frame_index: int) -> np.ndarray | None:
    """
    Read a single frame by index from an in-memory video.

    Used by the video page's "inspect a single frame" tool, so the operator
    can pull one frame out of an uploaded clip and run it through the full
    pipeline on its own, independent of the stride/budget settings
    :func:`process_video` uses for the batch run.

    Args:
        video_bytes: the uploaded MP4 as bytes.
        frame_index: zero-based frame number to read.

    Returns:
        The frame as a BGR array, or ``None`` if the video or the index
        could not be read.
    """
    if not video_bytes:
        return None
    path = _write_temp(video_bytes)
    try:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = capture.read()
        capture.release()
        return frame if ok else None
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
                # Student 1's own check (StageResult.pcb_valid) plus this
                # repository's independent second opinion (hue concentration +
                # texture) — either one rejecting is enough.
                pcb_valid, pcb_message = pcb_check.evaluate(
                    stages.pcb_valid, stages.pcb_message, stages.original
                )
                if pcb_valid is False:
                    # Rejected before Modules 1-3 did any work on this frame —
                    # see core.analysis.rejected.
                    image_shape = stages.final.shape[:2] if stages.final is not None else (0, 0)
                    detection_result = DetectionResult(
                        detections=[], inference_ms=0.0, image_shape=image_shape,
                        model_name=detector.model_name, error=None,
                    )
                    summary = rejected(
                        pcb_message or "Frame does not appear to contain a PCB.",
                        image_shape=image_shape,
                    )
                else:
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
                writer = _open_writer(output_path, fps, output_size)
                if writer is None:
                    capture.release()
                    return VideoResult(
                        None, records, summaries, frames_read, frames_analysed, fps, 0.0,
                        error="OpenCV could not open a video writer for the output file "
                              "(tried: " + ", ".join(_FOURCC_CANDIDATES) + ").",
                    )
            _write_frame(writer, frame_to_write, output_size)

            if budget is not None and frames_analysed >= budget:
                break

        capture.release()
        if writer is not None:
            writer.release()

        if output_path and os.path.exists(output_path):
            encoded = _read_bytes(output_path)
            # If none of the OpenCV codecs above produced real H.264 (the
            # common case: pip's opencv-python has no licensed H.264 encoder
            # built in), fall back to an external ffmpeg re-encode so the
            # video still plays inline in the browser. This is best-effort —
            # if ffmpeg isn't available the un-re-encoded file is used as-is,
            # same as before this fallback existed.
            encoded = _ensure_browser_playable(output_path, encoded) or encoded
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
def _open_writer(
    output_path: str, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter | None:
    """
    Open a ``VideoWriter`` for the output file, trying each codec in
    ``_FOURCC_CANDIDATES`` in order and keeping the first one that actually
    opens. Returns ``None`` if none of them work.
    """
    for fourcc in _FOURCC_CANDIDATES:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*fourcc), fps, size
        )
        if writer.isOpened():
            return writer
        writer.release()
    return None


def _write_frame(writer: cv2.VideoWriter | None, frame: np.ndarray | None,
                 size: tuple[int, int] | None) -> None:
    """Write one frame, resizing it to the writer's fixed frame size."""
    if writer is None or frame is None or size is None:
        return
    height, width = frame.shape[:2]
    if (width, height) != size:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    writer.write(frame)


def _read_bytes(path: str) -> bytes | None:
    """Read a file's contents, or ``None`` if it doesn't exist."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return handle.read()


def _ensure_browser_playable(source_path: str, fallback: bytes | None) -> bytes | None:
    """
    Re-encode the just-written video to real H.264 with an external ffmpeg,
    so it plays inline in Chrome/Edge/Firefox even when none of the
    ``_FOURCC_CANDIDATES`` OpenCV codecs opened (the usual case: the H.264
    encoder is left out of pip's ``opencv-python`` wheels for licensing
    reasons, so ``avc1``/``H264`` above silently fall back to ``mp4v``,
    which OpenCV can always write but browsers can't always play).

    This is best-effort and never raises: if no ffmpeg binary can be found,
    or the re-encode fails for any reason, the caller keeps using the file
    OpenCV already produced. Installing the ``imageio-ffmpeg`` package
    (``pip install imageio-ffmpeg``) is the easiest way to make this branch
    succeed, since it bundles a ready-to-run ffmpeg binary — no separate
    ffmpeg install needed.
    """
    ffmpeg_exe = _find_ffmpeg()
    if not ffmpeg_exe:
        return fallback

    import subprocess

    reencoded_path = tempfile.mktemp(suffix="_h264.mp4")
    try:
        result = subprocess.run(
            [
                ffmpeg_exe, "-y",
                "-i", source_path,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                reencoded_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            return fallback
        reencoded = _read_bytes(reencoded_path)
        return reencoded if reencoded else fallback
    except Exception:                                          # noqa: BLE001
        return fallback
    finally:
        _remove(reencoded_path)


def _find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary: prefer imageio-ffmpeg's bundled copy (no
    separate install needed), fall back to one already on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        pass

    import shutil
    return shutil.which("ffmpeg")


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
