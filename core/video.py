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

from .analysis import InspectionCriteria, InspectionSummary, rejected, summarise
from .detector import DefectDetector, Detection, DetectionResult
from .pipeline_bridge import PipelineBridge, StageResult  # noqa: F401  (documents the contract)
from .roi import crop_with_padding, find_pcb_regions

#: Either pipeline adapter — the HTTP client or the local bridge.
Pipeline = Any
from .viz import draw_board_labels, draw_detections, draw_verdict_banner

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
class BoardRecord:
    """Outcome of ONE board within one analysed frame."""

    label: str
    verdict: str
    defect_count: int
    quality_score: float
    confidence: float = 0.0    # best detection confidence on this board, 0 = none
    class_counts: dict[str, int] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "board": self.label,
            "verdict": self.verdict,
            "defects": self.defect_count,
            "quality_score": round(self.quality_score, 1),
            "confidence": round(self.confidence, 2),
        }


@dataclass
class FrameRecord:
    """Per-frame outcome, used for the timeline table and the defect chart."""

    frame_index: int
    timestamp_s: float
    verdict: str
    defect_count: int
    quality_score: float
    class_counts: dict[str, int] = field(default_factory=dict)
    boards: list[BoardRecord] = field(default_factory=list)

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


# --------------------------------------------------------------------------- #
# Board-anchored temporal consensus
# --------------------------------------------------------------------------- #
@dataclass
class _NormTrack:
    """One defect candidate, stored in board-normalised coordinates."""
    board_id: int
    class_id: int
    class_name: str
    nx1: float
    ny1: float
    nx2: float
    ny2: float
    confidences: list[float]
    last_seen: int        # index of the last analysed frame in which it matched
    used: bool = False    # consumed by the current frame's matching pass

    @property
    def seen(self) -> int:
        return len(self.confidences)


def _boxes_iou(a: tuple[float, float, float, float],
               b: tuple[float, float, float, float]) -> float:
    """IoU of two ``(x, y, w, h)`` rectangles."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class BoardAnchoredConsensus:
    """
    Anchor detection marks to the boards they belong to and stabilise them
    across time.

    The same physical board is analysed many times as it travels a conveyor,
    and independent per-frame predictions flicker (motion blur, alignment
    jitter, boxes hovering at the confidence threshold). Two consequences are
    solved here:

    1. **Stability** — a defect is only reported once it has been observed in
       at least ``confirm`` analyses of the same board; one-off findings never
       survive.
    2. **Fixed coordinates** — board rectangles move from frame to frame, so a
       defect is stored NORMALISED to its board (0..1 within the rectangle) and
       re-projected onto the board's CURRENT rectangle when rendered. The mark
       therefore stays exactly on the board even while it moves, and even on a
       frame where the detector missed the defect.

    ``confirm == 1`` disables smoothing and reports every detection as seen.
    """

    def __init__(self, confirm: int = 2, window: int = 3,
                 iou_threshold: float = 0.3) -> None:
        self.confirm = max(1, int(confirm))
        self.window = max(1, int(window))
        self.iou_threshold = float(iou_threshold)
        self._tracks: list[_NormTrack] = []
        self._next_board_id = 0
        self._boards: dict[int, tuple[float, float, float, float]] = {}

    @staticmethod
    def _norm_iou(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        return inter / union if union > 0 else 0.0

    def _match_boards(self, boxes: list[tuple[int, int, int, int]]):
        """
        Match the frame's board rectangles to known board identities by IoU,
        updating each identity with its latest rectangle. New rectangles get a
        fresh id. Returns ``[(box, board_id), ...]`` in the order given.
        """
        matched: list[tuple[tuple[int, int, int, int], int]] = []
        used: set[int] = set()
        for box in boxes:
            box_f = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            best_id: int | None = None
            best_iou = self.iou_threshold
            for board_id, known in self._boards.items():
                if board_id in used:
                    continue
                iou = _boxes_iou(box_f, known)
                if iou >= best_iou:
                    best_id, best_iou = board_id, iou
            if best_id is None:
                best_id = self._next_board_id
                self._next_board_id += 1
            self._boards[best_id] = box_f
            used.add(best_id)
            matched.append((box, best_id))
        return matched

    def _nearest_board(self, det: Detection,
                       boxes: list[tuple[int, int, int, int]]) -> int | None:
        """Index of the board containing the detection, else the nearest one."""
        cx, cy = det.centre
        best: int | None = None
        best_dist = float("inf")
        for i, (x, y, w, h) in enumerate(boxes):
            if x <= cx <= x + w and y <= cy <= y + h:
                return i
            dx = min(abs(cx - x), abs(cx - (x + w)))
            dy = min(abs(cy - y), abs(cy - (y + h)))
            dist = dx + dy
            if dist < best_dist:
                best, best_dist = i, dist
        return best

    def update(self, boxes: list[tuple[int, int, int, int]],
               detections: list[Detection], frame_index: int) -> None:
        """
        Feed one analysed frame's board rectangles and raw detections. Defects
        are stored normalised to their board and matched across analyses by
        class + IoU (in normalised space, so board size differences do not
        matter).
        """
        matched = self._match_boards(boxes)
        box_list = [box for box, _ in matched]

        # Drop candidates that have not been seen for a whole window.
        self._tracks = [
            t for t in self._tracks if frame_index - t.last_seen < self.window
        ]
        for track in self._tracks:
            track.used = False

        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            index = self._nearest_board(det, box_list)
            if index is None:
                continue
            (x, y, w, h), board_id = matched[index]
            if w <= 0 or h <= 0:
                continue
            norm = (
                (det.x1 - x) / w, (det.y1 - y) / h,
                (det.x2 - x) / w, (det.y2 - y) / h,
            )
            best: _NormTrack | None = None
            best_iou = self.iou_threshold
            for track in self._tracks:
                if track.used or track.board_id != board_id:
                    continue
                if track.class_name != det.class_name:
                    continue
                iou = self._norm_iou(
                    (track.nx1, track.ny1, track.nx2, track.ny2), norm
                )
                if iou >= best_iou:
                    best, best_iou = track, iou
            if best is not None:
                best.nx1, best.ny1, best.nx2, best.ny2 = norm
                best.confidences.append(det.confidence)
                best.last_seen = frame_index
                best.used = True
            else:
                self._tracks.append(_NormTrack(
                    board_id, det.class_id, det.class_name,
                    norm[0], norm[1], norm[2], norm[3],
                    [det.confidence], frame_index,
                ))

    def render_by_board(
        self, boxes: list[tuple[int, int, int, int]], frame_index: int
    ) -> list[tuple[int, list[Detection]]]:
        """
        Reproduce the STABLE marks per board, onto the frame's CURRENT board
        rectangles.

        Confirmed defects are re-projected from normalised coordinates onto the
        board's current position, so each mark stays exactly on the board even
        though it has moved since the defect was last detected. Returns
        ``[(board_id, detections), ...]`` in the order of ``boxes``, so callers
        can label each board and summarise per board.
        """
        matched = self._match_boards(boxes)
        out: list[tuple[int, list[Detection]]] = []
        for (x, y, w, h), board_id in matched:
            dets: list[Detection] = []
            for track in self._tracks:
                if track.board_id != board_id or track.seen < self.confirm:
                    continue
                if frame_index - track.last_seen >= self.window:
                    continue
                dets.append(Detection(
                    track.class_id, track.class_name,
                    float(sum(track.confidences) / len(track.confidences)),
                    x + track.nx1 * w, y + track.ny1 * h,
                    x + track.nx2 * w, y + track.ny2 * h,
                ))
            out.append((board_id, dets))
        return out

    def render(self, boxes: list[tuple[int, int, int, int]],
               frame_index: int) -> list[Detection]:
        """Flat version of :meth:`render_by_board` for compatibility."""
        flat: list[Detection] = []
        for _board_id, dets in self.render_by_board(boxes, frame_index):
            flat.extend(dets)
        return flat

    def reset(self) -> None:
        self._tracks.clear()
        self._boards.clear()


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


def stage_frame(
    bridge: Any,
    frame: np.ndarray,
    *,
    mode: str,
    do_preprocess: bool,
    do_align: bool,
) -> StageResult:
    """
    Run Modules 1 and 2 over one frame.

    With alignment enabled and more than one board detected (conveyor footage),
    Module 1 pre-processes the frame and Module 2 then rectifies every board IN
    PLACE — each board is warped onto its own axis-aligned rectangle, so the
    operator still sees the conveyor picture but every board is at its correct
    angle. Single-board frames and adapters without board rectification fall
    back to ``bridge.run`` unchanged.
    """
    if (do_align and hasattr(bridge, "find_boards")
            and hasattr(bridge, "preprocess") and hasattr(bridge, "rectify_frame")):
        try:
            boxes = list(bridge.find_boards(frame))
        except Exception:                              # noqa: BLE001
            boxes = []
        if len(boxes) > 1:
            working = frame
            preprocessed: np.ndarray | None = None
            notes: list[str] = []
            if do_preprocess:
                preprocessed = bridge.preprocess(frame)
                if preprocessed is None:
                    notes.append(
                        "Module 1 returned no output — the original frame was kept."
                    )
                else:
                    working = preprocessed
            rectified = bridge.rectify_frame(working)
            if rectified is not None:
                notes.append(
                    "Each board was straightened in place before detection."
                )
                return StageResult(original=frame, preprocessed=preprocessed,
                                   aligned=rectified, final=rectified,
                                   notes=notes, mode=mode,
                                   board_boxes=boxes)
    result = bridge.run(frame, mode=mode,
                        do_preprocess=do_preprocess, do_align=do_align)
    # Board rectangles power the detection anchoring (marks follow the moving
    # boards). The remote adapter gets them from the service response.
    if hasattr(bridge, "find_boards") and not result.board_boxes:
        try:
            result.board_boxes = list(bridge.find_boards(frame))
        except Exception:                              # noqa: BLE001
            pass
    return result


def process_video(
    video_bytes: bytes,
    bridge: Pipeline,
    detector: DefectDetector,
    criteria: InspectionCriteria | None = None,
    mode: str = "module",
    do_preprocess: bool = True,
    do_align: bool = True,
    frame_stride: int = 5,
    max_frames: int | None = 150,
    confirm_frames: int = 2,
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
        do_align: run Module 2 on each frame. Each detected board is
            straightened in place before detection, so conveyor footage with
            several boards is handled too (aligning the whole multi-board frame
            would be a no-op).
        frame_stride: analyse every *n*-th frame. Frames in between are written
            to the output carrying the most recent annotation, which keeps the
            output playing at the original speed without paying for detection on
            every frame.
        max_frames: stop after this many *analysed* frames, so a long clip cannot
            hang the interface. ``None`` processes the whole video.
        confirm_frames: temporal smoothing. The same board is analysed several
            times as it travels, and a defect is only reported once it has been
            seen in this many analyses of one board — random one-off findings
            (typical at the confidence threshold) never survive. ``1`` disables
            smoothing and reports every per-frame detection.
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

    # Board-anchored consensus: a defect is reported only after it has been
    # observed across `confirm_frames` analyses of the same board, and its mark
    # is re-projected onto the board's current rectangle so it always stays on
    # the board while it moves (see BoardAnchoredConsensus).
    consensus = BoardAnchoredConsensus(confirm=max(1, int(confirm_frames)))

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
                stages = stage_frame(bridge, frame, mode=mode,
                                     do_preprocess=do_preprocess,
                                     do_align=do_align)
                if stages.pcb_valid is False:
                    # Student 1's validate_pcb_image() rejected this frame
                    # before Modules 1-3 did any work on it — see
                    # StageResult.pcb_valid / core.analysis.rejected. Only
                    # reached when stage_frame() fell back to bridge.run():
                    # a frame stage_frame() already resolved to 2+ real boards
                    # never goes through this single-board check at all.
                    image_shape = stages.final.shape[:2] if stages.final is not None else (0, 0)
                    detection_result = DetectionResult(
                        detections=[], inference_ms=0.0, image_shape=image_shape,
                        model_name=detector.model_name, error=None,
                    )
                    summary = rejected(
                        stages.pcb_message or "Frame does not appear to contain a PCB.",
                        image_shape=image_shape,
                    )
                    raw_result = detection_result
                    board_boxes: list[tuple[int, int, int, int]] = []
                    real_boxes = False
                    per_board: list[tuple[int, list]] = []
                else:
                    raw_result = detector.predict(
                        stages.final, confidence=confidence, iou=iou
                    )
                    board_boxes = list(stages.board_boxes)
                    real_boxes = bool(board_boxes)
                    if not real_boxes and stages.final is not None:
                        # The adapter reported no board rectangles (e.g. a minimal
                        # service omitting the optional "boards" field): anchor to a
                        # full-frame pseudo-board so marks still render.
                        fh, fw = stages.final.shape[:2]
                        board_boxes = [(0, 0, fw, fh)]
                    consensus.update(board_boxes, raw_result.detections, frames_analysed)
                    per_board = consensus.render_by_board(board_boxes, frames_analysed)
                    stable = [d for _bid, dets in per_board for d in dets]
                    detection_result = DetectionResult(
                        stable, raw_result.inference_ms, raw_result.image_shape,
                        raw_result.model_name, error=raw_result.error,
                    )
                    summary = summarise(detection_result, criteria)
                summaries.append(summary)

                timestamp = (frames_read - 1) / fps if fps else 0.0
                board_records: list[BoardRecord] = []
                for (board_id, dets), board_box in zip(per_board, board_boxes):
                    board_summary = summarise(
                        DetectionResult(dets, raw_result.inference_ms,
                                        raw_result.image_shape,
                                        raw_result.model_name, raw_result.error),
                        criteria,
                    )
                    board_records.append(BoardRecord(
                        label=f"Board {board_id + 1}",
                        verdict=board_summary.verdict,
                        defect_count=board_summary.total_defects,
                        quality_score=board_summary.quality_score,
                        confidence=max((d.confidence for d in dets), default=0.0),
                        class_counts=dict(board_summary.class_counts),
                    ))
                records.append(
                    FrameRecord(
                        frame_index=frames_read,
                        timestamp_s=timestamp,
                        verdict=summary.verdict,
                        defect_count=summary.total_defects,
                        quality_score=summary.quality_score,
                        class_counts=dict(summary.class_counts),
                        boards=board_records,
                    )
                )

                annotated = draw_detections(stages.final, detection_result.detections)
                if per_board and real_boxes:
                    annotated = draw_board_labels(
                        annotated,
                        [(box, f"Board {board_id + 1}")
                         for (board_id, _dets), box in zip(per_board, board_boxes)],
                    )
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
