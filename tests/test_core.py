"""
test_core.py
============
Verification suite for the Module 4 backend.

The interface itself is exercised by running it, but the logic behind it — the
acceptance rules, the statistics, the rendering and the report builders — is
verified here without a running Streamlit server, so a regression is caught in
seconds rather than by clicking through the application.

Run with::

    python tests/test_core.py
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

_APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_APP_DIR))

from core import (                                                    # noqa: E402
    analysis,
    live,
    pipeline_remote,
    remote,
    report,
    storage,
    video,
    viz,
    workspace,
)
from core.analysis import InspectionCriteria                        # noqa: E402
from core.detector import (                                          # noqa: E402
    DEFAULT_CLASS_NAMES,
    DefectDetector,
    Detection,
    DetectionResult,
    discover_weights,
    read_class_names_from_yaml,
)
from core.pipeline_bridge import (                                   # noqa: E402
    MODE_MODULE,
    PipelineBridge,
    create_pipeline,
    decode_image,
    find_project_root,
    list_images,
)

_PASSED = 0
_FAILED: list[str] = []


def check(condition: bool, label: str) -> None:
    """Record one assertion without aborting the rest of the suite."""
    global _PASSED
    if condition:
        _PASSED += 1
        print(f"  ok   {label}")
    else:
        _FAILED.append(label)
        print(f"  FAIL {label}")


def synthetic_board(width: int = 512, height: int = 512) -> np.ndarray:
    """
    Build a PCB-like test image: a green board on a dark background, with copper
    tracks and drilled holes, so alignment has a real quadrilateral to find.
    """
    image = np.full((height, width, 3), 25, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (width - 40, height - 40), (40, 120, 40), cv2.FILLED)
    for y in range(90, height - 90, 55):
        cv2.line(image, (70, y), (width - 70, y), (30, 170, 210), 4)
    for x in range(90, width - 90, 70):
        cv2.circle(image, (x, height // 2), 9, (15, 15, 15), cv2.FILLED)
    noise = np.random.default_rng(7).normal(0, 6, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def fake_detections() -> list[Detection]:
    """A representative mixture: confident, uncertain, and critical."""
    return [
        Detection(0, "missing_hole", 0.91, 100, 120, 128, 148),
        Detection(4, "spur", 0.42, 200, 210, 222, 232),
        Detection(2, "open_circuit", 0.77, 320, 260, 352, 292),
    ]


# --------------------------------------------------------------------------- #
def test_analysis() -> None:
    print("\n[analysis] verdicts, scores and aggregation")

    empty = DetectionResult([], 12.0, (512, 512), "test.pt")
    clean = analysis.summarise(empty)
    check(clean.verdict == analysis.VERDICT_PASS, "a board with no detections passes")
    check(clean.quality_score == 100.0, "a clean board scores 100")

    result = DetectionResult(fake_detections(), 30.0, (512, 512), "test.pt")
    strict = analysis.summarise(result, InspectionCriteria(max_defects=0))
    check(strict.verdict == analysis.VERDICT_FAIL, "an open circuit fails the board")
    check(strict.critical_defects == 1, "the critical defect is counted once")
    check(strict.uncertain_defects == 1, "the 0.42 finding is counted as uncertain")
    check(strict.total_defects == 3, "all three findings are counted")
    check(0.0 <= strict.quality_score < 100.0, "a defective board scores below 100")

    # With no critical classes configured and a generous allowance, the same
    # findings should be sent for review rather than failed.
    lenient = analysis.summarise(
        result, InspectionCriteria(max_defects=5, critical_classes=())
    )
    check(lenient.verdict == analysis.VERDICT_REVIEW,
          "findings within the allowance are flagged for review, not failed")

    borderline = DetectionResult(
        [Detection(4, "spur", 0.31, 10, 10, 30, 30)], 5.0, (512, 512), "test.pt"
    )
    review = analysis.summarise(borderline, InspectionCriteria(max_defects=0,
                                                              critical_classes=()))
    check(review.verdict == analysis.VERDICT_REVIEW,
          "a lone low-confidence finding goes to review rather than failing")

    frame = analysis.detections_dataframe(fake_detections())
    check(len(frame) == 3, "the findings table has one row per detection")
    check("defect_type" in frame.columns, "the findings table names the defect type")
    check(analysis.detections_dataframe([]).empty,
          "an empty findings table still carries its columns")

    batch = analysis.summarise_batch([clean, strict, lenient])
    check(batch.total_boards == 3, "the batch counts every board")
    check(batch.passed == 1, "the batch counts the passing board")
    check(batch.failed == 1, "the batch counts the failing board")
    check(abs(batch.yield_pct - 100 / 3) < 0.01, "the yield is computed correctly")
    check(analysis.summarise_batch([]).total_boards == 0,
          "an empty batch summarises without raising")

    distribution = analysis.class_distribution([strict, lenient])
    check(not distribution.empty, "the class distribution is populated")
    check(abs(distribution["share_pct"].sum() - 100.0) < 0.5,
          "the distribution shares add up to 100 per cent")


def test_viz() -> None:
    print("\n[viz] annotation and encoding")

    board = synthetic_board()
    annotated = viz.draw_detections(board, fake_detections())
    check(annotated.shape == board.shape, "annotation preserves the image shape")
    check(not np.array_equal(annotated, board), "annotation actually draws something")
    check(np.array_equal(board, synthetic_board()), "the source image is not mutated")

    banner = viz.draw_verdict_banner(annotated, "FAIL", "3 defects")
    check(banner.shape == annotated.shape, "the verdict banner preserves the shape")

    check(viz.encode_jpeg(board) is not None, "JPEG encoding succeeds")
    check(viz.encode_png(board) is not None, "PNG encoding succeeds")
    check(viz.to_rgb(board).shape == board.shape, "BGR to RGB conversion keeps the shape")

    small = viz.thumbnail(board, 128)
    check(max(small.shape[:2]) == 128, "thumbnails are scaled to the requested size")
    check(viz.thumbnail(small, 512).shape == small.shape,
          "an image smaller than the target is returned unchanged")

    check(viz.colour_for_hex("short").startswith("#"), "colours convert to hex")
    check(viz.colour_for("missing_hole") != viz.colour_for("short"),
          "different classes get different colours")
    check(len(viz.legend_entries(fake_detections())) == 3,
          "the legend lists every class present")


def test_pipeline_local() -> None:
    """
    The local adapter, which is the fallback path.

    This repository ships no copy of ``image_pipeline.py``, so on a clean
    checkout the adapter is *expected* to be unavailable. What is verified here
    is that it says so accurately and still returns a usable image — the
    property the interface depends on whenever an upstream module is missing.
    """
    print("\n[pipeline_bridge] Modules 1 and 2 — local adapter")

    board = synthetic_board()
    bridge = PipelineBridge()
    status = bridge.status()

    check(status["source"] == "local", "the adapter identifies its source")
    check(status["available"] == bridge.available, "status and property agree on availability")
    check(bridge.available or bool(bridge.load_error), "an unavailable adapter explains why")

    stages = bridge.run(board, mode=MODE_MODULE, do_preprocess=True, do_align=True)
    check(stages.final is not None, "the adapter always returns a final image")
    check(np.array_equal(stages.original, board), "the original image is carried through")

    if bridge.available:
        check(status["module1_ready"], "Module 1's preprocess_image resolves")
        check(status["module2_ready"], "Module 2's align_image resolves")
        check(stages.preprocess_ok, "Module 1 produced an output")

        # A blank image has no board boundary: Module 2 must degrade, not fail.
        blank = np.full((256, 256, 3), 128, dtype=np.uint8)
        degraded = bridge.run(blank, do_preprocess=True, do_align=True)
        check(degraded.final is not None, "an un-alignable image still yields a final image")
        check(any("Module 2" in note for note in degraded.notes) or degraded.align_ok,
              "a failed alignment is explained in the notes")
    else:
        check(np.array_equal(stages.final, board),
              "with no local contract, the original image is passed through unchanged")
        check(bool(stages.notes), "the degradation is explained in the notes")

    bypassed = bridge.run(board, do_preprocess=False, do_align=False)
    check(np.array_equal(bypassed.final, board), "bypassing both modules returns the input")

    encoded = viz.encode_jpeg(board)
    check(decode_image(encoded) is not None, "uploaded bytes decode to an image")
    check(decode_image(b"not an image") is None, "invalid bytes decode to None")
    check(decode_image(b"") is None, "empty bytes decode to None")


def test_workspace() -> None:
    print("\n[workspace] the optional data folder")

    check(workspace.repo_root().is_dir(), "the repository root resolves")
    check(workspace.workspace_root(None) is not None,
          "the workspace always resolves to something")
    check(workspace.dataset_folders(None) == {}, "no workspace yields no datasets")

    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        (base / "Preprocessed_Dataset").mkdir()
        (base / "PCB_DATASET").mkdir()
        found = workspace.dataset_folders(base)
        check(len(found) == 2, f"the dataset folders present are found ({len(found)})")
        check(all(path.is_dir() for path in found.values()), "every reported folder exists")
        check(workspace.workspace_root(base) == base, "an explicitly configured folder wins")
        check(workspace.workspace_root(base / "nope") != base / "nope",
              "a configured folder that does not exist falls back rather than breaking")


def test_detector() -> None:
    print("\n[detector] loading, degradation and class names")

    missing = DefectDetector("/nonexistent/best.pt")
    check(not missing.available, "a missing checkpoint leaves the detector unloaded")
    check(missing.load_error is not None, "a missing checkpoint reports an error")

    unset = DefectDetector(None)
    check(not unset.available, "no checkpoint at all leaves the detector unloaded")

    board = synthetic_board()
    result = unset.predict(board)
    check(result.count == 0, "an unloaded detector returns no detections")
    check(result.error is not None, "an unloaded detector explains itself")
    check(result.image_shape == (512, 512), "the image shape is still reported")

    check(unset._name_for(0) == DEFAULT_CLASS_NAMES[0], "class 0 maps to the first name")
    check(unset._name_for(99) == "class_99", "an out-of-range class id degrades gracefully")

    check(discover_weights(None) == [], "weight discovery on no root returns nothing")
    check(isinstance(discover_weights(find_project_root()), list),
          "weight discovery returns a list")

    # data.yaml parsing, using a temporary copy of Module 3's real format.
    with tempfile.TemporaryDirectory() as temp:
        yaml_dir = Path(temp) / "Student3-Defect Detection" / "dataset" / "yolo"
        yaml_dir.mkdir(parents=True)
        (yaml_dir / "data.yaml").write_text(
            "path: /somewhere\ntrain: images/train\nnc: 6\nnames:\n"
            "  0: missing_hole\n  1: mouse_bite\n  2: open_circuit\n"
            "  3: short\n  4: spur\n  5: spurious_copper\n",
            encoding="utf-8",
        )
        names = read_class_names_from_yaml(Path(temp))
        check(names == DEFAULT_CLASS_NAMES, "data.yaml is parsed into the class order")
    check(read_class_names_from_yaml(None) is None, "no root yields no class names")

    detection = Detection(0, "missing_hole", 0.9, 10, 20, 40, 60)
    check(detection.width == 30 and detection.height == 40, "box dimensions are computed")
    check(detection.area == 1200, "box area is computed")
    check(detection.centre == (25, 40), "box centre is computed")
    check(detection.as_row()["defect_type"] == "missing_hole", "a detection flattens to a row")


def test_report() -> None:
    print("\n[report] PDF generation")

    board = synthetic_board()
    detections = fake_detections()
    annotated = viz.draw_detections(board, detections)
    summary = analysis.summarise(DetectionResult(detections, 41.0, (512, 512), "best.pt"))
    config = {"Detector weights": "best.pt", "Confidence threshold": "0.25",
              "Module 1 (pre-processing)": "enabled", "Module 2 (alignment)": "enabled"}

    single = report.build_single_report(
        annotated, summary, detections, config, "board_01.jpg", original=board
    )
    check(single.startswith(b"%PDF"), "the single-board report is a valid PDF")
    check(len(single) > 20_000, f"the single-board report embeds its images ({len(single)} bytes)")
    Path("/tmp/report_single.pdf").write_bytes(single)

    empty_summary = analysis.summarise(DetectionResult([], 8.0, (512, 512), "best.pt"))
    clean = report.build_single_report(board, empty_summary, [], config, "board_02.jpg")
    check(clean.startswith(b"%PDF"), "a clean board still produces a report")

    summaries = [summary, empty_summary, summary]
    batch = analysis.summarise_batch(summaries)
    rows = [s.as_row(f"board_{i:02d}.jpg") for i, s in enumerate(summaries, start=1)]
    batch_pdf = report.build_batch_report(
        batch, rows, config, "Preprocessed_Dataset",
        gallery=[("board_01.jpg — FAIL", annotated)],
    )
    check(batch_pdf.startswith(b"%PDF"), "the batch report is a valid PDF")
    check(len(batch_pdf) > 10_000, f"the batch report has content ({len(batch_pdf)} bytes)")
    Path("/tmp/report_batch.pdf").write_bytes(batch_pdf)

    empty_batch = report.build_batch_report(
        analysis.summarise_batch([]), [], config, "Empty run"
    )
    check(empty_batch.startswith(b"%PDF"), "an empty batch still produces a report")


def test_video() -> None:
    print("\n[video] frame-by-frame processing")

    # Build a short synthetic clip: a board drifting across the frame.
    path = Path(tempfile.mktemp(suffix=".mp4"))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    board = cv2.resize(synthetic_board(), (320, 240))
    for offset in range(20):
        frame = np.roll(board, offset * 4, axis=1)
        writer.write(frame)
    writer.release()
    clip = path.read_bytes()
    path.unlink(missing_ok=True)

    info = video.probe(clip)
    check(info["frame_count"] > 0, f"probe reads the frame count ({info['frame_count']})")
    check(info["width"] == 320 and info["height"] == 240, "probe reads the resolution")
    check(video.probe(b"")["frame_count"] == 0, "probing empty bytes is safe")

    bridge = PipelineBridge()
    detector = DefectDetector(None)          # unloaded: exercises the degraded path
    result = video.process_video(
        clip, bridge=bridge, detector=detector,
        do_preprocess=True, do_align=False,
        frame_stride=4, max_frames=5,
    )
    check(result.error is None, f"video processing completes ({result.error})")
    check(result.video_bytes is not None, "an annotated video is produced")
    check(result.frames_analysed > 0, f"frames were analysed ({result.frames_analysed})")
    check(result.frames_read >= result.frames_analysed, "more frames were read than analysed")
    check(len(result.records) == result.frames_analysed, "one record per analysed frame")
    check(all(r.verdict in ("PASS", "REVIEW", "FAIL") for r in result.records),
          "every frame carries a verdict")

    bad = video.process_video(b"", bridge=bridge, detector=detector)
    check(bad.error is not None, "empty video data is reported, not raised")
    check(not bad.ok, "an errored run is not marked ok")


def test_file_discovery() -> None:
    print("\n[discovery] folders and images")

    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        (base / "class_a").mkdir()
        cv2.imwrite(str(base / "class_a" / "one.jpg"), synthetic_board(64, 64))
        cv2.imwrite(str(base / "two.png"), synthetic_board(64, 64))
        (base / "notes.txt").write_text("ignored", encoding="utf-8")

        check(len(list_images(base, recursive=True)) == 2, "images are found recursively")
        check(len(list_images(base, recursive=False)) == 1, "a flat scan skips sub-folders")
        check(len(list_images(base, limit=1)) == 1, "the limit is honoured")
        check(list_images(base / "missing") == [], "a missing folder yields no images")


# --------------------------------------------------------------------------- #
# Remote detection: a stub inference service, exercised over a real socket
# --------------------------------------------------------------------------- #
class _StubServiceHandler(BaseHTTPRequestHandler):
    """
    Minimal implementation of ``docs/API_CONTRACT.md``.

    Running a real HTTP server, rather than monkey-patching ``requests``, is what
    makes this test meaningful: it exercises the multipart encoding, the status
    handling and the JSON parsing exactly as the live client will.
    """

    classes = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

    def log_message(self, *args) -> None:                     # silence the test output
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            self._json(200, {"status": "ok", "model": "stub-best.pt", "classes": self.classes})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:
        if not self.path.startswith("/predict"):
            self._json(404, {"detail": "not found"})
            return

        # Drain the multipart body so the client's write always completes.
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        if b"board.jpg" not in body:
            self._json(400, {"detail": "the image part was missing"})
            return

        self._json(200, {
            "model": "stub-best.pt",
            "inference_ms": 12.5,
            "detections": [
                {"class_id": 0, "class_name": "missing_hole",
                 "confidence": 0.88, "bbox": [10, 20, 40, 55]},
                # Given without a class_name, to prove the client falls back to
                # the class order reported by /health.
                {"class_id": 3, "confidence": 0.64, "x1": 100, "y1": 110, "x2": 130, "y2": 145},
            ],
        })

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _StubPipelineHandler(BaseHTTPRequestHandler):
    """
    Minimal implementation of ``docs/API_CONTRACT_PIPELINE.md``.

    It returns a *recognisable* pre-processed image — a solid colour — so the
    test can prove the client really adopted what the service sent rather than
    quietly falling back to the original. Alignment always fails, which is the
    degradation path that matters most.
    """

    #: BGR value of the "pre-processed" image the stub returns.
    PREPROCESSED_COLOUR = (17, 34, 51)

    def log_message(self, *args) -> None:                     # silence the test output
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/health"):
            self._json(200, {
                "status": "ok",
                "service": "stub-pipeline",
                "modules": {"preprocess": True, "align": True},
                "modes": ["module"],
            })
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self) -> None:
        if not self.path.startswith("/process"):
            self._json(404, {"detail": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        if b"board.jpg" not in body:
            self._json(400, {"detail": "the image part was missing"})
            return

        wants_preprocess = b'name="preprocess"\r\n\r\ntrue' in body
        flat = np.zeros((64, 64, 3), dtype=np.uint8)
        flat[:, :] = self.PREPROCESSED_COLOUR
        encoded = base64.b64encode(viz.encode_jpeg(flat, quality=100)).decode("ascii")

        self._json(200, {
            "preprocessed": encoded if wants_preprocess else None,
            # Alignment declines, which the contract calls a valid outcome.
            "aligned": None,
            "final": encoded if wants_preprocess else None,
            "notes": ["No four-corner board boundary was found."],
            "elapsed_ms": 7.5,
            "mode": "module",
        })

    def _json(self, code: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_pipeline_remote() -> None:
    print("\n[pipeline_remote] Modules 1 and 2 over HTTP")

    board = synthetic_board()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubPipelineHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        pipeline = pipeline_remote.RemotePipeline(base_url, timeout=10.0)
        status = pipeline.status()
        check(pipeline.available, f"the client connects to the service ({pipeline.load_error})")
        check(status["source"] == "remote", "the adapter identifies its source")
        check(status["service"] == "stub-pipeline", "the service name is read from /health")
        check(status["module1_ready"] and status["module2_ready"],
              "both stages are reported as offered")
        check(not status["notebook_mode_ready"],
              "a service that offers no notebook mode is reported honestly")

        stages = pipeline.run(board, do_preprocess=True, do_align=True)
        check(stages.final is not None, "a round trip yields a final image")
        check(stages.preprocess_ok, "Module 1's image is decoded from the response")
        colour = tuple(int(v) for v in stages.preprocessed[0, 0])
        check(max(abs(a - b) for a, b in
                  zip(colour, _StubPipelineHandler.PREPROCESSED_COLOUR)) <= 2,
              f"the decoded image is the one the service sent ({colour})")
        check(not stages.align_ok, "a null 'aligned' field is read as 'no boundary found'")
        check(np.array_equal(stages.original, board), "the original image is carried through")
        check(abs(stages.elapsed_ms - 7.5) < 0.01, "the service's own timing is reported")
        check(any("boundary" in note for note in stages.notes),
              "the service's own notes reach the operator")
        check(any("Module 2" in note for note in stages.notes),
              "the client adds its own note for the failed stage")

        bypassed = pipeline.run(board, do_preprocess=False, do_align=False)
        check(np.array_equal(bypassed.final, board),
              "bypassing both stages skips the request entirely")

        # An endpoint that exists but is not the service root.
        wrong = pipeline_remote.RemotePipeline(f"{base_url}/process", timeout=5.0)
        check(not wrong.available, "a URL pointing at an endpoint is rejected")
        check("HTTP 404" in (wrong.load_error or ""), "the rejection names the status code")
    finally:
        server.shutdown()
        server.server_close()

    unreachable = pipeline_remote.RemotePipeline("http://127.0.0.1:9", timeout=2.0)
    check(not unreachable.available, "an unreachable service leaves the adapter unavailable")
    check("Could not connect" in (unreachable.load_error or ""),
          "an unreachable service produces an actionable message")

    degraded = unreachable.run(board, do_preprocess=True, do_align=True)
    check(np.array_equal(degraded.final, board),
          "an unreachable service degrades to the original image rather than raising")
    check(bool(degraded.notes), "the degradation is explained in the notes")

    check(not pipeline_remote.RemotePipeline("").available, "no URL leaves the adapter unavailable")
    check(not pipeline_remote.RemotePipeline("127.0.0.1:8100").available,
          "a URL without a scheme is rejected")

    # The factory must hand back the adapter the sidebar asked for.
    check(isinstance(create_pipeline("local"), PipelineBridge),
          "the factory builds the local adapter on request")
    check(isinstance(create_pipeline("remote", base_url="http://127.0.0.1:9"),
                     pipeline_remote.RemotePipeline),
          "the factory builds the remote adapter by default")


def test_remote_detector() -> None:
    print("\n[remote] detection over HTTP")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubServiceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        detector = remote.RemoteDetector(base_url, timeout=10.0)
        check(detector.available, f"the client connects to the service ({detector.load_error})")
        check(detector.remote_model == "stub-best.pt", "the model name is read from /health")
        check(detector.class_names == DEFAULT_CLASS_NAMES,
              "the class order is adopted from /health")

        result = detector.predict(synthetic_board(), confidence=0.25)
        check(result.error is None, f"a prediction round trip succeeds ({result.error})")
        check(result.count == 2, f"both detections are parsed ({result.count})")
        check(result.inference_ms == 12.5, "the service's own timing is reported")
        check(result.detections[0].class_name == "missing_hole",
              "an explicit class_name is used as given")
        check(result.detections[1].class_name == "short",
              "a missing class_name falls back to the class order")
        check(result.detections[0].confidence > result.detections[1].confidence,
              "detections are sorted by descending confidence")
        check(result.detections[1].x1 == 100 and result.detections[1].y2 == 145,
              "separate x1/y1/x2/y2 fields are accepted")

        # An endpoint that exists but is not the service root.
        wrong = remote.RemoteDetector(f"{base_url}/predict", timeout=5.0)
        check(not wrong.available, "a URL pointing at an endpoint is rejected")
        check("HTTP 404" in (wrong.load_error or ""), "the rejection names the status code")
    finally:
        server.shutdown()
        server.server_close()

    unreachable = remote.RemoteDetector("http://127.0.0.1:9", timeout=2.0)
    check(not unreachable.available, "an unreachable service leaves the detector unloaded")
    check("Could not connect" in (unreachable.load_error or ""),
          "an unreachable service produces an actionable message")

    empty = remote.RemoteDetector("")
    check(not empty.available, "no URL leaves the detector unloaded")
    bad_scheme = remote.RemoteDetector("127.0.0.1:8000")
    check(not bad_scheme.available, "a URL without a scheme is rejected")

    degraded = unreachable.predict(synthetic_board())
    check(degraded.count == 0 and degraded.error is not None,
          "predicting through an unreachable service degrades rather than raising")

    # Coordinate-format handling, checked directly on the parser.
    client = remote.RemoteDetector("")
    client.class_names = DEFAULT_CLASS_NAMES
    xywh, error = client._parse(
        {"format": "xywh",
         "detections": [{"class_id": 1, "confidence": 0.5, "bbox": [10, 20, 30, 40]}]},
        max_detections=10,
    )
    check(error is None and len(xywh) == 1, "an xywh response parses")
    check(xywh[0].x2 == 40 and xywh[0].y2 == 60, "xywh boxes are converted to xyxy")

    _, missing = client._parse({"model": "x"}, max_detections=10)
    check(missing is not None, "a response with no detections field is reported")


# --------------------------------------------------------------------------- #
def test_storage() -> None:
    print("\n[storage] inspection history")

    disabled = storage.create_store("none")
    check(not disabled.available, "the null store reports itself unavailable")
    check(disabled.fetch() == [], "the null store returns no rows")
    check(disabled.log(storage.InspectionRecord("a", "single", "PASS", 100.0, 0)) is False,
          "logging to the null store is a no-op")

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "nested" / "history.db"
        store = storage.create_store("sqlite", sqlite_path=path)
        check(store.available, f"the SQLite store initialises ({store.last_error})")
        check(path.is_file(), "the database file is created, including its parent folder")

        summary = analysis.summarise(
            DetectionResult(fake_detections(), 31.0, (512, 512), "best.pt")
        )
        record = storage.InspectionRecord.from_summary(
            summary, source="board_01.jpg", mode="single", model="best.pt"
        )
        check(record.verdict == summary.verdict, "a record carries the summary's verdict")
        check(record.image_width == 512, "a record carries the image dimensions")
        check(record.inspected_at != "", "a record timestamps itself")
        check(record.class_counts == summary.class_counts, "a record carries the class counts")

        check(store.log(record) is True, "one record is written")
        check(store.count() == 1, "the row count reflects the write")

        many = [
            storage.InspectionRecord.from_summary(summary, f"board_{i:02d}.jpg", "batch", "best.pt")
            for i in range(5)
        ]
        check(store.log_many(many) == 5, "a batch of records is written in one call")
        check(store.count() == 6, "the row count reflects the batch")

        rows = store.fetch(limit=10)
        check(len(rows) == 6, "every row is read back")
        check(isinstance(rows[0]["class_counts"], dict),
              "class_counts is deserialised back into a dictionary")
        check(rows[0]["source"].startswith("board_"), "the source name survives the round trip")
        check(rows[0]["mode"] in ("single", "batch"), "the mode survives the round trip")
        check(len(store.fetch(limit=2)) == 2, "the fetch limit is honoured")

        status = store.status()
        check(status["kind"] == "sqlite" and status["available"], "the store reports its state")

        check(store.clear() is True, "the history clears")
        check(store.count() == 0, "the store is empty after clearing")

        # A second store on the same file must not fail on the existing schema.
        again = storage.create_store("sqlite", sqlite_path=path)
        check(again.available, "re-opening an existing database succeeds")

    unwritable = storage.create_store("sqlite", sqlite_path="/proc/nope/history.db")
    check(not unwritable.available, "an unwritable path degrades rather than raising")
    check(unwritable.log_many([]) == 0, "logging to a failed store is safe")

    supabase = storage.create_store("supabase", supabase_url="", supabase_key="")
    check(not supabase.available, "Supabase without credentials is unavailable")
    check(supabase.last_error is not None, "Supabase explains why it is unavailable")


# --------------------------------------------------------------------------- #
class _StubCapture:
    """A camera stand-in: hands out a fixed number of frames, then stops."""

    def __init__(self, frames: int, frame: np.ndarray) -> None:
        self.remaining = frames
        self.frame = frame
        self.released = False

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


def test_live() -> None:
    print("\n[live] real-time capture")

    stats = live.LiveStats()
    check(stats.frames == 0, "a new session starts empty")
    check(stats.effective_fps == 0.0, "an empty session reports no frame rate")
    check(stats.pass_rate == 0.0, "an empty session reports no pass rate")

    clean = analysis.summarise(DetectionResult([], 10.0, (240, 320), "m"))
    defective = analysis.summarise(DetectionResult(fake_detections(), 20.0, (240, 320), "m"))
    board = cv2.resize(synthetic_board(), (320, 240))

    stats.record(clean, board)
    stats.record(defective, board)
    check(stats.frames == 2, "each inspected frame is counted")
    check(stats.defects == 3, "defects accumulate across frames")
    check(stats.verdict_counts.get("PASS") == 1, "verdicts are tallied")
    check(stats.class_counts.get("missing_hole") == 1, "class counts accumulate")
    check(len(stats.timeline) == 2, "the timeline gains one entry per frame")
    check(abs(stats.pass_rate - 50.0) < 0.01, "the pass rate is computed")
    check(abs(stats.mean_inference_ms - 15.0) < 0.01, "mean inference time is computed")
    check(stats.worst_frame is not None, "the worst frame is retained")

    # The timeline must not grow without bound over a long session.
    for _ in range(live.TIMELINE_LIMIT + 50):
        stats.record(clean, None)
    check(len(stats.timeline) == live.TIMELINE_LIMIT, "the timeline is capped in length")

    bridge = PipelineBridge()
    detector = DefectDetector(None)          # unloaded: exercises the degraded path
    session = live.LiveStats()
    seen: list[str] = []

    chunk = live.run_chunk(
        _StubCapture(frames=6, frame=board),
        bridge=bridge,
        detector=detector,
        stats=session,
        seconds=1.5,
        target_fps=10.0,
        do_preprocess=True,
        do_align=False,
        on_frame=lambda annotated, summary: seen.append(summary.verdict),
    )
    check(chunk.frames_processed > 0, f"the chunk inspects frames ({chunk.frames_processed})")
    check(chunk.last_annotated is not None, "the chunk returns an annotated frame")
    check(len(seen) == chunk.frames_processed, "the frame callback fires once per frame")
    check(session.frames == chunk.frames_processed, "the session totals match the chunk")

    exhausted = live.run_chunk(
        _StubCapture(frames=0, frame=board),
        bridge=bridge, detector=detector, stats=live.LiveStats(),
        seconds=1.0, target_fps=10.0,
    )
    check(exhausted.error is not None, "a camera that stops delivering frames is reported")
    check(exhausted.frames_processed == 0, "no frames are counted when capture fails")

    check(live.list_cameras(0) == [], "probing zero indices finds nothing")
    check(live.open_camera(999) is None, "opening a non-existent camera returns None")
    live.close_camera(None)                  # must not raise
    check(True, "closing a null camera handle is safe")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 68)
    print("Module 4 — backend verification suite")
    print("=" * 68)

    for suite in (
        test_analysis,
        test_viz,
        test_workspace,
        test_pipeline_local,
        test_pipeline_remote,
        test_detector,
        test_report,
        test_video,
        test_file_discovery,
        test_remote_detector,
        test_storage,
        test_live,
    ):
        suite()

    print("\n" + "=" * 68)
    print(f"{_PASSED} check(s) passed, {len(_FAILED)} failed")
    for label in _FAILED:
        print(f"  FAILED: {label}")
    print("=" * 68)
    return 1 if _FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
