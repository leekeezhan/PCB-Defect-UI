"""
app.py
======
Module 4 — User Interface Module
PCB Defect Inspection System (Mode B, Innovative Solution Development)

This is the entry point of the inspection application. It is the only file the
operator runs::

    streamlit run app.py

Responsibilities
----------------
The interface owns presentation, orchestration, persistence and reporting. It
owns no image processing whatsoever: every algorithmic step is delegated through
the adapters in ``core`` to a teammate's module — normally over HTTP, because
this repository holds Module 4 alone and the other three are developed and
deployed by their own owners.

    Module 1  Image acquisition & pre-processing   core.pipeline_remote (API)
    Module 2  Image alignment & calibration        core.pipeline_bridge (local)
    Module 3  PCB defect detection                 core.remote          (API)
                                                   core.detector        (local)
    Module 4  User interface (this file)           core.analysis / report
                                                   core.video / live / storage

Each upstream boundary defaults to the service and can be switched to a local
implementation in the sidebar, which is what makes a demonstration possible when
a teammate's service is not running. The two wire formats are specified in
``docs/API_CONTRACT_PIPELINE.md`` and ``docs/API_CONTRACT.md``.

Pages
-----
    Single inspection : one board, every intermediate stage shown, PDF certificate.
    Batch inspection  : a folder or a multi-file upload, yield statistics, CSV + PDF.
    Video inspection  : frame-by-frame analysis of a recorded stream.
    Live inspection   : a camera attached to this machine, inspected in real time.
    History           : the yield dashboard, over every board ever inspected.
    System status     : which modules resolved, which weights loaded, and why not.

Design notes
------------
* Streamlit re-runs this script top to bottom on every interaction, including on
  a download-button click. Results are therefore held in ``st.session_state`` so
  that downloading a report never silently re-runs an inspection.
* The live page's capture loop runs in short chunks and asks for a re-run
  between them, because a loop that never returns would leave the Stop button
  unclickable. See ``core.live`` for the reasoning.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Make ``core`` and ``ui`` importable regardless of the shell's working directory.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import pandas as pd
import streamlit as st

from core import analysis, live, pcb_check, report, roi, storage, video, viz
from core.analysis import BatchSummary, InspectionSummary
from core.detector import DefectDetector, DetectionResult, filter_by_class
from core.pipeline_bridge import (
    create_pipeline,
    decode_image,
    list_images,
    read_image,
)
from core.remote import RemoteDetector
from core.storage import InspectionRecord
from ui.components import (
    class_bar_chart,
    defect_legend,
    download_row,
    metric_row,
    stage_chips,
    status_row,
    verdict_banner,
)
from ui.sidebar import SOURCE_REMOTE, Settings, render_sidebar
from ui.theme import inject_css, render_header, section

APP_TITLE = "PCB Defect Inspection System"

#: Either Modules 1 & 2 adapter — ``core.pipeline_remote.RemotePipeline``
#: (the default) or ``core.pipeline_bridge.PipelineBridge``. They expose the
#: same surface, so no page needs to know which one it is holding.
Pipeline = Any


# --------------------------------------------------------------------------- #
# Cached backends
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Connecting to the processing service…")
def get_pipeline(
    source: str,
    base_url: str,
    api_key: str,
    timeout: float,
    verify_tls: bool,
    workspace: str,
) -> Any:
    """
    Build the Module 1 / Module 2 adapter once per configuration.

    Cached because both adapters do work up front that should not be repeated on
    every script re-run: the remote one performs a health check, and the local
    one parses a multi-megabyte notebook the first time notebook mode is used.
    The arguments are plain scalars so Streamlit can hash them — changing any of
    them in the sidebar builds a new adapter rather than reusing the cached one.
    """
    return create_pipeline(
        source,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        verify_tls=verify_tls,
        workspace=workspace,
    )


def _pipeline_for(settings: Settings) -> Any:
    """Return the Modules 1 & 2 adapter this run should use."""
    return get_pipeline(
        settings.pipeline_source,
        settings.pipeline_url,
        settings.pipeline_key,
        settings.pipeline_timeout,
        settings.pipeline_verify_tls,
        settings.workspace,
    )


@st.cache_resource(show_spinner="Loading the detection model…")
def get_local_detector(weights: str | None, device: str | None) -> DefectDetector:
    """
    Load a Module 3 checkpoint once per (weights, device) combination.

    Args are plain strings so Streamlit can hash them; changing either in the
    sidebar loads a new model rather than reusing the cached one.
    """
    return DefectDetector(weights, device=device)


@st.cache_resource(show_spinner="Connecting to the inference service…")
def get_remote_detector(
    base_url: str, api_key: str, timeout: float, verify_tls: bool
) -> RemoteDetector:
    """Connect to the inference service once per configuration."""
    return RemoteDetector(base_url, api_key=api_key, timeout=timeout, verify_tls=verify_tls)


@st.cache_resource(show_spinner=False)
def get_store(kind: str, sqlite_path: str, url: str, key: str, table: str):
    """Open the configured inspection-history store once per configuration."""
    return storage.create_store(
        kind, sqlite_path=sqlite_path, supabase_url=url, supabase_key=key, table=table
    )


def _detector_for(settings: Settings, fast: bool = False):
    """
    Return the detector this page should use.

    Args:
        settings: the sidebar configuration.
        fast: request the lighter checkpoint configured for the video and live
            pages. Ignored when detection runs remotely, since the service picks
            its own model.
    """
    if settings.uses_remote:
        return get_remote_detector(
            settings.remote_url,
            settings.remote_key,
            settings.remote_timeout,
            settings.remote_verify_tls,
        )
    path = settings.fast_weights_path if (fast and settings.has_fast_model) else settings.weights_path
    return get_local_detector(str(path) if path else None, settings.device)


def _store_for(settings: Settings):
    return get_store(
        settings.store_kind,
        settings.sqlite_path,
        settings.supabase_url,
        settings.supabase_key,
        settings.supabase_table,
    )


# --------------------------------------------------------------------------- #
# Shared inspection routine
# --------------------------------------------------------------------------- #
def inspect(image, bridge: Pipeline, detector, settings: Settings, do_align: bool | None = None):
    """
    Run one board through the whole system.

    Args:
        image: BGR array to inspect.
        bridge: Module 1 / Module 2 adapter.
        detector: Module 3 adapter — local or remote; only ``predict`` is used.
        settings: the sidebar configuration.
        do_align: override ``settings.do_align`` for this call only. ``None``
            (the default) uses the sidebar setting, matching every existing
            call site. Passed explicitly by the video page's single-frame
            tool, which crops out one board at a time and always wants
            alignment attempted on that crop — a lone, roughly-cropped board
            resolves a four-corner boundary far more reliably than the full,
            multi-board frame it came from.

    Returns:
        ``(stages, detection_result, summary, annotated)`` — every artefact the
        pages need, computed exactly once.
    """
    stages = bridge.run(
        image,
        mode=settings.mode,
        do_preprocess=settings.do_preprocess,
        do_align=settings.do_align if do_align is None else do_align,
    )

    # Student 1's validate_pcb_image() alone (StageResult.pcb_valid) only
    # checks for a plausibly-sized, plausibly-shaped saturated blob — this
    # repository's own pcb_check.evaluate() adds a second, independent
    # opinion (dominant-hue concentration + edge/texture density) on top of
    # it, run on the ORIGINAL upload. Either one rejecting is enough.
    pcb_valid, pcb_message = pcb_check.evaluate(
        stages.pcb_valid, stages.pcb_message, stages.original
    )
    if pcb_valid is False:
        # Rejected before Modules 1-3 ran any work on it. Reported as an
        # outright FAIL via analysis.rejected() rather than being sent to the
        # detector, so an arbitrary non-PCB photo can no longer report PASS
        # simply because the detector found nothing it recognises.
        image_shape = stages.final.shape[:2] if stages.final is not None else (0, 0)
        detection_result = DetectionResult(
            detections=[], inference_ms=0.0, image_shape=image_shape,
            model_name=detector.model_name, error=None,
        )
        summary = analysis.rejected(
            pcb_message or "Uploaded image does not appear to contain a PCB.",
            image_shape=image_shape,
        )
        annotated = stages.final
        return stages, detection_result, summary, annotated

    detection_result = detector.predict(
        stages.final, confidence=settings.confidence, iou=settings.iou
    )
    # A defect type the operator unchecked in "Defect types to show" is
    # dropped here, before the verdict, the score, the image or the table
    # ever see it — filtering only the display would leave a hidden type
    # still able to fail the board, which would be confusing. An empty
    # selection is honoured literally too: it means "show nothing".
    detection_result = filter_by_class(detection_result, settings.visible_classes)
    summary = analysis.summarise(detection_result, settings.criteria)
    annotated = viz.draw_detections(
        stages.final,
        detection_result.detections,
        show_labels=settings.show_labels,
        show_confidence=settings.show_confidence,
    )
    return stages, detection_result, summary, annotated


def _log(store, summaries, sources: list[str], mode: str, model: str) -> int:
    """
    Persist inspection outcomes, never letting a storage failure break the page.

    Args:
        store: the configured history store.
        summaries: one summary per inspected unit.
        sources: matching identifiers — file names, frame labels, camera labels.
        mode: which page produced them.
        model: the detector that produced the detections.

    Returns:
        How many rows were written. Zero when history is switched off.
    """
    if not summaries or store is None or not getattr(store, "available", False):
        return 0
    try:
        records = [
            InspectionRecord.from_summary(summary, source=source, mode=mode, model=model)
            for summary, source in zip(summaries, sources)
        ]
        return store.log_many(records)
    except Exception:                                        # noqa: BLE001
        return 0


def _build_pdf(builder, *args, **kwargs) -> tuple[bytes | None, str | None]:
    """
    Call a report builder, converting any failure into a message.

    PDF generation depends on ``reportlab``; if it is missing the rest of the
    interface must still work, so the failure is reported rather than raised.
    """
    try:
        return builder(*args, **kwargs), None
    except Exception as exc:                                 # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# Page 1 — single board
# --------------------------------------------------------------------------- #
def page_single(bridge: Pipeline, detector, settings: Settings, store) -> None:
    """One board in, one inspection certificate out."""
    st.markdown("#### Single board inspection")
    st.caption(
        "Upload a board image, or pick one from the datasets produced by Modules 1 "
        "and 2. Every intermediate stage is shown so the pipeline can be inspected "
        "step by step."
    )

    source_column, action_column = st.columns([3, 1])

    with source_column:
        uploaded = st.file_uploader(
            "Board image",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            key="single_upload",
        )
        sample_path = _sample_picker(bridge, key="single_sample")

    image = None
    source_name = ""
    if uploaded is not None:
        image = decode_image(uploaded.getvalue())
        source_name = uploaded.name
        if image is None:
            st.error("That file could not be decoded as an image.", icon="⛔")
    elif sample_path is not None:
        image = read_image(sample_path)
        source_name = sample_path.name
        if image is None:
            st.error(f"Could not read: {sample_path}", icon="⛔")

    with action_column:
        st.write("")
        st.write("")
        run = st.button(
            "Inspect board", type="primary", use_container_width=True,
            disabled=image is None,
        )

    if run and image is not None:
        with st.spinner("Running Modules 1 → 2 → 3…"):
            stages, detection_result, summary, annotated = inspect(
                image, bridge, detector, settings
            )
        logged = _log(store, [summary], [source_name], "single", detector.model_name)
        st.session_state["single_result"] = {
            "name": source_name,
            "stages": stages,
            "detections": detection_result.detections,
            "error": detection_result.error,
            "summary": summary,
            "annotated": annotated,
            "config": settings.as_config(detector.status()),
            "logged": logged,
        }

    stored = st.session_state.get("single_result")
    if not stored:
        st.info("Choose an image and select **Inspect board** to begin.", icon="🔍")
        return

    _render_single_result(stored, detector)


def _render_single_result(stored: dict[str, Any], detector) -> None:
    """Render the stored outcome of a single-board inspection."""
    stages = stored["stages"]
    summary: InspectionSummary = stored["summary"]
    detections = stored["detections"]
    annotated = stored["annotated"]

    st.divider()
    stage_chips(stages, detector.available)
    verdict_banner(summary)
    metric_row(summary)

    if stored.get("error"):
        st.warning(f"Module 3: {stored['error']}", icon="⚠️")

    for reason in summary.reasons:
        st.caption(f"· {reason}")
    if stored.get("logged"):
        st.caption("· Recorded in the inspection history.")

    # -- stage-by-stage images --------------------------------------------- #
    section("Processing stages")
    tabs = st.tabs([
        "Detection result",
        "Module 2 — aligned",
        "Module 1 — pre-processed",
        "Operator input",
        "Side by side",
    ])
    with tabs[0]:
        st.image(viz.to_rgb(annotated), use_container_width=True,
                 caption="Detections drawn on the image handed to Module 3.")
        defect_legend(detections)
    with tabs[1]:
        if stages.aligned is not None:
            st.image(viz.to_rgb(stages.aligned), use_container_width=True,
                     caption="Perspective-corrected top-down view of the board.")
        else:
            st.info(
                "Module 2 did not produce an aligned image for this board — no "
                "clean four-corner boundary was found, so detection ran on the "
                "pre-processed image instead.",
                icon="ℹ️",
            )
    with tabs[2]:
        if stages.preprocessed is not None:
            st.image(viz.to_rgb(stages.preprocessed), use_container_width=True,
                     caption="Colour-normalised and contrast-enhanced (CLAHE).")
        else:
            st.info("Module 1 was bypassed or returned no output.", icon="ℹ️")
    with tabs[3]:
        st.image(viz.to_rgb(stages.original), use_container_width=True,
                 caption="The image exactly as supplied.")
    with tabs[4]:
        left, right = st.columns(2)
        left.image(viz.to_rgb(stages.original), use_container_width=True, caption="Before")
        right.image(viz.to_rgb(annotated), use_container_width=True, caption="After")

    # -- findings ----------------------------------------------------------- #
    section("Findings")
    table = analysis.detections_dataframe(detections)
    if detections:
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(table, use_container_width=True, hide_index=True)
        with right:
            class_bar_chart(summary.class_counts, "Defect count by class.")
    else:
        st.success("No defects were detected above the confidence threshold.", icon="✅")

    # -- exports ------------------------------------------------------------ #
    section("Export")
    stem = Path(stored["name"]).stem or "board"
    pdf_bytes, pdf_error = _build_pdf(
        report.build_single_report,
        annotated=annotated,
        summary=summary,
        detections=detections,
        config=stored["config"],
        board_id=stored["name"] or "Board",
        original=stages.original,
    )
    if pdf_error:
        st.warning(f"The PDF report could not be generated — {pdf_error}", icon="⚠️")

    download_row([
        ("📄 PDF inspection report", pdf_bytes, f"{stem}_report.pdf", "application/pdf"),
        ("🖼 Annotated image (PNG)", viz.encode_png(annotated), f"{stem}_annotated.png", "image/png"),
        ("📊 Findings (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
         f"{stem}_findings.csv", "text/csv"),
    ])


# --------------------------------------------------------------------------- #
# Page 2 — batch
# --------------------------------------------------------------------------- #
def page_batch(bridge: Pipeline, detector, settings: Settings, store) -> None:
    """Bulk ingestion: a whole folder or a multi-file upload in one run."""
    st.markdown("#### Batch inspection")
    st.caption(
        "Inspect a production batch in one pass — a folder on disk or a "
        "multi-file upload — and export the yield statistics as CSV and PDF."
    )

    source = st.radio(
        "Source", ["Folder on disk", "Upload multiple images"],
        horizontal=True, key="batch_source",
    )

    paths: list[Path] = []
    uploads: list[Any] = []
    run_name = "Batch run"

    if source == "Folder on disk":
        folder = _folder_picker(bridge, key="batch_folder")
        recursive = st.checkbox(
            "Include sub-folders", value=True,
            help="The datasets are organised one folder per defect class, so this "
                 "is normally left on.",
        )
        if folder:
            paths = list_images(folder, recursive=recursive)
            run_name = Path(folder).name or folder
            if paths:
                st.success(f"Found {len(paths)} image(s) in `{folder}`.", icon="📁")
            else:
                st.warning(f"No images were found in `{folder}`.", icon="⚠️")
    else:
        uploads = st.file_uploader(
            "Board images",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="batch_upload",
        ) or []
        run_name = f"Upload of {len(uploads)} image(s)"
        if uploads:
            st.success(f"{len(uploads)} image(s) ready for inspection.", icon="📁")

    total_available = len(paths) or len(uploads)
    limit_column, action_column = st.columns([3, 1])
    with limit_column:
        # A slider needs a range; with zero or one candidate there is nothing to
        # choose, so the control is replaced by a plain statement of the count.
        if total_available > 1:
            limit = st.slider(
                "Maximum images to inspect this run",
                min_value=1,
                max_value=min(500, total_available),
                value=min(25, total_available),
                help="Caps a run so a large dataset folder cannot lock the "
                     "interface up. Raise it once the throughput on this machine "
                     "is known.",
            )
        else:
            limit = max(1, total_available)
            st.caption(
                f"{total_available} image(s) selected — choose a source with more "
                "images to set a run limit."
            )
    with action_column:
        st.write("")
        st.write("")
        run = st.button(
            "Run batch", type="primary", use_container_width=True,
            disabled=total_available == 0,
        )

    if run:
        _run_batch(paths[:limit], uploads[:limit], run_name, bridge, detector, settings, store)

    stored = st.session_state.get("batch_result")
    if not stored:
        st.info("Choose a source and select **Run batch** to begin.", icon="📁")
        return

    _render_batch_result(stored)


def _run_batch(paths, uploads, run_name, bridge, detector, settings, store) -> None:
    """Inspect every item, updating a progress bar, and store the outcome."""
    items: list[tuple[str, Any]] = (
        [(p.name, p) for p in paths] if paths else [(u.name, u) for u in uploads]
    )
    if not items:
        return

    progress = st.progress(0.0, text="Starting batch…")
    summaries: list[InspectionSummary] = []
    names: list[str] = []
    rows: list[dict[str, Any]] = []
    gallery: list[tuple[str, Any]] = []
    skipped: list[str] = []

    for index, (name, item) in enumerate(items, start=1):
        image = read_image(item) if isinstance(item, Path) else decode_image(item.getvalue())
        if image is None:
            skipped.append(name)
            progress.progress(index / len(items), text=f"Skipped {name} (unreadable)")
            continue

        _, detection_result, summary, annotated = inspect(image, bridge, detector, settings)
        summaries.append(summary)
        names.append(name)
        rows.append(summary.as_row(name))

        # Keep a handful of failures as evidence for the PDF appendix and gallery.
        if summary.verdict in ("FAIL", "REVIEW") and len(gallery) < 6:
            gallery.append((f"{name} — {summary.verdict}", viz.thumbnail(annotated, 520)))

        progress.progress(
            index / len(items),
            text=f"{index}/{len(items)} — {name}: {summary.verdict} "
                 f"({summary.total_defects} defect(s))",
        )

    progress.empty()
    logged = _log(store, summaries, names, "batch", detector.model_name)
    st.session_state["batch_result"] = {
        "run_name": run_name,
        "summaries": summaries,
        "rows": rows,
        "gallery": gallery,
        "skipped": skipped,
        "batch": analysis.summarise_batch(summaries),
        "config": settings.as_config(detector.status()),
        "logged": logged,
    }


def _render_batch_result(stored: dict[str, Any]) -> None:
    """Render the stored outcome of a batch run."""
    batch = stored["batch"]
    summaries: list[InspectionSummary] = stored["summaries"]
    rows: list[dict[str, Any]] = stored["rows"]

    st.divider()
    section("Run summary")
    columns = st.columns(5)
    columns[0].metric("Boards inspected", batch.total_boards)
    columns[1].metric("Yield", f"{batch.yield_pct:.1f}%")
    columns[2].metric("Failed", batch.failed)
    columns[3].metric("For review", batch.review)
    columns[4].metric("Defects per board", f"{batch.defect_rate:.2f}")

    if stored.get("logged"):
        st.caption(f"· {stored['logged']} result(s) recorded in the inspection history.")

    if stored["skipped"]:
        st.warning(
            f"{len(stored['skipped'])} file(s) could not be decoded and were "
            f"skipped: {', '.join(stored['skipped'][:5])}"
            + ("…" if len(stored["skipped"]) > 5 else ""),
            icon="⚠️",
        )

    left, right = st.columns([2, 3])
    with left:
        section("Verdict split")
        verdicts = pd.DataFrame(
            {"count": [batch.passed, batch.review, batch.failed]},
            index=["PASS", "REVIEW", "FAIL"],
        )
        st.bar_chart(verdicts, height=260)
    with right:
        section("Defect distribution")
        class_bar_chart(batch.class_counts, "Total detections per defect class across the run.")

    section("Per-board results")
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True, height=340)

    distribution = analysis.class_distribution(summaries)
    if not distribution.empty:
        with st.expander("Defect distribution table"):
            st.dataframe(distribution, use_container_width=True, hide_index=True)

    gallery = stored["gallery"]
    if gallery:
        section("Flagged boards")
        for row_start in range(0, len(gallery), 3):
            for column, (caption, image) in zip(
                st.columns(3), gallery[row_start:row_start + 3]
            ):
                column.image(viz.to_rgb(image), caption=caption, use_container_width=True)

    section("Export")
    pdf_bytes, pdf_error = _build_pdf(
        report.build_batch_report,
        batch=batch,
        rows=rows,
        config=stored["config"],
        run_name=stored["run_name"],
        gallery=gallery,
    )
    if pdf_error:
        st.warning(f"The PDF report could not be generated — {pdf_error}", icon="⚠️")

    download_row([
        ("📄 PDF batch report", pdf_bytes, "batch_report.pdf", "application/pdf"),
        ("📊 Results (CSV)", frame.to_csv(index=False).encode("utf-8-sig"),
         "batch_results.csv", "text/csv"),
        ("📈 Distribution (CSV)",
         distribution.to_csv(index=False).encode("utf-8-sig") if not distribution.empty else None,
         "batch_distribution.csv", "text/csv"),
    ])


# --------------------------------------------------------------------------- #
# Page 3 — recorded video
# --------------------------------------------------------------------------- #
def page_video(bridge: Pipeline, detector, settings: Settings, store) -> None:
    """Frame-by-frame inspection of a recorded conveyor video."""
    st.markdown("#### Video stream inspection")
    st.caption(
        "Ingest a video of the inspection conveyor and run the pipeline across "
        "individual frames. The annotated stream and the per-frame defect "
        "timeline are both available for download."
    )
    _fast_model_note(settings)

    uploaded = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"],
                               key="video_upload")
    sample = _video_picker(bridge)

    video_bytes: bytes | None = None
    video_name = ""
    if uploaded is not None:
        video_bytes = uploaded.getvalue()
        video_name = uploaded.name
    elif sample is not None:
        try:
            video_bytes = sample.read_bytes()
            video_name = sample.name
        except OSError as exc:
            st.error(f"Could not read {sample}: {exc}", icon="⛔")

    if video_bytes:
        info = video.probe(video_bytes)
        columns = st.columns(4)
        columns[0].metric("Frames", info["frame_count"])
        columns[1].metric("Frame rate", f"{info['fps']:.1f} fps")
        columns[2].metric("Resolution", f"{info['width']}×{info['height']}")
        columns[3].metric("Duration", f"{info['duration_s']:.1f} s")

    section("Sampling")
    left, middle, right = st.columns([2, 2, 1])
    with left:
        stride = st.slider(
            "Analyse every n-th frame", 1, 30, 5,
            help="Detection is the expensive step. Frames in between reuse the "
                 "most recent annotation, so the output still plays at the "
                 "original frame rate.",
        )
    with middle:
        max_frames = st.slider(
            "Maximum frames to analyse", 10, 600, 120, step=10,
            help="Caps the run so a long clip cannot lock the interface up.",
        )
    with right:
        st.write("")
        st.write("")
        run = st.button("Inspect video", type="primary", use_container_width=True,
                        disabled=not video_bytes)

    align_in_video = st.checkbox(
        "Apply Module 2 alignment to every frame", value=False,
        help="A conveyor stream rarely presents a clean four-corner board "
             "boundary, so alignment usually fails per frame and only slows the "
             "run down. Enable it for footage of stationary boards.",
    )

    if run and video_bytes:
        progress = st.progress(0.0, text="Starting…")

        def report_progress(fraction: float, message: str) -> None:
            progress.progress(min(1.0, max(0.0, fraction)), text=message)

        with st.spinner("Processing frames…"):
            result = video.process_video(
                video_bytes,
                bridge=bridge,
                detector=detector,
                criteria=settings.criteria,
                mode=settings.mode,
                do_preprocess=settings.do_preprocess,
                do_align=align_in_video,
                frame_stride=int(stride),
                max_frames=int(max_frames),
                confidence=settings.confidence,
                iou=settings.iou,
                progress=report_progress,
            )
        progress.empty()
        logged = _log(
            store,
            result.summaries,
            [f"{video_name}#frame_{record.frame_index}" for record in result.records],
            "video",
            detector.model_name,
        )
        st.session_state["video_result"] = {
            "name": video_name,
            "result": result,
            "config": settings.as_config(detector.status()),
            "logged": logged,
        }

    stored = st.session_state.get("video_result")
    if not stored:
        st.info("Choose a video and select **Inspect video** to begin.", icon="🎞")
    else:
        _render_video_result(stored)

    st.divider()
    section("Board-by-board inspection")
    st.caption(
        "Inspect the clip one *board* at a time rather than one frame at a "
        "time: each board is cut out of the stream and run through Modules "
        "1-3 on its own, exactly as a still image would be, and every result "
        "can be downloaded."
    )
    scan_tab, frame_tab = st.tabs(["Scan the whole video", "Pick one frame"])
    with scan_tab:
        _render_board_scan(video_bytes, video_name, bridge, detector, settings, store)
    with frame_tab:
        _render_frame_inspector(
            video_bytes, info if video_bytes else None, bridge, detector, settings
        )


def _render_board_scan(
    video_bytes: bytes | None,
    video_name: str,
    bridge: Pipeline,
    detector,
    settings: Settings,
    store,
) -> None:
    """
    Walk the whole clip, capture every board once, and inspect each on its own.

    This is the board-oriented counterpart to the frame-by-frame run above.
    A board stays in view for hundreds of frames, so inspecting frames returns
    the same physical board over and over; tracking each board and capturing it
    at its most complete moment gives one clean shot per board, and each of
    those goes through Modules 1, 2 and 3 individually.
    """
    if not video_bytes:
        st.info("Upload a video above to use this tool.", icon="🎞")
        return

    left, middle, right = st.columns([2, 2, 1])
    with left:
        every_n = st.slider(
            "Check every n-th frame", 1, 15, 3, key="scan_every_n",
            help="A board moves only a few pixels per frame, so testing every "
                 "frame buys nothing. Lower this only if boards travel fast.",
        )
    with middle:
        max_boards = st.slider(
            "Maximum boards to capture", 1, 50, 20, key="scan_max_boards",
            help="Stops a long clip from producing hundreds of inspections.",
        )
    with right:
        st.write("")
        st.write("")
        run_scan = st.button("Scan video for boards", type="primary",
                             use_container_width=True, key="scan_run")

    include_partial = st.checkbox(
        "Also inspect boards that are never completely in frame", value=False,
        key="scan_include_partial",
        help="A board still entering when the clip ends is cut off by the edge "
             "of the picture. It is found and listed either way; inspecting one "
             "reports only the defects on the visible part, so the verdict is "
             "not a verdict on the whole board.",
    )

    if run_scan:
        progress = st.progress(0.0, text="Scanning…")

        def on_progress(fraction: float, message: str) -> None:
            progress.progress(min(1.0, max(0.0, fraction)), text=message)

        with st.spinner("Finding boards…"):
            captured = video.scan_boards(
                video_bytes,
                every_n_frames=int(every_n),
                max_boards=int(max_boards),
                progress=on_progress,
            )

        results = []
        skipped = 0
        for position, board in enumerate(captured, start=1):
            progress.progress(
                min(1.0, position / max(1, len(captured))),
                text=f"Inspecting board {position} of {len(captured)}…",
            )
            item = {
                "index": position,
                "frame": board["frame"],
                "timestamp_s": board["timestamp_s"],
                "box": board["box"],
                "fully_visible": board["fully_visible"],
                "stages": None, "detection": None, "summary": None, "annotated": None,
            }
            # A board the clip never shows whole is reported either way, but is
            # only put through the pipeline when the operator asks for it — its
            # verdict would otherwise read as a verdict on a board that was
            # only half in the picture.
            if board["fully_visible"] or include_partial:
                stages, detection_result, summary, annotated = inspect(
                    board["crop"], bridge, detector, settings, do_align=True
                )
                item.update({"stages": stages, "detection": detection_result,
                             "summary": summary, "annotated": annotated})
            else:
                skipped += 1
            results.append(item)
        progress.empty()

        inspected = [item for item in results if item["summary"] is not None]
        logged = _log(
            store,
            [item["summary"] for item in inspected],
            [f"{video_name}#board_{item['index']}_frame_{item['frame']}" for item in inspected],
            "video-board",
            detector.model_name,
        )
        st.session_state["board_scan_result"] = {
            "name": video_name, "boards": results, "logged": logged, "skipped": skipped,
        }

    scan = st.session_state.get("board_scan_result")
    if not scan:
        st.info("Select **Scan video for boards** to inspect the clip board by board.",
                icon="🔍")
        return

    boards = scan["boards"]
    if not boards:
        st.warning(
            "No board was found in this clip. Lower **Check every n-th frame**, "
            "or use the *Pick one frame* tab to look at a frame directly.",
            icon="⚠️",
        )
        return

    inspected = [item for item in boards if item["summary"] is not None]
    failed = sum(1 for item in inspected if item["summary"].verdict == "FAIL")
    columns = st.columns(4)
    columns[0].metric("Boards found", len(boards))
    columns[1].metric("Boards inspected", len(inspected))
    columns[2].metric("Failing boards", failed)
    columns[3].metric(
        "Defects found", sum(item["summary"].total_defects for item in inspected)
    )

    if scan.get("skipped"):
        st.info(
            f"{scan['skipped']} board(s) are never completely inside the picture "
            "in this clip — they are still listed below, but were not inspected, "
            "because a verdict on a board that is half out of frame is not a "
            "verdict on the board. Tick the box above to inspect them anyway.",
            icon="ℹ️",
        )
    if scan.get("logged"):
        st.caption(f"· {scan['logged']} inspection(s) written to history.")

    table = pd.DataFrame([
        {
            "board": item["index"],
            "frame": item["frame"],
            "time_s": round(item["timestamp_s"], 2),
            "in_frame": "whole board" if item["fully_visible"] else "partly cut off",
            "verdict": item["summary"].verdict if item["summary"] else "not inspected",
            "defects": item["summary"].total_defects if item["summary"] else None,
            "quality_score": (round(item["summary"].quality_score, 1)
                              if item["summary"] else None),
            "alignment": (_alignment_state(item["stages"]) if item["stages"] else "—"),
        }
        for item in boards
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    section("Export")
    stem = Path(scan["name"]).stem or "stream"
    pdf_bytes = None
    if inspected:
        batch = analysis.summarise_batch([item["summary"] for item in inspected])
        rows = [
            item["summary"].as_row(f"board{item['index']:02d}_frame{item['frame']}")
            for item in inspected
        ]
        # Failing/review boards make the most useful evidence appendix — same
        # choice as the other batch-style reports in this app.
        gallery = [
            (f"Board {item['index']} — frame {item['frame']} — {item['summary'].verdict}",
             viz.thumbnail(item["annotated"], 520))
            for item in inspected
            if item["summary"].verdict in ("FAIL", "REVIEW")
        ][:6]
        pdf_bytes, pdf_error = _build_pdf(
            report.build_batch_report,
            batch=batch,
            rows=rows,
            config=settings.as_config(detector.status()),
            run_name=f"Video board scan — {scan['name']}",
            gallery=gallery,
        )
        if pdf_error:
            st.warning(f"The PDF report could not be generated — {pdf_error}", icon="⚠️")
    else:
        st.caption("Inspect at least one board to generate a PDF report.")

    download_row([
        ("📄 PDF stream report", pdf_bytes, f"{stem}_boards_report.pdf", "application/pdf"),
        ("🗂 All boards (ZIP)", _boards_zip(inspected, scan["name"]),
         "video_boards.zip", "application/zip"),
        ("📊 Summary (CSV)", table.to_csv(index=False).encode("utf-8-sig"),
         "video_boards.csv", "text/csv"),
    ])

    for item in boards:
        heading = (f"###### Board {item['index']} — frame {item['frame']} "
                   f"(t = {item['timestamp_s']:.1f} s)")
        if not item["fully_visible"]:
            heading += " · partly cut off"
        st.markdown(heading)
        if item["summary"] is None:
            st.caption("Not inspected — the clip never shows this board whole.")
            continue
        _render_board_pair(item, f"board{item['index']}_frame{item['frame']}")


def _alignment_state(stages) -> str:
    """One word for what Module 2 actually did to this board."""
    if not stages.align_ok:
        return "no outline"
    return "straightened" if getattr(stages, "align_effective", True) else "rescaled only"


def _render_board_pair(item: dict[str, Any], stem: str) -> None:
    """Show one board's detector input and annotated result, with downloads."""
    stages = item["stages"]
    state = _alignment_state(stages)
    captions = {
        "straightened": "Module 1 + 2 output — board straightened (detector input)",
        "rescaled only": ("Module 1 output — Module 2 rescaled it without "
                          "straightening it (detector input)"),
        "no outline": "Module 1 output — Module 2 found no board outline (detector input)",
    }
    columns = st.columns(2)
    columns[0].image(viz.to_rgb(stages.final), caption=captions[state],
                     use_container_width=True)
    columns[1].image(
        viz.to_rgb(item["annotated"]),
        caption=f"{item['summary'].verdict} · {item['summary'].total_defects} defect(s)",
        use_container_width=True,
    )
    download_row([
        (f"⬇️ Detector input", viz.encode_jpeg(stages.final), f"{stem}_input.jpg", "image/jpeg"),
        (f"⬇️ Annotated", viz.encode_jpeg(item["annotated"]),
         f"{stem}_annotated.jpg", "image/jpeg"),
    ])


def _boards_zip(boards: list[dict[str, Any]], video_name: str) -> bytes | None:
    """
    Bundle every inspected board into one archive.

    With a dozen boards on screen, downloading them one button at a time is
    the slow path — this packs both images per board plus the summary table
    into a single file.
    """
    if not boards:
        return None

    import csv
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        rows = io.StringIO()
        writer = csv.writer(rows)
        writer.writerow(["board", "frame", "time_s", "verdict", "defects",
                         "quality_score", "alignment", "source_video"])
        for item in boards:
            if item.get("summary") is None:      # found but not inspected
                continue
            stem = f"board{item['index']:02d}_frame{item['frame']}"
            for suffix, image in (("input", item["stages"].final),
                                  ("annotated", item["annotated"])):
                encoded = viz.encode_jpeg(image)
                if encoded:
                    archive.writestr(f"{stem}_{suffix}.jpg", encoded)
            writer.writerow([
                item["index"], item["frame"], round(item["timestamp_s"], 2),
                item["summary"].verdict, item["summary"].total_defects,
                round(item["summary"].quality_score, 1),
                _alignment_state(item["stages"]), video_name,
            ])
        # utf-8-sig so Excel (which assumes the system codepage without a
        # byte-order mark) doesn't mangle non-ASCII characters like "—".
        archive.writestr("summary.csv", rows.getvalue().encode("utf-8-sig"))
    return buffer.getvalue()


def _render_frame_inspector(
    video_bytes: bytes | None,
    info: dict[str, Any] | None,
    bridge: Pipeline,
    detector,
    settings: Settings,
) -> None:
    """
    Pull one operator-chosen frame out of the clip and run it through Modules
    1-3 on its own. A frame can hold more than one board — each detected
    region is cropped and inspected separately.
    """
    st.caption(
        "Pick a frame number and run it through the full pipeline by itself — "
        "useful for a close look at one moment in the clip. If the frame "
        "contains more than one board, each one is detected, cropped and "
        "inspected on its own."
    )

    if not video_bytes:
        st.info("Upload a video above to use this tool.", icon="🎞")
        return

    frame_count = info["frame_count"] if info and info["frame_count"] else 1
    pick_col, button_col = st.columns([3, 1])
    with pick_col:
        frame_index = st.number_input(
            "Frame number", min_value=0, max_value=max(0, frame_count - 1),
            value=0, step=1, key="frame_inspector_index",
            help=f"This clip has {frame_count} frame(s).",
        )
    with button_col:
        st.write("")
        st.write("")
        inspect_frame = st.button(
            "Detect boards in this frame", use_container_width=True,
            key="frame_inspector_run",
        )

    if inspect_frame:
        frame = video.extract_frame(video_bytes, int(frame_index))
        if frame is None:
            st.error("Could not read that frame from the video.", icon="⛔")
        else:
            boxes = roi.find_pcb_regions(frame)
            board_results = []
            for box in boxes:
                crop = roi.crop_with_padding(frame, box)
                stages, detection_result, summary, annotated = inspect(
                    crop, bridge, detector, settings, do_align=True
                )
                board_results.append({
                    "box": box,
                    "stages": stages,
                    "detection": detection_result,
                    "summary": summary,
                    "annotated": annotated,
                })
            st.session_state["frame_boards_result"] = {
                "frame_index": int(frame_index),
                "overview": roi.draw_regions(frame, boxes),
                "boards": board_results,
            }

    frame_boards = st.session_state.get("frame_boards_result")
    if not frame_boards:
        return

    st.markdown(
        f"**Boards detected: {len(frame_boards['boards'])}** "
        f"— frame {frame_boards['frame_index']}"
    )
    st.image(
        viz.to_rgb(frame_boards["overview"]),
        caption=f"Detected ROIs ({len(frame_boards['boards'])} board(s))",
        use_container_width=True,
    )

    if not frame_boards["boards"]:
        st.info("No PCB-shaped region was found in this frame.", icon="ℹ️")
        return

    for i, board in enumerate(frame_boards["boards"], start=1):
        st.markdown(f"###### Board {i}")
        _render_board_pair(board, f"frame{frame_boards['frame_index']}_board{i}")


def _render_video_result(stored: dict[str, Any]) -> None:
    """Render the stored outcome of a video run."""
    result: video.VideoResult = stored["result"]

    st.divider()
    if result.error:
        st.error(f"Video processing failed — {result.error}", icon="⛔")
        if not result.records:
            return

    section("Stream summary")
    columns = st.columns(5)
    columns[0].metric("Frames read", result.frames_read)
    columns[1].metric("Frames analysed", result.frames_analysed)
    columns[2].metric("Defects detected", result.total_defects)
    columns[3].metric("Failing frames", result.failed_frames)
    columns[4].metric("Duration", f"{result.duration_s:.1f} s")

    if stored.get("logged"):
        st.caption(f"· {stored['logged']} frame result(s) recorded in the inspection history.")

    if result.video_bytes:
        section("Annotated stream")
        st.video(result.video_bytes)

    if result.worst_frame is not None:
        section("Worst frame")
        st.image(viz.to_rgb(result.worst_frame), caption=result.worst_caption,
                 use_container_width=True)

    if result.records:
        section("Defect timeline")
        frame = pd.DataFrame([record.as_row() for record in result.records])
        st.line_chart(frame.set_index("time_s")[["defects", "quality_score"]], height=260)
        st.caption(
            "Defect count and quality score per analysed frame, against elapsed time."
        )

        with st.expander("Per-frame results"):
            st.dataframe(frame, use_container_width=True, hide_index=True, height=300)

        batch = analysis.summarise_batch(result.summaries)
        section("Export")
        rows = [
            summary.as_row(f"frame_{record.frame_index}")
            for summary, record in zip(result.summaries, result.records)
        ]
        gallery = (
            [(result.worst_caption or "Worst frame", result.worst_frame)]
            if result.worst_frame is not None else []
        )
        pdf_bytes, pdf_error = _build_pdf(
            report.build_batch_report,
            batch=batch,
            rows=rows,
            config=stored["config"],
            run_name=f"Video — {stored['name']}",
            gallery=gallery,
        )
        if pdf_error:
            st.warning(f"The PDF report could not be generated — {pdf_error}", icon="⚠️")

        stem = Path(stored["name"]).stem or "stream"
        download_row([
            ("📄 PDF stream report", pdf_bytes, f"{stem}_report.pdf", "application/pdf"),
            ("🎞 Annotated video", result.video_bytes, f"{stem}_inspected.mp4", "video/mp4"),
            ("📊 Timeline (CSV)", frame.to_csv(index=False).encode("utf-8-sig"),
             f"{stem}_timeline.csv", "text/csv"),
        ])


# --------------------------------------------------------------------------- #
# Page 4 — live camera
# --------------------------------------------------------------------------- #
def page_live(bridge: Pipeline, detector, settings: Settings, store) -> None:
    """Real-time inspection from a camera attached to this machine."""
    st.markdown("#### Live inspection")
    st.caption(
        "Inspect a board as the camera sees it. Snapshot mode works in any "
        "browser; continuous mode drives a camera attached to the machine "
        "running this interface."
    )
    _fast_model_note(settings)

    capture_mode = st.radio(
        "Capture mode",
        ["Snapshot", "Continuous stream"],
        horizontal=True,
        key="live_mode",
        help="Snapshot takes one photograph through the browser and inspects it "
             "like any uploaded board. Continuous stream repeatedly captures and "
             "inspects, reporting a running yield.",
    )

    if capture_mode == "Snapshot":
        _live_snapshot(bridge, detector, settings, store)
    else:
        _live_stream(bridge, detector, settings, store)


def _live_snapshot(bridge: Pipeline, detector, settings: Settings, store) -> None:
    """
    One photograph, inspected exactly like an uploaded board.

    ``st.camera_input`` runs in the browser, so this mode also works when the
    interface is deployed to a server with no camera of its own.
    """
    photo = st.camera_input("Point the camera at the board and take a photograph",
                            key="live_camera_input")
    if photo is None:
        st.info("Grant the browser camera access, then take a photograph.", icon="📷")
        return

    image = decode_image(photo.getvalue())
    if image is None:
        st.error("The photograph could not be decoded.", icon="⛔")
        return

    with st.spinner("Running Modules 1 → 2 → 3…"):
        stages, detection_result, summary, annotated = inspect(image, bridge, detector, settings)

    logged = _log(store, [summary], ["camera snapshot"], "live", detector.model_name)
    _render_single_result(
        {
            "name": "camera_snapshot.jpg",
            "stages": stages,
            "detections": detection_result.detections,
            "error": detection_result.error,
            "summary": summary,
            "annotated": annotated,
            "config": settings.as_config(detector.status()),
            "logged": logged,
        },
        detector,
    )


def _live_stream(bridge: Pipeline, detector, settings: Settings, store) -> None:
    """
    Continuous capture from a local camera.

    The loop runs in short chunks (see ``core.live``) so the Stop button stays
    responsive. Session state carries the camera handle and the running totals
    between chunks.
    """
    running = st.session_state.get("live_running", False)

    cameras = _camera_list()
    if not cameras and not running:
        st.warning(
            "No camera could be opened from the machine running this interface. "
            "Continuous mode captures here in Python, not in the browser, so it "
            "needs the camera to be free and reachable from this process. The "
            "usual cause is that something else already holds it — **Snapshot "
            "mode in this or another tab keeps the camera open**, so close those "
            "tabs (and any other app using it) and re-scan. Use **Snapshot** "
            "mode instead if the camera is on the machine viewing this page "
            "rather than the one running it.",
            icon="📷",
        )
        rescan_col, _ = st.columns([1, 3])
        with rescan_col:
            if st.button("↻ Re-scan for cameras", use_container_width=True,
                         key="live_rescan"):
                _camera_list.clear()
                st.rerun()
        with st.expander("What the camera probe actually found"):
            st.caption(
                "Every device index is tried against every capture backend. "
                "*opened* but not *delivered a frame* normally means the camera "
                "is busy in another program; nothing opening at all means no "
                "device at that index, or the operating system is denying this "
                "Python process access to it."
            )
            st.dataframe(pd.DataFrame(live.probe_report()),
                         use_container_width=True, hide_index=True)

    controls = st.columns([2, 2, 2, 1, 1])
    with controls[0]:
        camera_index = st.selectbox(
            "Camera", cameras or [0],
            format_func=lambda i: f"Camera {i}",
            disabled=running or not cameras,
            key="live_camera_index",
        )
    with controls[1]:
        target_fps = st.slider(
            "Inspections per second", 0.5, 10.0, 3.0, 0.5, disabled=running,
            help="How often to run the full pipeline. Detection, not capture, "
                 "sets the achievable ceiling — the measured rate is shown below.",
        )
    with controls[2]:
        align_live = st.checkbox(
            "Apply Module 2 alignment", value=False, disabled=running,
            help="A hand-held board rarely presents a clean four-corner boundary, "
                 "so alignment usually fails per frame and only costs time.",
        )
    with controls[3]:
        st.write("")
        st.write("")
        start = st.button("▶ Start", type="primary", use_container_width=True,
                          disabled=running or not cameras)
    with controls[4]:
        st.write("")
        st.write("")
        stop = st.button("■ Stop", use_container_width=True, disabled=not running)

    if start:
        st.session_state["live_running"] = True
        st.session_state["live_stats"] = live.LiveStats()
        st.session_state["live_capture"] = live.open_camera(int(camera_index))
        st.rerun()

    if stop:
        st.session_state["live_running"] = False
        live.close_camera(st.session_state.pop("live_capture", None))
        st.rerun()

    stats: live.LiveStats | None = st.session_state.get("live_stats")

    if running:
        capture = st.session_state.get("live_capture")
        if capture is None:
            st.session_state["live_running"] = False
            st.error(
                "The camera could not be opened. It is most likely held by "
                "another program — Snapshot mode in this or another browser tab "
                "keeps it open. Close those and press Start again.",
                icon="⛔",
            )
            return

        st.markdown(
            '<div class="pcb-chips"><span class="pcb-chip pcb-chip--ok">● LIVE</span></div>',
            unsafe_allow_html=True,
        )
        frame_slot = st.empty()
        metric_slot = st.empty()

        def on_frame(annotated, summary) -> None:
            frame_slot.image(viz.to_rgb(annotated), use_container_width=True)
            with metric_slot.container():
                columns = st.columns(4)
                columns[0].metric("Verdict", summary.verdict)
                columns[1].metric("Defects in frame", summary.total_defects)
                columns[2].metric("Frames inspected", stats.frames)
                columns[3].metric("Effective rate", f"{stats.effective_fps:.1f} /s")

        chunk = live.run_chunk(
            capture,
            bridge=bridge,
            detector=detector,
            stats=stats,
            criteria=settings.criteria,
            seconds=2.0,
            target_fps=float(target_fps),
            mode=settings.mode,
            do_preprocess=settings.do_preprocess,
            do_align=bool(align_live),
            confidence=settings.confidence,
            iou=settings.iou,
            show_labels=settings.show_labels,
            show_confidence=settings.show_confidence,
            on_frame=on_frame,
        )

        # Persist this chunk's frames in one write rather than one per frame.
        if chunk.frames_processed:
            recent = stats.timeline[-chunk.frames_processed:]
            _log_live_chunk(store, stats, recent, detector.model_name)

        if chunk.error:
            st.session_state["live_running"] = False
            live.close_camera(st.session_state.pop("live_capture", None))
            st.error(f"Capture stopped — {chunk.error}", icon="⛔")
            return

        st.rerun()

    # -- session summary, shown while stopped ------------------------------- #
    if stats is None or stats.frames == 0:
        st.info("Select **Start** to begin inspecting the live camera feed.", icon="📷")
        return

    _render_live_summary(stats, settings, detector)


def _log_live_chunk(store, stats: live.LiveStats, recent: list[dict[str, Any]], model: str) -> None:
    """
    Write one chunk's frames to the history in a single call.

    The timeline entries hold only what the chart needs, so the records written
    here carry the headline figures rather than the full per-class breakdown;
    that keeps a long live session from flooding the table.
    """
    if store is None or not getattr(store, "available", False) or not recent:
        return
    try:
        records = [
            InspectionRecord(
                source=f"live#frame_{entry['frame']}",
                mode="live",
                verdict=str(entry["verdict"]),
                quality_score=float(entry["quality_score"]),
                total_defects=int(entry["defects"]),
                model=model,
            )
            for entry in recent
        ]
        store.log_many(records)
    except Exception:                                        # noqa: BLE001
        pass


def _render_live_summary(stats: live.LiveStats, settings: Settings, detector) -> None:
    """Session totals shown once the live capture has been stopped."""
    st.divider()
    section("Session summary")
    columns = st.columns(5)
    columns[0].metric("Frames inspected", stats.frames)
    columns[1].metric("Pass rate", f"{stats.pass_rate:.1f}%")
    columns[2].metric("Defects seen", stats.defects)
    columns[3].metric("Effective rate", f"{stats.effective_fps:.1f} /s")
    columns[4].metric("Mean inference", f"{stats.mean_inference_ms:.0f} ms")

    if stats.worst_frame is not None:
        section("Worst frame")
        st.image(viz.to_rgb(stats.worst_frame), caption=stats.worst_caption,
                 use_container_width=True)

    left, right = st.columns([3, 2])
    with left:
        section("Defect timeline")
        frame = pd.DataFrame(stats.timeline)
        if not frame.empty:
            st.line_chart(frame.set_index("time_s")[["defects", "quality_score"]], height=250)
    with right:
        section("Defect distribution")
        class_bar_chart(stats.class_counts, "Detections per class across the session.")

    section("Export")
    frame = pd.DataFrame(stats.timeline)
    batch = BatchSummary(
        total_boards=stats.frames,
        passed=stats.verdict_counts.get("PASS", 0),
        review=stats.verdict_counts.get("REVIEW", 0),
        failed=stats.verdict_counts.get("FAIL", 0),
        total_defects=stats.defects,
        mean_quality=(
            sum(entry["quality_score"] for entry in stats.timeline) / len(stats.timeline)
            if stats.timeline else 0.0
        ),
        mean_inference_ms=stats.mean_inference_ms,
        class_counts=dict(stats.class_counts),
    )
    rows = [
        {
            "image": f"frame_{entry['frame']}",
            "verdict": entry["verdict"],
            "quality_score": entry["quality_score"],
            "total_defects": entry["defects"],
            "critical_defects": "",
            "mean_confidence": "",
        }
        for entry in stats.timeline
    ]
    gallery = (
        [(stats.worst_caption or "Worst frame", stats.worst_frame)]
        if stats.worst_frame is not None else []
    )
    pdf_bytes, pdf_error = _build_pdf(
        report.build_batch_report,
        batch=batch,
        rows=rows,
        config=settings.as_config(detector.status()),
        run_name="Live camera session",
        gallery=gallery,
    )
    if pdf_error:
        st.warning(f"The PDF report could not be generated — {pdf_error}", icon="⚠️")

    download_row([
        ("📄 PDF session report", pdf_bytes, "live_session_report.pdf", "application/pdf"),
        ("🖼 Worst frame (PNG)",
         viz.encode_png(stats.worst_frame) if stats.worst_frame is not None else None,
         "live_worst_frame.png", "image/png"),
        ("📊 Timeline (CSV)",
         frame.to_csv(index=False).encode("utf-8-sig") if not frame.empty else None,
         "live_timeline.csv", "text/csv"),
    ])


# --------------------------------------------------------------------------- #
# Page 5 — history
# --------------------------------------------------------------------------- #
def page_history(store, settings: Settings) -> None:
    """The yield dashboard: every board this system has ever inspected."""
    st.markdown("#### Inspection history")
    st.caption(
        "Every result the system has recorded, across all four inspection modes. "
        "This is the production view — yield over time rather than one board at "
        "a time."
    )

    status = store.status()
    if not store.available:
        st.warning(
            f"History is not being recorded — {status.get('error') or status.get('note')} "
            "Choose a store in the sidebar under **Inspection history**.",
            icon="⚠️",
        )
        return

    st.caption(f"· Store: **{status['kind']}** → `{status['target']}`")

    limit = st.slider("Records to load", 50, 5000, 500, 50)
    rows = store.fetch(limit=limit)
    if not rows:
        st.info(
            "No results have been recorded yet. Inspect a board on any of the "
            "other pages and it will appear here.",
            icon="📋",
        )
        return

    frame = pd.DataFrame(rows)
    frame = frame.drop(columns=[c for c in ("class_counts", "created_at") if c in frame.columns])
    if "inspected_at" in frame.columns:
        frame["inspected_at"] = pd.to_datetime(frame["inspected_at"], errors="coerce", utc=True)

    # -- filters ------------------------------------------------------------ #
    section("Filters")
    left, right = st.columns(2)
    with left:
        modes = sorted(frame["mode"].dropna().unique()) if "mode" in frame else []
        chosen_modes = st.multiselect("Inspection mode", modes, default=modes)
    with right:
        verdicts = sorted(frame["verdict"].dropna().unique()) if "verdict" in frame else []
        chosen_verdicts = st.multiselect("Verdict", verdicts, default=verdicts)

    filtered = frame
    if chosen_modes:
        filtered = filtered[filtered["mode"].isin(chosen_modes)]
    if chosen_verdicts:
        filtered = filtered[filtered["verdict"].isin(chosen_verdicts)]

    if filtered.empty:
        st.info("No records match the current filters.", icon="🔎")
        return

    # -- headline figures ---------------------------------------------------- #
    section("Yield")
    total = len(filtered)
    passed = int((filtered["verdict"] == "PASS").sum())
    review = int((filtered["verdict"] == "REVIEW").sum())
    failed = int((filtered["verdict"] == "FAIL").sum())
    columns = st.columns(5)
    columns[0].metric("Boards recorded", total)
    columns[1].metric("Yield", f"{100.0 * passed / total:.1f}%")
    columns[2].metric("Failed", failed)
    columns[3].metric("For review", review)
    columns[4].metric("Defects per board", f"{filtered['total_defects'].mean():.2f}")

    # -- charts -------------------------------------------------------------- #
    left, right = st.columns([3, 2])
    with left:
        section("Quality over time")
        if "inspected_at" in filtered.columns and filtered["inspected_at"].notna().any():
            trend = (
                filtered.dropna(subset=["inspected_at"])
                .set_index("inspected_at")
                .sort_index()[["quality_score", "total_defects"]]
            )
            st.line_chart(trend, height=260)
            st.caption("Quality score and defect count per inspected board, in order.")
        else:
            st.caption("No timestamps are available to plot.")
    with right:
        section("Verdict split")
        st.bar_chart(
            pd.DataFrame({"count": [passed, review, failed]},
                         index=["PASS", "REVIEW", "FAIL"]),
            height=260,
        )

    section("Records")
    st.dataframe(
        filtered.sort_values("inspected_at", ascending=False)
        if "inspected_at" in filtered.columns else filtered,
        use_container_width=True, hide_index=True, height=360,
    )

    # -- export and maintenance --------------------------------------------- #
    section("Export")
    batch = BatchSummary(
        total_boards=total,
        passed=passed,
        review=review,
        failed=failed,
        total_defects=int(filtered["total_defects"].sum()),
        mean_quality=float(filtered["quality_score"].mean()),
        mean_inference_ms=float(filtered["inference_ms"].mean())
        if "inference_ms" in filtered else 0.0,
        class_counts=_history_class_counts(rows, filtered),
    )
    report_rows = [
        {
            "image": str(row.get("source", "")),
            "verdict": row.get("verdict", ""),
            "quality_score": round(float(row.get("quality_score", 0)), 1),
            "total_defects": int(row.get("total_defects", 0)),
            "critical_defects": int(row.get("critical_defects", 0)),
            "mean_confidence": round(float(row.get("mean_confidence", 0)), 3),
        }
        for row in filtered.head(200).to_dict("records")
    ]
    pdf_bytes, pdf_error = _build_pdf(
        report.build_batch_report,
        batch=batch,
        rows=report_rows,
        config=settings.as_config(),
        run_name=f"Inspection history ({status['kind']})",
    )
    if pdf_error:
        st.warning(f"The PDF report could not be generated — {pdf_error}", icon="⚠️")

    download_row([
        ("📄 PDF history report", pdf_bytes, "inspection_history.pdf", "application/pdf"),
        ("📊 History (CSV)", filtered.to_csv(index=False).encode("utf-8-sig"),
         "inspection_history.csv", "text/csv"),
    ])

    with st.expander("Maintenance"):
        st.caption(
            "Clearing removes every recorded result from the configured store. "
            "This cannot be undone — export the CSV first if the run data matters."
        )
        confirm = st.checkbox("I understand this permanently deletes the history.")
        if st.button("Clear inspection history", disabled=not confirm):
            if store.clear():
                st.success("The inspection history was cleared.", icon="✅")
                st.rerun()
            else:
                st.error(f"Could not clear the history — {store.last_error}", icon="⛔")


def _history_class_counts(rows: list[dict[str, Any]], filtered: pd.DataFrame) -> dict[str, int]:
    """
    Total the per-class defect counts across the filtered records.

    ``class_counts`` was dropped from the display frame (it is a nested
    dictionary), so the totals are taken from the raw rows, restricted to the
    sources that survived filtering.
    """
    if "source" not in filtered.columns:
        return {}
    kept = set(filtered["source"].astype(str))
    totals: dict[str, int] = {}
    for row in rows:
        if str(row.get("source", "")) not in kept:
            continue
        counts = row.get("class_counts") or {}
        if isinstance(counts, dict):
            for name, count in counts.items():
                try:
                    totals[name] = totals.get(name, 0) + int(count)
                except (TypeError, ValueError):
                    continue
    return totals


# --------------------------------------------------------------------------- #
# Page 6 — system status
# --------------------------------------------------------------------------- #
def page_status(bridge: Pipeline, detector, settings: Settings, store) -> None:
    """Diagnostics: what resolved, what did not, and what to do about it."""
    st.markdown("#### System status")
    st.caption(
        "Which modules the interface managed to reach, and what to fix when one "
        "of them is unavailable."
    )

    bridge_status = bridge.status()
    detector_status = detector.status()
    store_status = store.status()

    remote_pipeline = settings.uses_remote_pipeline
    pipeline_kind = "remote API" if remote_pipeline else "local module"

    section("Module integration")
    status_row(
        f"Processing pipeline ({pipeline_kind})",
        bridge_status["available"],
        f"{bridge_status['endpoint']}"
        + (f" · {bridge_status['service']}" if bridge_status.get("service") else "")
        if bridge_status["available"]
        else (bridge_status["error"] or "Not reachable."),
    )
    status_row(
        "Module 1 — image acquisition & pre-processing",
        bridge_status["module1_ready"],
        ("The service reports the pre-processing stage as available."
         if remote_pipeline else "image_pipeline.preprocess_image resolved.")
        if bridge_status["module1_ready"]
        else ("The service did not offer a pre-processing stage."
              if remote_pipeline else "image_pipeline.preprocess_image could not be resolved."),
    )
    status_row(
        "Module 2 — image alignment & calibration",
        bridge_status["module2_ready"],
        ("The service reports the alignment stage as available."
         if remote_pipeline else "image_pipeline.align_image resolved.")
        if bridge_status["module2_ready"]
        else ("The service did not offer an alignment stage."
              if remote_pipeline else "image_pipeline.align_image could not be resolved."),
    )
    status_row(
        f"Module 3 — PCB defect detection ({'remote API' if settings.uses_remote else 'local checkpoint'})",
        detector_status["available"],
        f"{detector_status['backend']} · {detector_status['weights']}"
        if detector_status["available"]
        else (detector_status["error"] or "No model loaded."),
    )
    if settings.has_fast_model:
        fast_status = _detector_for(settings, fast=True).status()
        status_row(
            "Fast model for video and live pages",
            fast_status["available"],
            f"{settings.fast_weights_path.name} — {fast_status['backend']}"
            if fast_status["available"]
            else (fast_status["error"] or "Not loaded."),
        )
    status_row(
        f"Inspection history ({store_status['kind']})",
        store.available if store_status["kind"] != "none" else None,
        store_status.get("error") or store_status.get("note") or "",
    )
    status_row(
        "Teammates' notebook functions (optional execution mode)",
        True if bridge_status["notebook_mode_ready"] else None,
        "The teammates' original notebook functions are available and can be "
        "executed directly."
        if bridge_status["notebook_mode_ready"]
        else (bridge_status["notebook_error"] or
              "Not loaded yet — it is resolved lazily the first time notebook mode runs."),
    )

    if remote_pipeline and bridge_status.get("health"):
        with st.expander("Pipeline service health response"):
            st.json(bridge_status["health"])
    elif remote_pipeline and not bridge_status["available"]:
        st.warning(
            "The processing service is not reachable, so every board is being "
            "detected on its raw pixels. The wire format the client expects is "
            "specified in `docs/API_CONTRACT_PIPELINE.md`, which includes a "
            "runnable FastAPI reference implementation.",
            icon="⚠️",
        )

    section("Detector configuration")
    if detector_status["available"]:
        columns = st.columns(3)
        columns[0].metric("Backend", detector_status["backend"])
        columns[1].metric("Classes", detector_status["num_classes"])
        columns[2].metric("Source", "remote" if settings.uses_remote else "local")
        if settings.uses_remote and detector_status.get("remote_health"):
            with st.expander("Service health response"):
                st.json(detector_status["remote_health"])
        st.write("**Class order**")
        st.code("\n".join(
            f"{index}: {name}" for index, name in enumerate(detector_status["classes"])
        ), language="text")
    elif settings.uses_remote:
        st.warning(
            f"The inference service is not reachable — {detector_status['error']} "
            "The wire format the client expects is specified in "
            "`docs/API_CONTRACT.md`, which includes a runnable FastAPI "
            "reference implementation.",
            icon="⚠️",
        )
    else:
        st.warning(
            "No detection model is loaded, so the interface will show the "
            "pre-processing and alignment stages but report no defects. Select a "
            "checkpoint in the sidebar. Module 3 writes its trained weights to "
            "`Student3-Defect Detection/runs/<run>/weights/best.pt`.",
            icon="⚠️",
        )

    section("Cameras")
    cameras = _camera_list()
    if cameras:
        st.success(
            f"{len(cameras)} camera(s) available for continuous live capture: "
            + ", ".join(f"index {i}" for i in cameras),
            icon="📷",
        )
    else:
        st.info(
            "No camera is attached to the machine running this interface. "
            "Snapshot mode on the Live page still works, because it captures "
            "through the browser.",
            icon="📷",
        )

    section("Discovered datasets")
    st.caption(
        "Optional. Nothing in the inspection path reads from disk — these "
        "folders exist so the sample pickers have something to offer. The folder "
        "searched is set under **Sample data folder** in the sidebar"
        + (f", currently `{bridge_status['workspace']}`." if bridge_status.get("workspace") else ".")
    )
    folders = bridge.dataset_folders()
    if folders:
        rows = [
            {"dataset": label, "path": str(path), "images": len(list_images(path, limit=5000))}
            for label, path in folders.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No dataset folders were found, which is the ordinary state for a "
            "standalone checkout — this repository ships no image data. Point "
            "**Sample data folder** at a checkout of the shared repository to "
            "browse `PCB_DATASET/`, `Clean_Dataset/`, `Preprocessed_Dataset/` "
            "and `Calibrated_Dataset/` from the pickers. Uploading a board works "
            "either way.",
            icon="ℹ️",
        )

    section("Environment")
    st.dataframe(pd.DataFrame(_environment_rows()), use_container_width=True, hide_index=True)

    section("Active configuration")
    config = settings.as_config(detector_status)
    st.dataframe(
        pd.DataFrame({"setting": list(config), "value": [str(v) for v in config.values()]}),
        use_container_width=True,
        hide_index=True,
    )


def _environment_rows() -> list[dict[str, str]]:
    """Report the versions of the libraries the system depends on."""
    rows = [{"component": "Python", "version": sys.version.split()[0], "status": "installed"}]
    for label, module_name in [
        ("Streamlit", "streamlit"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("pandas", "pandas"),
        ("requests", "requests"),
        ("Ultralytics", "ultralytics"),
        ("PyTorch", "torch"),
        ("torchvision", "torchvision"),
        ("ReportLab", "reportlab"),
        ("Supabase", "supabase"),
    ]:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            rows.append({"component": label, "version": str(version), "status": "installed"})
        except Exception:                                    # noqa: BLE001
            rows.append({"component": label, "version": "—", "status": "not installed"})
    return rows


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=60)
def _camera_list() -> list[int]:
    """
    Cache the camera probe.

    Probing opens and closes each device, which is slow enough to be noticeable
    on every re-run, and the set of attached cameras rarely changes within a
    minute.
    """
    return live.list_cameras()


def _fast_model_note(settings: Settings) -> None:
    """Tell the operator which checkpoint this page will use, and why."""
    if settings.has_fast_model:
        st.caption(
            f"· Using the fast model **{settings.fast_weights_path.name}** on this "
            "page; the accurate model stays in use for single-board and batch "
            "inspection."
        )
    elif not settings.uses_remote and settings.weights_path is not None:
        st.caption(
            "· Detection dominates the cost of a stream. A lighter checkpoint can "
            "be selected in the sidebar under **Fast model for video and live "
            "pages** — on this dataset YOLOv10n runs about nine times faster than "
            "RT-DETR-L, at some cost in recall."
        )


def _sample_picker(bridge: Pipeline, key: str) -> Path | None:
    """Offer a board from the datasets that Modules 1 and 2 produced."""
    folders = bridge.dataset_folders()
    if not folders:
        return None

    with st.expander("…or pick a sample from the project datasets"):
        label = st.selectbox("Dataset", list(folders), key=f"{key}_dataset")
        images = list_images(folders[label], recursive=True, limit=400)
        if not images:
            st.caption("That folder holds no images.")
            return None
        root = folders[label]
        names = [str(path.relative_to(root)) for path in images]
        chosen = st.selectbox(f"Image ({len(images)} available)", names, key=f"{key}_image")
        return root / chosen


def _folder_picker(bridge: Pipeline, key: str) -> str:
    """Choose a source folder, defaulting to a dataset the project produced."""
    folders = bridge.dataset_folders()
    options = list(folders) + ["Enter a path manually…"]
    choice = st.selectbox("Dataset folder", options, key=f"{key}_choice")
    if choice != "Enter a path manually…":
        return str(folders[choice])
    return st.text_input(
        "Folder path", value="", key=f"{key}_manual",
        placeholder=r"C:\Users\...\Preprocessed_Dataset\Missing_hole",
    ).strip().strip('"')


def _video_picker(bridge: Pipeline) -> Path | None:
    """Offer the conveyor video that Module 1 generated, if it is present."""
    root = bridge.project_root
    if root is None:
        return None
    videos = sorted(root.glob("**/*.mp4"))
    videos = [v for v in videos if ".git" not in v.parts][:20]
    if not videos:
        return None

    with st.expander("…or use a video from the project"):
        names = [str(path.relative_to(root)) for path in videos]
        chosen = st.selectbox("Project video", ["None"] + names, key="video_sample")
        if chosen == "None":
            return None
        return root / chosen


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Configure the page, build the backends, and route to the six pages."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_header(APP_TITLE)

    settings = render_sidebar()
    bridge = _pipeline_for(settings)
    detector = _detector_for(settings)
    fast_detector = _detector_for(settings, fast=True)
    store = _store_for(settings)

    if not bridge.available:
        remedy = (
            "Check the service URL in the sidebar under **Processing pipeline** — "
            "the wire format the client expects is documented in "
            "docs/API_CONTRACT_PIPELINE.md."
            if settings.uses_remote_pipeline
            else "Point the sample data folder at a checkout of the shared "
                 "repository, or switch the pipeline source to Remote API."
        )
        st.error(
            f"Modules 1 and 2 are unavailable — {bridge.load_error} {remedy} "
            "Detection will still run, but on unprocessed images.",
            icon="⛔",
        )
    if not detector.available:
        remedy = (
            "Check the service URL in the sidebar — the wire format the client "
            "expects is documented in docs/API_CONTRACT.md."
            if settings.detector_source == SOURCE_REMOTE
            else "Select a checkpoint in the sidebar."
        )
        st.warning(
            f"Module 3 is not loaded — {detector.load_error} {remedy} "
            "The pre-processing and alignment stages remain fully usable in the "
            "meantime.",
            icon="⚠️",
        )

    tabs = st.tabs([
        "🔍  Single inspection",
        "📁  Batch inspection",
        "🎞  Video stream",
        "📷  Live inspection",
        "📋  History",
        "⚙️  System status",
    ])

    with tabs[0]:
        page_single(bridge, detector, settings, store)
    with tabs[1]:
        page_batch(bridge, detector, settings, store)
    with tabs[2]:
        page_video(bridge, fast_detector, settings, store)
    with tabs[3]:
        page_live(bridge, fast_detector, settings, store)
    with tabs[4]:
        page_history(store, settings)
    with tabs[5]:
        page_status(bridge, detector, settings, store)


if __name__ == "__main__":
    main()
