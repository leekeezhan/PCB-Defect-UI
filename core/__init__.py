"""
core
====
Backend package for Module 4 (User Interface Module).

The user interface layer (``app.py``) contains no image-processing logic of its
own. Every algorithmic step is delegated to a teammate's module through the
adapters in this package, which keeps the presentation layer decoupled from the
implementation details of Modules 1-3 — and lets this repository stand alone,
with the other three modules reached over HTTP.

    pipeline_remote : adapter for Module 1 (pre-processing) and Module 2
                      (alignment and calibration) over HTTP. The default.
                      Wire format: ``docs/API_CONTRACT_PIPELINE.md``.
    pipeline_bridge : ``StageResult``, the container both pipeline adapters
                      fill; the local adapter for a checkout of the shared
                      repository on the same machine; and ``create_pipeline``,
                      the factory that picks between the two.
    remote          : adapter for Module 3 over HTTP. The default.
                      Wire format: ``docs/API_CONTRACT.md``.
    detector        : adapter for Module 3 loading a checkpoint in-process, for
                      when the weights are on this machine. Same interface.
    workspace       : resolves the optional folder of datasets and checkpoints
                      that the sample pickers offer.
    analysis        : converts raw detections into inspection statistics and a
                      pass/fail verdict.
    viz             : renders annotated images for display.
    report          : builds the downloadable PDF inspection reports.
    video           : frame-by-frame execution of the pipeline over a recorded video.
    live            : the same, over a camera attached to this machine.
    storage         : persists every verdict, so the History page can report
                      yield over time rather than one board at a time.
"""

__all__ = [
    "analysis",
    "detector",
    "live",
    "pipeline_bridge",
    "pipeline_remote",
    "remote",
    "report",
    "storage",
    "video",
    "viz",
    "workspace",
]
