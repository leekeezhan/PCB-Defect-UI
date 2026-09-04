"""
components.py
=============
Small reusable pieces of interface, kept out of ``app.py`` so each page reads as
a sequence of intentions rather than a wall of markup.

Every component takes plain data (a summary, a list of detections, a status
dictionary) and renders it. None of them perform any computation of their own.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import html
import pandas as pd
import streamlit as st

from core.analysis import InspectionSummary
from core.detector import Detection
from core.pipeline_bridge import StageResult
from core.viz import legend_entries


def verdict_banner(summary: InspectionSummary) -> None:
    """
    Draw the coloured pass / review / fail banner for one board.

    Args:
        summary: the inspection outcome to display.
    """
    modifier = {"PASS": "pass", "REVIEW": "review", "FAIL": "fail"}.get(
        summary.verdict, "review"
    )
    note = (
        f"Quality score {summary.quality_score:.0f}/100 &nbsp;·&nbsp; "
        f"{summary.total_defects} defect(s) detected &nbsp;·&nbsp; "
        f"inference {summary.inference_ms:.0f} ms"
    )
    st.markdown(
        f"""
        <div class="pcb-verdict pcb-verdict--{modifier}">
            <div class="pcb-verdict-label">{summary.verdict}</div>
            <div class="pcb-verdict-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _align_effective(stages: StageResult) -> bool:
    """
    Whether Module 2 actually changed the image's geometry.

    Delegates to ``StageResult.align_effective``, tolerating its absence: the
    pipeline adapter is held by ``@st.cache_resource``, so right after this
    property is added to the code a cached adapter can still produce instances
    of the previously imported class. Treating that as "effective" keeps the
    old wording until the server is restarted, instead of raising.
    """
    return bool(getattr(stages, "align_effective", True))


def stage_chips(stages: StageResult, detector_ready: bool) -> None:
    """
    Show which pipeline stages actually ran on this image.

    This makes the module boundaries visible to the examiner and tells the
    operator immediately when a stage was skipped or degraded.

    Args:
        stages: the result returned by ``PipelineBridge.run``.
        detector_ready: whether Module 3 had a model loaded.
    """
    def chip(label: str, state: str) -> str:
        return f'<span class="pcb-chip pcb-chip--{state}">{html.escape(label)}</span>'

    chips = [
        chip(
            "Module 1 · Pre-processing " + ("✓" if stages.preprocess_ok else "skipped"),
            "ok" if stages.preprocess_ok else "off",
        ),
        # Module 2 has three outcomes worth distinguishing, not two: it can
        # straighten the board, it can return a board outline that is already
        # square to the camera (so the image is only rescaled — see
        # StageResult.align_effective), or it can find no outline at all.
        # Read defensively: a StageResult built by a pipeline adapter that
        # Streamlit cached before this attribute existed would otherwise raise.
        chip(
            "Module 2 · Alignment " + (
                "✓" if stages.align_ok and _align_effective(stages)
                else "rescaled only" if stages.align_ok
                else "not applied"
            ),
            "ok" if stages.align_ok and _align_effective(stages) else "warn",
        ),
        chip(
            "Module 3 · Detection " + ("✓" if detector_ready else "unavailable"),
            "ok" if detector_ready else "off",
        ),
        chip(f"Mode · {stages.mode}", "ok"),
    ]
    st.markdown(f'<div class="pcb-chips">{"".join(chips)}</div>', unsafe_allow_html=True)

    for note in stages.notes:
        st.caption(f"· {note}")


def defect_legend(detections: Sequence[Detection]) -> None:
    """Draw the colour key for the defect classes present in this image."""
    entries = legend_entries(detections)
    if not entries:
        return
    items = "".join(
        f'<span class="pcb-legend-item">'
        f'<span class="pcb-swatch" style="background:{colour}"></span>'
        f"{html.escape(name)} <span class=\"pcb-legend-count\">×{count}</span>"
        f"</span>"
        for name, colour, count in entries
    )
    st.markdown(f'<div class="pcb-legend">{items}</div>', unsafe_allow_html=True)


def metric_row(summary: InspectionSummary) -> None:
    """Show the four headline figures for one inspected board."""
    columns = st.columns(4)
    columns[0].metric("Verdict", summary.verdict)
    columns[1].metric("Defects found", summary.total_defects)
    columns[2].metric("Quality score", f"{summary.quality_score:.0f}")
    columns[3].metric(
        "Mean confidence",
        f"{summary.mean_confidence:.2f}" if summary.total_defects else "—",
    )


def status_row(title: str, ok: bool | None, note: str = "") -> None:
    """
    One line on the System status page.

    Args:
        title: what is being reported on.
        ok: ``True`` for healthy, ``False`` for broken, ``None`` for a warning
            (present but degraded).
        note: supporting detail, shown smaller underneath.
    """
    state = {True: "ok", False: "bad", None: "warn"}[ok]
    icon = {True: "✅", False: "⛔", None: "⚠️"}[ok]
    note_html = f'<div class="pcb-status-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="pcb-status pcb-status--{state}">
            <div>{icon}</div>
            <div>
                <div class="pcb-status-title">{html.escape(title)}</div>
                {note_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def class_bar_chart(class_counts: dict[str, int], caption: str = "") -> None:
    """
    Plot defect counts per class.

    Args:
        class_counts: mapping of defect type to count.
        caption: optional caption drawn under the chart.
    """
    if not class_counts:
        st.info("No defects were detected, so there is no distribution to plot.")
        return
    frame = pd.DataFrame(
        sorted(class_counts.items(), key=lambda item: -item[1]),
        columns=["defect_type", "count"],
    ).set_index("defect_type")
    st.bar_chart(frame, height=260)
    if caption:
        st.caption(caption)


def download_row(
    items: Iterable[tuple[str, bytes | None, str, str]],
    key_prefix: str = "",
) -> None:
    """
    Lay a set of download buttons out in a single row.

    Args:
        items: ``(label, data, file_name, mime)`` tuples. Entries whose ``data``
            is ``None`` render as a disabled button, so the row keeps its shape
            when, for example, PDF generation is unavailable.
        key_prefix: namespace for the element keys. Each button's key is
            derived from its file name, so a row rendered by a tool that
            appears on more than one page — and every page of this app is
            rendered on every script run — needs a prefix to keep those keys
            unique. Without one, Streamlit raises
            ``StreamlitDuplicateElementKey``.
    """
    items = [item for item in items]
    if not items:
        return
    prefix = f"{key_prefix}_" if key_prefix else ""
    columns = st.columns(len(items))
    for column, (label, data, file_name, mime) in zip(columns, items):
        with column:
            if data is None:
                st.button(label, disabled=True, use_container_width=True,
                          key=f"{prefix}disabled_{file_name}")
            else:
                st.download_button(
                    label,
                    data=data,
                    file_name=file_name,
                    mime=mime,
                    use_container_width=True,
                    key=f"{prefix}download_{file_name}",
                )
