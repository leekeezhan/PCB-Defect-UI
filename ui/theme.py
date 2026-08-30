"""
theme.py
========
Visual styling for the interface.

Streamlit's default look is functional but generic. A small, self-contained
stylesheet is injected once per session to give the inspection system a
consistent industrial appearance: a fixed accent colour, calmer surfaces, and
verdict colours that match the ones drawn on the annotated images by
``core.viz``.

The palette is defined with CSS custom properties and every rule is scoped to a
class of its own, so a Streamlit upgrade that renames internal DOM classes
degrades the styling rather than breaking the interface.
"""

from __future__ import annotations

import streamlit as st

#: Verdict colours, matching ``core.viz.VERDICT_COLOURS``.
VERDICT_HEX: dict[str, str] = {
    "PASS": "#27ae60",
    "REVIEW": "#f39c12",
    "FAIL": "#c0392b",
}

_CSS = """
<style>
:root {
    --pcb-accent:  #0f8f6f;
    --pcb-ink:     #1f2937;
    --pcb-muted:   #6b7280;
    --pcb-surface: #ffffff;
    --pcb-rule:    #e5e7eb;
    --pcb-pass:    #27ae60;
    --pcb-review:  #f39c12;
    --pcb-fail:    #c0392b;
}

/* Tighten the default top padding so the header sits near the top. */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; }

/* ---- Application header ------------------------------------------------ */
.pcb-header {
    display: flex; align-items: center; gap: 0.9rem;
    padding: 1.0rem 1.3rem; margin-bottom: 1.1rem;
    border-radius: 12px;
    background: linear-gradient(100deg, #0b3d33 0%, #0f8f6f 100%);
    color: #ffffff;
}
.pcb-header .pcb-mark { font-size: 1.9rem; line-height: 1; }
.pcb-header h1 { font-size: 1.35rem; margin: 0; font-weight: 700; color: #ffffff; }
.pcb-header p  { font-size: 0.82rem; margin: 0.15rem 0 0; opacity: 0.88; }

/* ---- Verdict banner ---------------------------------------------------- */
.pcb-verdict {
    border-radius: 10px; padding: 0.95rem 1.2rem; color: #ffffff;
    margin: 0.4rem 0 0.9rem;
}
.pcb-verdict .pcb-verdict-label { font-size: 1.45rem; font-weight: 800; letter-spacing: 0.04em; }
.pcb-verdict .pcb-verdict-note  { font-size: 0.85rem; opacity: 0.92; margin-top: 0.15rem; }
.pcb-verdict--pass   { background: var(--pcb-pass); }
.pcb-verdict--review { background: var(--pcb-review); }
.pcb-verdict--fail   { background: var(--pcb-fail); }

/* ---- Stage chips (Module 1 / 2 / 3 progress) --------------------------- */
.pcb-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.2rem 0 0.9rem; }
.pcb-chip {
    font-size: 0.74rem; font-weight: 600; padding: 0.28rem 0.7rem;
    border-radius: 999px; border: 1px solid var(--pcb-rule); color: var(--pcb-muted);
    background: #f9fafb; white-space: nowrap;
}
.pcb-chip--ok   { border-color: #a7e5c9; background: #eafaf3; color: #12694f; }
.pcb-chip--warn { border-color: #fbe1b0; background: #fef7ea; color: #8a5a08; }
.pcb-chip--off  { border-color: var(--pcb-rule); background: #f3f4f6; color: #9ca3af; }

/* ---- Defect colour legend ---------------------------------------------- */
.pcb-legend { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0 0.2rem; }
.pcb-legend-item {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.78rem; padding: 0.2rem 0.6rem 0.2rem 0.35rem;
    border: 1px solid var(--pcb-rule); border-radius: 999px; background: var(--pcb-surface);
}
.pcb-swatch { width: 0.72rem; height: 0.72rem; border-radius: 3px; display: inline-block; }
.pcb-legend-count { color: var(--pcb-muted); font-variant-numeric: tabular-nums; }

/* ---- Section headings -------------------------------------------------- */
.pcb-section {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--pcb-muted);
    border-bottom: 1px solid var(--pcb-rule);
    padding-bottom: 0.35rem; margin: 1.4rem 0 0.7rem;
}

/* ---- Status rows on the System status page ----------------------------- */
.pcb-status {
    display: flex; align-items: flex-start; gap: 0.7rem;
    padding: 0.65rem 0.85rem; margin-bottom: 0.45rem;
    border: 1px solid var(--pcb-rule); border-left-width: 4px; border-radius: 8px;
    background: var(--pcb-surface);
}
.pcb-status--ok   { border-left-color: var(--pcb-pass); }
.pcb-status--warn { border-left-color: var(--pcb-review); }
.pcb-status--bad  { border-left-color: var(--pcb-fail); }
.pcb-status-title { font-weight: 650; font-size: 0.9rem; color: var(--pcb-ink); }
.pcb-status-note  { font-size: 0.79rem; color: var(--pcb-muted); margin-top: 0.1rem;
                    word-break: break-word; }

/* ---- Sidebar ----------------------------------------------------------- */
section[data-testid="stSidebar"] .pcb-sidebar-title {
    font-size: 0.95rem; font-weight: 700; color: var(--pcb-ink); margin-bottom: 0.1rem;
}
section[data-testid="stSidebar"] .pcb-sidebar-note {
    font-size: 0.75rem; color: var(--pcb-muted); margin-bottom: 0.7rem;
}

/* Tab labels a little larger than the default. */
button[data-baseweb="tab"] { font-size: 0.92rem; font-weight: 600; }
</style>
"""


def inject_css() -> None:
    """
    Inject the stylesheet.

    Streamlit rebuilds the whole element tree on every re-run, so the stylesheet
    must be emitted on every run as well — caching it behind a session flag would
    leave the page unstyled from the first interaction onwards. Re-emitting is
    cheap and does not accumulate, because the previous tree is discarded.
    """
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(
    title: str = "PCB Defect Inspection System",
    subtitle: str = (
        "Module 4 — User Interface · orchestrating pre-processing, alignment "
        "and defect detection"
    ),
) -> None:
    """Draw the application header band."""
    st.markdown(
        f"""
        <div class="pcb-header">
            <div class="pcb-mark">&#9635;</div>
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    """Draw a small uppercase section rule."""
    st.markdown(f'<div class="pcb-section">{label}</div>', unsafe_allow_html=True)
