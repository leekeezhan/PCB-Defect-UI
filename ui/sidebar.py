"""
sidebar.py
==========
The single control panel for the whole system.

Every setting that changes how a board is inspected lives here rather than being
scattered across the pages, so the same configuration applies to the single
image, the batch run, the video stream and the live camera — and so the
configuration recorded in the PDF report is guaranteed to be the one that
produced the result.

The panel is organised into collapsible sections because it now covers five
concerns: where the pre-processing and alignment stages run, where detection
runs, how sensitive detection should be, what counts as an acceptable board, and
where the inspection history is kept.

Both upstream boundaries — Modules 1 & 2, and Module 3 — default to a remote
service, because this repository holds Module 4 alone and the other three
modules are developed and deployed by their own owners. Each can be pointed at
something on this machine instead, which is what makes a demonstration possible
without the teammates' services running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import streamlit as st

from core.analysis import DEFAULT_CRITICAL_CLASSES, DEFAULT_SEVERITY, InspectionCriteria
from core.detector import discover_weights, is_pretrained_stock, read_class_names_from_yaml
from core.pipeline_bridge import MODE_MODULE, MODE_NOTEBOOK
from core.storage import DEFAULT_SQLITE_NAME, DEFAULT_TABLE, STORE_NONE, STORE_SQLITE, STORE_SUPABASE
from core.workspace import ENV_WORKSPACE, workspace_root

_MANUAL_ENTRY = "Enter a path manually…"
_NO_FAST_MODEL = "Use the primary model"

#: Detector sources offered in the sidebar. Also used for the pipeline source,
#: which offers the same two choices.
SOURCE_LOCAL = "local"
SOURCE_REMOTE = "remote"


@dataclass
class Settings:
    """Everything the pages need to know about how to run an inspection."""

    # -- where detection runs ---------------------------------------------- #
    detector_source: str = SOURCE_REMOTE
    weights_path: Path | None = None
    fast_weights_path: Path | None = None
    device: str | None = None
    remote_url: str = ""
    remote_key: str = ""
    remote_timeout: float = 30.0
    remote_verify_tls: bool = True

    # -- how sensitive ------------------------------------------------------ #
    confidence: float = 0.25
    iou: float = 0.45

    # -- which upstream modules, and where they run -------------------------- #
    pipeline_source: str = SOURCE_REMOTE
    pipeline_url: str = ""
    pipeline_key: str = ""
    pipeline_timeout: float = 30.0
    pipeline_verify_tls: bool = True
    mode: str = MODE_MODULE
    do_preprocess: bool = True
    do_align: bool = True

    # -- where sample data lives (optional) ---------------------------------- #
    workspace: str = ""

    # -- what counts as acceptable ------------------------------------------ #
    criteria: InspectionCriteria = field(default_factory=InspectionCriteria)

    # -- display ------------------------------------------------------------ #
    show_labels: bool = True
    show_confidence: bool = True

    # -- history ------------------------------------------------------------ #
    store_kind: str = STORE_NONE
    sqlite_path: str = DEFAULT_SQLITE_NAME
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_table: str = DEFAULT_TABLE

    class_names: tuple[str, ...] = field(default_factory=tuple)

    # -- helpers ------------------------------------------------------------ #
    @property
    def uses_remote(self) -> bool:
        """True when Module 3 runs as a service rather than in this process."""
        return self.detector_source == SOURCE_REMOTE

    @property
    def uses_remote_pipeline(self) -> bool:
        """True when Modules 1 and 2 run as a service rather than in this process."""
        return self.pipeline_source == SOURCE_REMOTE

    @property
    def pipeline_label(self) -> str:
        """Short description of the active pipeline source, for the chrome."""
        if self.uses_remote_pipeline:
            return self.pipeline_url or "remote service"
        return "local image_pipeline.py"

    @property
    def has_fast_model(self) -> bool:
        """
        True when a separate, faster checkpoint is configured for the video and
        live pages. Only meaningful for locally loaded checkpoints — a remote
        service decides its own model.
        """
        return self.detector_source == SOURCE_LOCAL and self.fast_weights_path is not None

    def detector_label(self) -> str:
        """Short description of the active detector, for the interface chrome."""
        if self.uses_remote:
            return self.remote_url or "remote service"
        return self.weights_path.name if self.weights_path else "none"

    def as_config(self, detector_status: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Flatten the settings into the traceability table printed in the PDF
        report, so a report can always be tied back to the run that produced it.
        """
        status = detector_status or {}
        config: dict[str, Any] = {
            "Detector source": "remote API" if self.uses_remote else "local checkpoint",
            "Detector weights": self.detector_label(),
            "Detector backend": status.get("backend") or "unavailable",
        }
        if self.uses_remote and status.get("remote_model"):
            config["Remote model"] = status["remote_model"]
        if self.has_fast_model:
            config["Fast model (video / live)"] = self.fast_weights_path.name

        config.update(
            {
                "Compute device": self.device or "auto",
                "Confidence threshold": f"{self.confidence:.2f}",
                "NMS IoU threshold": f"{self.iou:.2f}",
                "Pipeline source": "remote API" if self.uses_remote_pipeline else "local module",
                "Pipeline endpoint": self.pipeline_label,
                "Pipeline mode": self.mode,
                "Module 1 (pre-processing)": "enabled" if self.do_preprocess else "bypassed",
                "Module 2 (alignment)": "enabled" if self.do_align else "bypassed",
                "Defect allowance": str(self.criteria.max_defects),
                "Critical classes": ", ".join(self.criteria.critical_classes) or "none",
                "Review confidence": f"{self.criteria.review_confidence:.2f}",
                "Inspection history": self.store_kind,
            }
        )
        if status.get("classes"):
            config["Detector classes"] = ", ".join(status["classes"])
        return config


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_sidebar() -> Settings:
    """
    Draw the sidebar and collect the current configuration.

    The sidebar is drawn *before* any adapter is built, because the adapters are
    configured from what it returns — which service to call, which checkpoint to
    load, which folder to offer samples from.

    Returns:
        The :class:`Settings` in force for this script run.
    """
    with st.sidebar:
        st.markdown(
            '<div class="pcb-sidebar-title">Inspection settings</div>'
            '<div class="pcb-sidebar-note">These apply to every page.</div>',
            unsafe_allow_html=True,
        )

        pipeline = _pipeline_section()
        workspace = _workspace_section()

        # The weight picker, the class list and the default history location all
        # look inside the data workspace, so it has to be resolved first.
        root = workspace_root(workspace["workspace"])

        detector = _detector_section(root)
        thresholds = _threshold_section()
        criteria, available_classes = _criteria_section(root)
        display = _display_section()
        history = _history_section(root)

        st.divider()
        st.caption(
            "Module 4 — User Interface. The interface performs no image "
            "processing of its own; every stage is delegated to Modules 1-3."
        )

    return Settings(
        **detector,
        **thresholds,
        **pipeline,
        **workspace,
        criteria=criteria,
        **display,
        **history,
        class_names=tuple(available_classes),
    )


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _detector_section(root: Path | None) -> dict[str, Any]:
    """Choose where detection runs, and on which checkpoint."""
    st.divider()
    st.markdown("**Detection model (Module 3)**")

    source_label = st.radio(
        "Detector source",
        ["Remote API", "Local checkpoint"],
        index=0,
        horizontal=True,
        help="'Remote API' forwards each image to the team's inference service, "
             "which is how this repository normally runs and keeps this machine "
             "free of a multi-gigabyte PyTorch install. 'Local checkpoint' loads "
             "the trained weights into this process instead, which needs the "
             ".pt file on this machine but no running service.",
    )
    source = SOURCE_LOCAL if source_label == "Local checkpoint" else SOURCE_REMOTE

    settings: dict[str, Any] = {
        "detector_source": source,
        "weights_path": None,
        "fast_weights_path": None,
        "device": None,
        "remote_url": "",
        "remote_key": "",
        "remote_timeout": 30.0,
        "remote_verify_tls": True,
    }

    if source == SOURCE_LOCAL:
        candidates = discover_weights(root)
        settings["weights_path"] = _weights_picker(candidates, root)
        settings["fast_weights_path"] = _fast_weights_picker(candidates, root, settings["weights_path"])
        settings["device"] = _device_picker()
    else:
        settings.update(_remote_picker())

    return settings


def _weights_picker(candidates: list[Path], root: Path | None) -> Path | None:
    """
    Offer the checkpoints found in the repository, plus a manual path field.

    Module 3's ``runs/`` folder is excluded from version control, so on a fresh
    clone there may be nothing to find. That is treated as an ordinary state:
    the picker explains where the weights are expected and the rest of the
    interface stays usable.
    """
    if candidates:
        labels = [_weight_label(path, root) for path in candidates] + [_MANUAL_ENTRY]
        choice = st.selectbox(
            "Primary model", labels, index=0,
            help="Used by the single-board and batch pages, where accuracy "
                 "matters more than speed. On this dataset RT-DETR-L is the more "
                 "accurate of the two trained architectures.",
        )
        if choice != _MANUAL_ENTRY:
            selected = candidates[labels.index(choice)]
            if is_pretrained_stock(selected):
                st.warning(
                    "This is a stock pre-trained download, not a checkpoint trained "
                    "on the PCB dataset. It will predict generic object classes "
                    "rather than defect types — useful for verifying the interface, "
                    "not for inspection.",
                    icon="⚠️",
                )
            return selected
    else:
        st.info(
            "No checkpoint was found in the repository. Module 3 writes its "
            "weights to `Student3-Defect Detection/runs/…/weights/best.pt`, which "
            "is excluded from version control — copy that file in, or point to it "
            "below.",
            icon="ℹ️",
        )

    manual = st.text_input(
        "Path to a .pt / .pth checkpoint",
        value="",
        placeholder=r"C:\Users\...\runs\detect\train\weights\best.pt",
        help="Any Ultralytics (YOLO, RT-DETR) or torchvision (Faster R-CNN) "
             "checkpoint produced by Module 3.",
    ).strip().strip('"')

    if not manual:
        return None
    path = Path(manual)
    if not path.is_file():
        st.error(f"No file at: {path}", icon="⛔")
        return None
    return path


def _fast_weights_picker(
    candidates: list[Path], root: Path | None, primary: Path | None
) -> Path | None:
    """
    Offer a second, faster checkpoint for the video and live pages.

    The measured trade-off on this dataset is stark: RT-DETR-L reaches mAP@0.5
    of 0.669 at 2.1 FPS on CPU, while YOLOv10n reaches 0.566 at 19.2 FPS. Neither
    is the right answer for every page — accuracy wins on a still board, and
    throughput wins on a moving stream — so both can be configured and each page
    uses the one that suits it.
    """
    if not candidates:
        return None

    with st.expander("Fast model for video and live pages"):
        st.caption(
            "Detection dominates the cost of a stream. Selecting a lighter "
            "checkpoint here lets the video and live pages run at a usable frame "
            "rate while the single-board and batch pages keep the accurate model."
        )
        labels = [_NO_FAST_MODEL] + [_weight_label(path, root) for path in candidates]
        choice = st.selectbox("Fast model", labels, index=0)
        if choice == _NO_FAST_MODEL:
            return None
        selected = candidates[labels.index(choice) - 1]
        if primary is not None and selected == primary:
            st.caption("This is the primary model, so the pages behave identically.")
            return None
        return selected


def _remote_picker() -> dict[str, Any]:
    """Collect the inference service's address and credentials."""
    url = st.text_input(
        "Service base URL",
        value=st.session_state.get("_remote_url", "http://127.0.0.1:8000"),
        key="_remote_url",
        help="The service root, not an endpoint — the client appends /health "
             "and /predict itself. See docs/API_CONTRACT.md.",
    ).strip()

    key = st.text_input(
        "API key (optional)", value="", type="password",
        help="Sent as an Authorization: Bearer header. Leave empty for a local "
             "service with no authentication.",
    ).strip()

    with st.expander("Connection options"):
        timeout = st.slider(
            "Request timeout (seconds)", 5, 120, 30, 5,
            help="A batch run makes one request per board, so a slow service "
                 "needs a generous timeout here.",
        )
        verify = not st.checkbox(
            "Allow an untrusted TLS certificate", value=False,
            help="Only for a development server with a self-signed certificate.",
        )

    return {
        "remote_url": url,
        "remote_key": key,
        "remote_timeout": float(timeout),
        "remote_verify_tls": bool(verify),
    }


def _weight_label(path: Path, root: Path | None) -> str:
    """Human-readable label for a checkpoint in the picker."""
    try:
        relative = path.relative_to(root) if root else path
    except ValueError:
        relative = path
    marker = "  (stock pre-trained)" if is_pretrained_stock(path) else ""
    return f"{relative}{marker}"


def _device_picker() -> str | None:
    """Choose the compute device, showing whether CUDA is actually available."""
    cuda_available = False
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:                                        # noqa: BLE001
        cuda_available = False

    options = ["Automatic", "CPU"] + (["GPU (CUDA)"] if cuda_available else [])
    choice = st.selectbox(
        "Compute device", options, index=0,
        help=("A CUDA GPU was detected." if cuda_available
              else "No CUDA GPU was detected; inference will run on the CPU."),
    )
    return {"Automatic": None, "CPU": "cpu", "GPU (CUDA)": "cuda"}[choice]


def _threshold_section() -> dict[str, Any]:
    """Detection sensitivity."""
    st.divider()
    st.markdown("**Detection thresholds**")
    confidence = st.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05,
        help="A detection is reported only when the detector scores it at least "
             "this highly. Raise it to suppress false alarms, lower it to catch "
             "faint defects.",
    )
    iou = st.slider(
        "NMS IoU threshold", 0.10, 0.90, 0.45, 0.05,
        help="How much two boxes may overlap before the lower-scoring one is "
             "suppressed. The NMS-free architectures ignore this internally.",
    )
    return {"confidence": float(confidence), "iou": float(iou)}


def _pipeline_section() -> dict[str, Any]:
    """Which upstream modules to run, where they run, and how to reach them."""
    st.markdown("**Processing pipeline (Modules 1 & 2)**")

    source_label = st.radio(
        "Pipeline source",
        ["Remote API", "Local module"],
        index=0,
        horizontal=True,
        help="'Remote API' forwards each image to the team's processing service, "
             "which is how this repository normally runs — Modules 1 and 2 live "
             "in their own repositories and are called over HTTP. 'Local module' "
             "imports image_pipeline.py from a checkout of the shared repository "
             "on this machine, which is useful for working offline.",
    )
    source = SOURCE_REMOTE if source_label == "Remote API" else SOURCE_LOCAL

    settings: dict[str, Any] = {
        "pipeline_source": source,
        "pipeline_url": "",
        "pipeline_key": "",
        "pipeline_timeout": 30.0,
        "pipeline_verify_tls": True,
    }

    if source == SOURCE_REMOTE:
        settings.update(_pipeline_remote_picker())

    mode_label = st.radio(
        "Execution mode",
        ["Shared module functions", "Teammates' notebook functions"],
        index=0,
        help="'Shared module functions' runs the plain functions the two modules "
             "export and is the faster, more robust path. 'Teammates' notebook "
             "functions' executes the original notebook cells instead, proving "
             "the system drives the teammates' own code. The choice is forwarded "
             "to the service, which may decline to honour it.",
    )
    do_preprocess = st.checkbox(
        "Run Module 1 — pre-processing", value=True,
        help="Bicubic resize, colour normalisation and CLAHE contrast "
             "enhancement. Switch off when the input already comes from "
             "Preprocessed_Dataset.",
    )
    do_align = st.checkbox(
        "Run Module 2 — alignment and calibration", value=True,
        help="Detects the board boundary and warps it to a top-down view. If no "
             "clean boundary is found the pre-processed image is used instead, "
             "and the interface says so.",
    )
    settings.update(
        {
            "mode": MODE_MODULE if mode_label.startswith("Shared") else MODE_NOTEBOOK,
            "do_preprocess": bool(do_preprocess),
            "do_align": bool(do_align),
        }
    )
    return settings


def _pipeline_remote_picker() -> dict[str, Any]:
    """Collect the processing service's address and credentials."""
    url = st.text_input(
        "Pipeline service base URL",
        value=st.session_state.get("_pipeline_url", "http://127.0.0.1:8100"),
        key="_pipeline_url",
        help="The service root, not an endpoint — the client appends /health and "
             "/process itself. See docs/API_CONTRACT_PIPELINE.md.",
    ).strip()

    with st.expander("Pipeline connection options"):
        key = st.text_input(
            "API key (optional)", value="", type="password", key="_pipeline_key",
            help="Sent as an Authorization: Bearer header. Leave empty for a "
                 "local service with no authentication.",
        ).strip()
        timeout = st.slider(
            "Request timeout (seconds)", 5, 120, 30, 5, key="_pipeline_timeout",
            help="One request per board on the batch page, and one per sampled "
                 "frame on the video and live pages.",
        )
        verify = not st.checkbox(
            "Allow an untrusted TLS certificate", value=False, key="_pipeline_tls",
            help="Only for a development server with a self-signed certificate.",
        )

    return {
        "pipeline_url": url,
        "pipeline_key": key,
        "pipeline_timeout": float(timeout),
        "pipeline_verify_tls": bool(verify),
    }


def _workspace_section() -> dict[str, Any]:
    """
    Where sample data lives on this machine.

    Nothing in the inspection path needs it — every stage arrives over the
    network. It exists so the sample pickers, the checkpoint picker and the
    project-video picker have something to offer when a checkout of the shared
    repository is on the same machine.
    """
    st.divider()
    with st.expander("Sample data folder (optional)"):
        st.caption(
            "Point this at a checkout of the shared repository to browse "
            "`PCB_DATASET/`, `Preprocessed_Dataset/` and the trained checkpoints "
            f"from the pickers. It is resolved from the `{ENV_WORKSPACE}` "
            "environment variable, or a sibling checkout, when left empty."
        )
        default = str(workspace_root(None) or "")
        folder = st.text_input(
            "Folder", value=st.session_state.get("_workspace", default),
            key="_workspace",
            placeholder=r"C:\Users\...\Image-Processing",
        ).strip().strip('"')
        resolved = workspace_root(folder)
        if folder and resolved is not None and str(resolved) != folder:
            st.warning(f"No folder at: {folder} — using {resolved} instead.", icon="⚠️")
    return {"workspace": folder}


def _criteria_section(root: Path | None) -> tuple[InspectionCriteria, tuple[str, ...]]:
    """What counts as an acceptable board."""
    st.divider()
    st.markdown("**Acceptance criteria**")
    max_defects = st.number_input(
        "Defects allowed per board", min_value=0, max_value=50, value=0, step=1,
        help="A board carrying more confident defects than this is failed. Zero "
             "is a zero-tolerance policy.",
    )
    available_classes = read_class_names_from_yaml(root) or tuple(DEFAULT_SEVERITY)
    critical_classes = st.multiselect(
        "Critical defect types",
        options=list(available_classes),
        default=[c for c in DEFAULT_CRITICAL_CLASSES if c in available_classes],
        help="These fail a board on sight, whatever the allowance, because they "
             "break electrical continuity.",
    )
    review_confidence = st.slider(
        "Manual-review confidence", 0.10, 0.95, 0.50, 0.05,
        help="Findings scoring below this are treated as uncertain. A board whose "
             "only findings are uncertain is sent for review rather than failed "
             "automatically.",
    )
    criteria = InspectionCriteria(
        max_defects=int(max_defects),
        critical_classes=tuple(critical_classes),
        review_confidence=float(review_confidence),
    )
    return criteria, available_classes


def _display_section() -> dict[str, Any]:
    """Annotation preferences."""
    st.divider()
    st.markdown("**Display**")
    return {
        "show_labels": bool(st.checkbox("Label each bounding box", value=True)),
        "show_confidence": bool(st.checkbox("Show confidence on labels", value=True)),
    }


def _history_section(root: Path | None) -> dict[str, Any]:
    """Where inspection results are persisted."""
    st.divider()
    st.markdown("**Inspection history**")

    label = st.radio(
        "Store results in",
        ["Off", "Local file (SQLite)", "Supabase"],
        index=1,
        help="Persisting each verdict is what makes the History page a yield "
             "dashboard rather than a viewer. The local file needs no "
             "configuration and no network; Supabase shares the history across "
             "every machine running the system.",
    )
    kind = {"Off": STORE_NONE, "Local file (SQLite)": STORE_SQLITE, "Supabase": STORE_SUPABASE}[label]

    default_sqlite = str((root / DEFAULT_SQLITE_NAME) if root else DEFAULT_SQLITE_NAME)
    settings: dict[str, Any] = {
        "store_kind": kind,
        "sqlite_path": default_sqlite,
        "supabase_url": "",
        "supabase_key": "",
        "supabase_table": DEFAULT_TABLE,
    }

    if kind == STORE_SQLITE:
        settings["sqlite_path"] = st.text_input(
            "Database file", value=default_sqlite,
            help="Created automatically. Add it to .gitignore — it is run data, "
                 "not source.",
        ).strip() or default_sqlite

    elif kind == STORE_SUPABASE:
        settings["supabase_url"] = st.text_input(
            "Project URL", value="", placeholder="https://abcdefgh.supabase.co",
        ).strip()
        settings["supabase_key"] = st.text_input(
            "API key (anon)", value="", type="password",
            help="Use the anon key together with the row-level-security policies "
                 "in docs/supabase_schema.sql. The service-role key bypasses "
                 "row-level security and does not belong in a desktop application.",
        ).strip()
        settings["supabase_table"] = st.text_input(
            "Table", value=DEFAULT_TABLE,
        ).strip() or DEFAULT_TABLE
        st.caption("Run `docs/supabase_schema.sql` in the Supabase SQL editor first.")

    return settings
