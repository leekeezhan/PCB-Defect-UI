"""
detector.py
===========
Adapter between the user interface (Module 4) and Module 3, the PCB defect
detection module.

Module 3 trains three architectures (YOLO, RT-DETR and Faster R-CNN) and stores
its weights under ``Student3-Defect Detection/runs/``. That folder is excluded
from version control, so the interface must not assume any particular file
exists. This adapter therefore:

* discovers candidate weight files automatically and lets the operator override
  the choice from the sidebar;
* selects the loading backend from the checkpoint rather than from a hard-coded
  assumption (Ultralytics for YOLO/RT-DETR, torchvision for Faster R-CNN);
* reports a clear, actionable message when nothing can be loaded, leaving the
  rest of the interface fully operable.

The rest of Module 4 only ever sees :class:`Detection` objects, so swapping the
detector for a different architecture requires no change outside this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

#: Class order fixed by Module 3's ``dataset/yolo/data.yaml``. Used as a fallback
#: when a checkpoint carries no class names of its own.
DEFAULT_CLASS_NAMES: tuple[str, ...] = (
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
)

#: Backend identifiers.
BACKEND_ULTRALYTICS = "ultralytics"
BACKEND_TORCHVISION = "torchvision"


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Detection:
    """
    One detected defect.

    Coordinates are absolute pixels in the image that was passed to the detector,
    in ``(x1, y1)`` top-left / ``(x2, y2)`` bottom-right order.
    """

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Bounding-box area in square pixels."""
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_row(self) -> dict[str, Any]:
        """Flat representation for tables and CSV export."""
        cx, cy = self.centre
        return {
            "class_id": self.class_id,
            "defect_type": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "x1": int(round(self.x1)),
            "y1": int(round(self.y1)),
            "x2": int(round(self.x2)),
            "y2": int(round(self.y2)),
            "width_px": int(round(self.width)),
            "height_px": int(round(self.height)),
            "area_px2": int(round(self.area)),
            "centre_x": int(round(cx)),
            "centre_y": int(round(cy)),
        }


@dataclass
class DetectionResult:
    """Everything the interface needs about one detector invocation."""

    detections: list[Detection]
    inference_ms: float
    image_shape: tuple[int, int]          # (height, width)
    model_name: str
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.detections)


def filter_by_class(result: DetectionResult, keep: Iterable[str] | None) -> DetectionResult:
    """
    Keep only the detections whose class is in ``keep``.

    Used to honour the sidebar's "Defect types to show" picker: a defect type
    the operator did not select is treated as if it had never been detected —
    it is dropped before the verdict, the quality score, the annotated image
    and the findings table ever see it, not just hidden from the display.

    ``keep`` is normally the full class list (nothing is filtered), but an
    empty selection is a valid, explicit choice too — it means "show nothing",
    not "show everything" — so it is honoured literally rather than treated
    as unset.
    """
    allowed = set(keep or ())
    kept = [d for d in result.detections if d.class_name in allowed]
    return DetectionResult(
        detections=kept,
        inference_ms=result.inference_ms,
        image_shape=result.image_shape,
        model_name=result.model_name,
        error=result.error,
    )


# --------------------------------------------------------------------------- #
# Weight discovery
# --------------------------------------------------------------------------- #
#: Search patterns, ordered from "most likely to be a trained model" downwards.
_WEIGHT_PATTERNS: tuple[str, ...] = (
    # A checkpoint copied into this repository, which is the arrangement when
    # detection runs locally from a standalone checkout.
    "weights/**/*.pt",
    "weights/**/*.pth",
    "runs/**/weights/best.pt",
    "runs/**/weights/last.pt",
    # A checkout of the shared repository set as the data workspace.
    "Student3-Defect Detection/runs/**/weights/best.pt",
    "Student3-Defect Detection/runs/**/weights/last.pt",
    "Student3-Defect Detection/runs/**/*.pt",
    "Student3-Defect Detection/**/*.pth",
    "Student3-Defect Detection/*.pt",
    "*.pt",
)


def discover_weights(project_root: Path | None) -> list[Path]:
    """
    Find candidate detector checkpoints inside the repository.

    Args:
        project_root: repository root, as returned by
            ``pipeline_bridge.find_project_root()``.

    Returns:
        Unique paths ordered so that purpose-trained weights (``best.pt`` inside a
        training run) come before the stock pre-trained downloads that sit at the
        top of the repository.
    """
    if project_root is None:
        return []
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in _WEIGHT_PATTERNS:
        for path in sorted(project_root.glob(pattern)):
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                seen.add(resolved)
                found.append(path)
    return found


#: Exact filenames Ultralytics ships pre-trained on COCO. Matched by full
#: equality rather than by prefix — a fine-tuned checkpoint such as
#: ``rtdetr_l_pcb.pt`` or ``yolov10n_pcb.pt`` starts with the same prefix as
#: its stock parent but is not one of these, and must not be flagged as stock.
_STOCK_CHECKPOINT_STEMS = frozenset({
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolov9c", "yolov9e",
    "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
    "rtdetr-l", "rtdetr-x",
})


def is_pretrained_stock(path: Path) -> bool:
    """
    True when ``path`` looks like a stock COCO download rather than a checkpoint
    trained on the PCB dataset. Used to warn the operator that the class labels
    will not be defect names.
    """
    stem = path.stem.lower()
    return path.parent.name != "weights" and stem in _STOCK_CHECKPOINT_STEMS


def read_class_names_from_yaml(project_root: Path | None) -> tuple[str, ...] | None:
    """
    Read the class order from Module 3's ``data.yaml`` without requiring PyYAML.

    The file is generated by Module 3 and has a fixed, simple shape::

        names:
          0: missing_hole
          1: mouse_bite

    Returns:
        The class names in index order, or ``None`` if the file is absent or
        does not match the expected shape.
    """
    if project_root is None:
        return None
    yaml_path = project_root / "Student3-Defect Detection" / "dataset" / "yolo" / "data.yaml"
    if not yaml_path.is_file():
        return None
    try:
        lines = yaml_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    names: dict[int, str] = {}
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line.startswith((" ", "\t")) or not stripped:
                break
            key, _, value = stripped.partition(":")
            try:
                names[int(key.strip())] = value.strip().strip("'\"")
            except ValueError:
                break
    if not names:
        return None
    return tuple(names[i] for i in sorted(names))


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
class DefectDetector:
    """
    Loads a Module 3 checkpoint and runs inference on BGR images.

    The constructor never raises. Inspect :attr:`available` and :attr:`load_error`
    to find out whether a model is ready.

    Args:
        weights_path: path to a ``.pt``/``.pth`` checkpoint. ``None`` leaves the
            detector unloaded.
        class_names: class order to apply when the checkpoint carries none.
        device: ``"cpu"``, ``"cuda"`` or ``None`` to let the backend decide.
    """

    def __init__(
        self,
        weights_path: str | Path | None,
        class_names: Iterable[str] | None = None,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.class_names: tuple[str, ...] = tuple(class_names) if class_names else DEFAULT_CLASS_NAMES
        self.device = device
        self.backend: str | None = None
        self.load_error: str | None = None
        self._model: Any = None

        if self.weights_path is None:
            self.load_error = "No detector weights were selected."
        elif not self.weights_path.is_file():
            self.load_error = f"Weights file not found: {self.weights_path}"
        else:
            self._load()

    # -- loading ----------------------------------------------------------- #
    def _load(self) -> None:
        """Choose a backend from the checkpoint and load it."""
        suffix = self.weights_path.suffix.lower()
        if suffix == ".pth" or self._looks_like_torchvision_path():
            self._load_torchvision()
        else:
            self._load_ultralytics()

    def _looks_like_torchvision_path(self) -> bool:
        """
        Recognise a Faster R-CNN checkpoint even when it was saved with a
        ``.pt`` extension. Module 3 saves it as ``faster_rcnn/best.pt`` — same
        extension as the Ultralytics checkpoints, but the wrong loader entirely
        (Ultralytics' ``YOLO(...)`` cannot read a torchvision state dict). The
        containing folder name is the reliable signal, so it is checked
        alongside the ``.pth`` suffix rather than instead of it.
        """
        parts = {p.lower() for p in self.weights_path.parts}
        return "faster_rcnn" in parts or "fasterrcnn" in parts

    def _load_ultralytics(self) -> None:
        """Load a YOLO or RT-DETR checkpoint through the Ultralytics API."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.load_error = (
                "The 'ultralytics' package is not installed. "
                "Run: pip install ultralytics"
            )
            return

        try:
            model = YOLO(str(self.weights_path))
        except Exception:                                    # noqa: BLE001
            # RT-DETR checkpoints need the dedicated class in some versions.
            try:
                from ultralytics import RTDETR

                model = RTDETR(str(self.weights_path))
            except Exception as exc:                         # noqa: BLE001
                self.load_error = f"Could not load the checkpoint: {type(exc).__name__}: {exc}"
                return

        self._model = model
        self.backend = BACKEND_ULTRALYTICS

        # Prefer the names embedded in the checkpoint — they are authoritative.
        embedded = getattr(model, "names", None)
        if isinstance(embedded, dict) and embedded:
            self.class_names = tuple(embedded[k] for k in sorted(embedded))
        elif isinstance(embedded, (list, tuple)) and embedded:
            self.class_names = tuple(embedded)

    def _load_torchvision(self) -> None:
        """
        Load a Faster R-CNN checkpoint saved by Module 3.

        Module 3 replaces torchvision's default anchor sizes (32-512 px) with
        8-128 px because the rescaled defects are only 10-25 px across, so the
        same anchor generator must be rebuilt here before the state dictionary
        will fit the model.
        """
        try:
            import torch
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
            from torchvision.models.detection.anchor_utils import AnchorGenerator
        except ImportError:
            self.load_error = (
                "The 'torch' / 'torchvision' packages are not installed. "
                "Run: pip install torch torchvision"
            )
            return

        try:
            checkpoint = torch.load(str(self.weights_path), map_location="cpu", weights_only=False)
            state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

            if hasattr(state, "eval"):          # a whole pickled model was saved
                model = state
            else:                               # a state dictionary was saved
                # weights_backbone=None: skip fetching ImageNet backbone weights
                # from the network — load_state_dict below overwrites every
                # parameter anyway, and Module 3's own notebook documents this
                # exact download failing on campus/office networks.
                model = fasterrcnn_resnet50_fpn_v2(
                    weights=None,
                    weights_backbone=None,
                    num_classes=len(self.class_names) + 1,   # +1 for background
                )
                # torchvision's v2 builder already supplies its own
                # rpn_anchor_generator internally, so passing one as a
                # constructor kwarg raises "got multiple values for keyword
                # argument 'rpn_anchor_generator'". Module 3's own training
                # notebook works around this the same way: build with the
                # default anchors, then swap the attribute afterwards.
                anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
                aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
                model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
                model.load_state_dict(state)

            model.eval()
            self._model = model
            self._torch = torch
            self.backend = BACKEND_TORCHVISION
        except Exception as exc:                             # noqa: BLE001
            self.load_error = (
                f"Could not load the Faster R-CNN checkpoint: {type(exc).__name__}: {exc}"
            )

    # -- state ------------------------------------------------------------- #
    @property
    def available(self) -> bool:
        """True when a model is loaded and ready for inference."""
        return self._model is not None

    @property
    def model_name(self) -> str:
        return self.weights_path.name if self.weights_path else "none"

    def status(self) -> dict[str, Any]:
        """Summary used by the 'System status' page of the interface."""
        return {
            "available": self.available,
            "weights": str(self.weights_path) if self.weights_path else None,
            "backend": self.backend,
            "classes": list(self.class_names),
            "num_classes": len(self.class_names),
            "error": self.load_error,
        }

    # -- inference --------------------------------------------------------- #
    def predict(
        self,
        image: np.ndarray,
        confidence: float = 0.25,
        iou: float = 0.45,
        max_detections: int = 300,
    ) -> DetectionResult:
        """
        Detect defects in one BGR image.

        Args:
            image: BGR array.
            confidence: minimum score a detection must reach to be reported.
            iou: IoU threshold for non-maximum suppression (Ultralytics only;
                the NMS-free architectures ignore it internally).
            max_detections: hard cap on the number of boxes returned.

        Returns:
            A :class:`DetectionResult`. When no model is loaded, the result is
            empty and carries the loading error, so the interface can show the
            processing stages and simply report that detection was unavailable.
        """
        height, width = image.shape[:2]
        if not self.available:
            return DetectionResult([], 0.0, (height, width), self.model_name, self.load_error)

        started = time.perf_counter()
        try:
            if self.backend == BACKEND_ULTRALYTICS:
                detections = self._predict_ultralytics(image, confidence, iou, max_detections)
            else:
                detections = self._predict_torchvision(image, confidence, max_detections)
        except Exception as exc:                             # noqa: BLE001
            return DetectionResult(
                [], 0.0, (height, width), self.model_name,
                f"Inference failed: {type(exc).__name__}: {exc}",
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionResult(detections, elapsed_ms, (height, width), self.model_name)

    def _predict_ultralytics(
        self, image: np.ndarray, confidence: float, iou: float, max_detections: int
    ) -> list[Detection]:
        kwargs: dict[str, Any] = {
            "conf": confidence,
            "iou": iou,
            "max_det": max_detections,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self._model.predict(image, **kwargs)
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, class_id in zip(xyxy, scores, class_ids):
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        class_name=self._name_for(int(class_id)),
                        confidence=float(score),
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                    )
                )
        return detections

    def _predict_torchvision(
        self, image: np.ndarray, confidence: float, max_detections: int
    ) -> list[Detection]:
        import cv2

        torch = self._torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)

        with torch.no_grad():
            output = self._model([tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        labels = output["labels"].cpu().numpy().astype(int)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            if score < confidence:
                continue
            class_id = int(label) - 1          # torchvision reserves id 0 for background
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self._name_for(class_id),
                    confidence=float(score),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                )
            )
            if len(detections) >= max_detections:
                break
        return detections

    def _name_for(self, class_id: int) -> str:
        """Map a class index to its label, tolerating an out-of-range index."""
        if 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return f"class_{class_id}"
