"""
viz.py
======
Rendering helpers for the user interface.

All drawing happens on a copy of the input, so the arrays returned by Modules 1
to 3 are never mutated. Colours are keyed to the defect class rather than being
assigned in detection order, so the same defect type always appears in the same
colour across the single-image page, the batch page, the video overlay and the
PDF report.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from .detector import Detection

#: Per-class colours in OpenCV BGR order. Chosen to stay distinguishable against
#: the green solder mask that dominates a PCB image.
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "missing_hole":    (231, 76, 60)[::-1],     # red
    "mouse_bite":      (243, 156, 18)[::-1],    # amber
    "open_circuit":    (155, 89, 182)[::-1],    # purple
    "short":           (52, 152, 219)[::-1],    # blue
    "spur":            (26, 188, 156)[::-1],    # teal
    "spurious_copper": (241, 196, 15)[::-1],    # yellow
}
_FALLBACK_COLOUR = (255, 255, 255)

#: Verdict colours, used for the banner drawn on annotated images.
VERDICT_COLOURS: dict[str, tuple[int, int, int]] = {
    "PASS":   (39, 174, 96)[::-1],
    "REVIEW": (243, 156, 18)[::-1],
    "FAIL":   (192, 57, 43)[::-1],
}


def colour_for(class_name: str) -> tuple[int, int, int]:
    """Return the BGR colour assigned to ``class_name``."""
    return CLASS_COLOURS.get(class_name, _FALLBACK_COLOUR)


def colour_for_hex(class_name: str) -> str:
    """Return the same colour as a ``#rrggbb`` string, for charts and HTML."""
    b, g, r = colour_for(class_name)
    return f"#{r:02x}{g:02x}{b:02x}"


def _scaled(image: np.ndarray, base: float = 900.0) -> float:
    """
    Compute a drawing scale so that annotations stay legible on both a 512 px
    dataset image and a full-resolution capture.
    """
    longest = max(image.shape[:2])
    return max(0.4, min(2.0, longest / base))


def draw_detections(
    image: np.ndarray,
    detections: Iterable[Detection],
    show_labels: bool = True,
    show_confidence: bool = True,
    thickness: int | None = None,
) -> np.ndarray:
    """
    Draw bounding boxes and labels onto a copy of ``image``.

    Args:
        image: BGR array to annotate.
        detections: detections to draw.
        show_labels: draw the defect name above each box.
        show_confidence: append the confidence score to the label.
        thickness: box line width in pixels. ``None`` scales with image size.

    Returns:
        A new BGR array with the annotations drawn on it.
    """
    canvas = image.copy()
    scale = _scaled(canvas)
    box_thickness = thickness if thickness is not None else max(1, int(round(2 * scale)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale
    text_thickness = max(1, int(round(1 * scale)))

    for detection in detections:
        colour = colour_for(detection.class_name)
        x1, y1 = int(round(detection.x1)), int(round(detection.y1))
        x2, y2 = int(round(detection.x2)), int(round(detection.y2))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, box_thickness)

        if not show_labels:
            continue

        label = detection.class_name
        if show_confidence:
            label = f"{label} {detection.confidence:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        pad = max(2, int(round(3 * scale)))

        # Place the label above the box, or inside it when the box touches the top edge.
        label_bottom = y1 - pad if y1 - text_h - baseline - pad > 0 else y2 + text_h + pad
        label_top = label_bottom - text_h - baseline

        cv2.rectangle(
            canvas,
            (x1, max(0, label_top)),
            (min(canvas.shape[1], x1 + text_w + 2 * pad), max(0, label_bottom)),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + pad, max(text_h, label_bottom - baseline // 2)),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return canvas


#: BGR palette for board tags, so several boards in one frame are easy to tell apart.
_BOARD_TAG_COLORS = [
    (41, 128, 185),    # blue
    (211, 84, 0),      # orange
    (39, 174, 96),     # green
    (142, 68, 173),    # purple
    (192, 57, 43),     # red
    (243, 156, 18),    # amber
]


def draw_board_labels(
    image: np.ndarray,
    labelled_boxes: Sequence[tuple[tuple[int, int, int, int], str]],
) -> np.ndarray:
    """
    Draw a coloured tag with its label on to each board rectangle.

    ``labelled_boxes`` is a sequence of ``((x, y, w, h), label)``. The tags use
    the same palette as the per-board results table, so the label on the video
    and the row in the results table always match.

    Returns:
        A new BGR array with the tags drawn on it.
    """
    canvas = image.copy()
    scale = _scaled(canvas)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale
    text_thickness = max(1, int(round(1 * scale)))
    pad = max(2, int(round(3 * scale)))

    for i, ((x, y, w, h), label) in enumerate(labelled_boxes):
        colour = _BOARD_TAG_COLORS[i % len(_BOARD_TAG_COLORS)]
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale,
                                                     text_thickness)
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        cv2.rectangle(
            canvas,
            (x0, y0),
            (min(canvas.shape[1], x0 + text_w + 2 * pad),
             min(canvas.shape[0], y0 + text_h + baseline + 2 * pad)),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            canvas,
            label,
            (x0 + pad, max(text_h, y0 + text_h + pad)),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return canvas


def draw_verdict_banner(
    image: np.ndarray,
    verdict: str,
    subtitle: str = "",
) -> np.ndarray:
    """
    Add a coloured verdict strip along the top of an annotated image.

    Used for the video overlay and the batch thumbnails, where the verdict must
    be readable without an accompanying caption.

    Args:
        image: BGR array to annotate.
        verdict: ``"PASS"``, ``"REVIEW"`` or ``"FAIL"``.
        subtitle: optional short text drawn to the right of the verdict.

    Returns:
        A new BGR array with the banner drawn on it.
    """
    canvas = image.copy()
    scale = _scaled(canvas)
    height = max(22, int(round(30 * scale)))
    colour = VERDICT_COLOURS.get(verdict, _FALLBACK_COLOUR)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], height), colour, cv2.FILLED)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = verdict if not subtitle else f"{verdict}  |  {subtitle}"
    cv2.putText(
        canvas,
        text,
        (int(round(8 * scale)), int(round(height * 0.72))),
        font,
        0.6 * scale,
        (255, 255, 255),
        max(1, int(round(1.4 * scale))),
        cv2.LINE_AA,
    )
    return canvas


def to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an OpenCV BGR array to RGB for display.

    Streamlit's ``st.image`` expects RGB, so every array leaving the backend for
    the screen passes through here. Greyscale arrays are returned untouched.
    """
    if image is None:
        return image
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_jpeg(image: np.ndarray, quality: int = 92) -> bytes | None:
    """Encode a BGR array as JPEG bytes, for downloads and the PDF report."""
    if image is None:
        return None
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if ok else None


def encode_png(image: np.ndarray) -> bytes | None:
    """Encode a BGR array as PNG bytes, for lossless downloads."""
    if image is None:
        return None
    ok, buffer = cv2.imencode(".png", image)
    return buffer.tobytes() if ok else None


def thumbnail(image: np.ndarray, longest_side: int = 320) -> np.ndarray:
    """
    Downscale ``image`` so its longest side is ``longest_side`` pixels.

    Batch runs render dozens of previews at once; sending full-resolution arrays
    to the browser makes the gallery sluggish, so previews are shrunk first.
    Images already smaller than the target are returned unchanged.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= longest_side:
        return image
    scale = longest_side / longest
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def legend_entries(detections: Sequence[Detection]) -> list[tuple[str, str, int]]:
    """
    Build the colour legend for a set of detections.

    Returns:
        ``(class_name, hex_colour, count)`` tuples, ordered by descending count,
        covering only the classes that actually appear.
    """
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
    return [
        (name, colour_for_hex(name), count)
        for name, count in sorted(counts.items(), key=lambda item: -item[1])
    ]
