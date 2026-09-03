"""
roi.py
======
Multi-board region-of-interest detection for one frame.

This is a coarse, axis-aligned locator that answers "how many boards are in
this frame, and roughly where" — a rough pre-filter, not the detector's real
input. It reproduces the ``find_pcb_regions`` / ``crop_pcb`` helpers visible in
Student 1's ``ImagePreprocessing.ipynb`` (HSV-saturation threshold + contour
bounding boxes). Those helpers are loaded internally by
``image_pipeline.load_student1_functions()`` (which executes every
definition-only cell in that notebook so Student 2's wired functions have
their dependencies available) but are not among the four names that function
actually *returns* (``preprocess_image``, ``process_image_from_bytes``,
``process_full_video_backend``, ``validate_pcb_image``) — so they are not
reachable through the shared contract, and are reproduced here instead of
reaching into that notebook's internals or modifying it.

Once a region is located, it still has to go through the REAL Module 1 +
Module 2 (``preprocess_image`` + ``align_image``, via the configured
pipeline bridge) before Module 3 sees it — exactly like a single board on
any other page — because that is the only input format the detector was
actually trained against. This module only answers "where are the boards";
the caller still runs the normal ``inspect()`` routine per board.
"""

from __future__ import annotations

import cv2
import numpy as np

#: Matches Student 1's SAT_THRESHOLD — belt / desk / paper sit near 0-10 in
#: HSV saturation, PCB copper sits far above it.
SAT_THRESHOLD = 30


def _pcb_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    # Fixed threshold, not Otsu — Otsu adapts to the image and will split a
    # board's own interior when the whole frame is already one board.
    mask = (sat > SAT_THRESHOLD).astype(np.uint8) * 255

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=3,
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    return mask


def find_pcb_regions(
    img: np.ndarray, min_area_frac: float = 0.005, max_boards: int = 10
) -> list[tuple[int, int, int, int]]:
    """
    Locate every PCB-like region in ``img``.

    Args:
        img: BGR frame to search.
        min_area_frac: minimum region area, as a fraction of the frame area,
            to be considered a board rather than noise.
        max_boards: hard cap on how many regions are returned.

    Returns:
        ``(x, y, w, h)`` bounding boxes, left to right.
    """
    height, width = img.shape[:2]
    mask = _pcb_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)

        if bw * bh < min_area_frac * width * height:      # too small to be a board
            continue
        if not (0.4 < bw / float(bh) < 2.5):               # boards are roughly square
            continue

        boxes.append((x, y, bw, bh))

    boxes.sort(key=lambda b: b[0])
    return boxes[:max_boards]


def crop_with_padding(
    img: np.ndarray, box: tuple[int, int, int, int], pad: int = 6
) -> np.ndarray:
    """Crop ``img`` to ``box``, expanded by ``pad`` pixels and clipped to bounds."""
    height, width = img.shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + bw + pad), min(height, y + bh + pad)
    return img[y0:y1, x0:x1]


def draw_regions(img: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Draw numbered rectangles around each detected region, for an overview image."""
    canvas = img.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(
            canvas, str(i), (x + 6, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA,
        )
    return canvas
