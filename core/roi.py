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


#: How much brighter than its background the board must be before
#: :func:`board_polarity_ok` calls the case inverted. A deadband, not a
#: hair-trigger: measured margins are about -18 to -27 for boards on a light
#: surface (Module 2 works) and +120 for a lit screen on a dark bench (it does
#: not), but a mixed-lighting frame can sit near +18, and with ordinary sensor
#: noise a threshold inside that band flips from frame to frame — which would
#: make this guard itself a source of the twisting it exists to prevent.
POLARITY_MARGIN = 25.0


def board_polarity_ok(
    img: np.ndarray, min_outside_frac: float = 0.05
) -> tuple[bool, str | None]:
    """
    Check the assumption Module 2's corner finder is built on.

    ``image_pipeline.board_corners_threshold`` separates the board from the
    background with ``THRESH_BINARY_INV + OTSU``, i.e. it takes the DARK side
    of the split as the board. That holds for the dataset the system was built
    around — a board photographed on a white surface — but it inverts when the
    board is the bright thing in the frame, such as a lit screen or a
    strongly-lit board on a dark bench. Module 2 then traces the dark
    *background* instead, and either finds no four corners at all or, worse,
    finds four corners of the wrong shape and warps the frame around them. On
    a live stream that lands differently on nearly every frame, so the picture
    appears to twist and jump continuously.

    This is a cheap pre-flight test of that assumption, so the caller can skip
    an alignment whose output cannot be trusted rather than showing the result
    of one.

    The board is located with a brightness floor (unlike :func:`_pcb_mask`,
    which thresholds saturation alone — in HSV a near-black pixel has an
    unstable, often high, saturation, so on a dark background that mask
    swallows the whole frame).

    Args:
        img: BGR frame Module 2 is about to be given.
        min_outside_frac: when less of the frame than this lies outside the
            board, there is no meaningful background to compare against and
            the assumption is left alone — the board-fills-the-frame case,
            which is exactly the dataset Module 2 works correctly on.

    Returns:
        ``(True, None)`` when alignment is worth attempting, otherwise
        ``(False, reason)`` naming the measurement that failed.
    """
    height, width = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    mask = (((sat > SAT_THRESHOLD) & (val > 40)).astype(np.uint8)) * 255
    k = max(3, int(round(min(height, width) * 0.02)) | 1)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True, None                      # nothing located — let Module 2 try

    largest = max(contours, key=cv2.contourArea)
    inside = np.zeros((height, width), np.uint8)
    cv2.drawContours(inside, [largest], -1, 255, -1)
    inside = inside > 0
    outside = ~inside

    if outside.sum() < min_outside_frac * height * width:
        return True, None                      # board fills the frame

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    board_mean = float(gray[inside].mean())
    background_mean = float(gray[outside].mean())
    if board_mean <= background_mean + POLARITY_MARGIN:
        return True, None

    return False, (
        f"the board is much brighter than its surroundings (board "
        f"{board_mean:.0f} vs background {background_mean:.0f}), which inverts "
        "the dark-board assumption Module 2's corner finder relies on"
    )


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
