"""
pcb_check.py
============
A second, independent opinion on whether an uploaded image actually shows a
PCB — added at this repository's own request, on top of (never instead of)
Student 1's ``validate_pcb_image``.

Why a second opinion is needed
-------------------------------
Student 1's check (wired in via ``pipeline_bridge.py`` / ``pipeline_remote.py``
/ ``local_service/serve.py``, and exposed on every ``StageResult`` as
``pcb_valid`` / ``pcb_message``) answers one question: is there a
not-too-small, not-too-elongated blob of saturated pixels somewhere in the
frame? That single signal accepts far more than actual boards. Measured
against this project's own ``sample_images`` versus three synthetic
counter-examples (pure random noise, a mocked-up slide with one coloured
accent shape, a mocked-up skin-tone portrait crop), all three counter-examples
were reported as "PCB image validated successfully" — only a perfectly flat
grey/white image was ever rejected.

The two signals below were chosen because they are exactly what that check
does not look at, and both were verified on the same samples:

    dominant-hue concentration   A real PCB is overwhelmingly one substrate
                                  colour (green / blue / red / black FR4)
                                  with small high-contrast features on top, so
                                  most "colourful" pixels in the board region
                                  cluster within one hue band. Scattered or
                                  multi-subject colour (random noise, a face
                                  against a background, several differently
                                  coloured slide elements) does not.
                                  Measured: 0.97-0.99 on real board samples,
                                  0.17 on random noise.
    edge / texture density        Silkscreen text, pads, vias and copper
                                  traces put real structure inside the board.
                                  A single flat colour block — the case the
                                  hue check alone cannot catch, since a flat
                                  block is 100% one hue too — has almost none.
                                  Measured: 0.06-0.16 on real board samples,
                                  ~0.000-0.003 on the flat synthetic blocks.

This is still a heuristic, not a trained classifier — a sufficiently
PCB-shaped decoy could still fool it — so it is one more gate, combined with
Student 1's own answer via :func:`evaluate`, not a replacement for it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .roi import find_pcb_regions

#: A "colourful" pixel for the hue histogram: saturated and not near-black.
#: Mirrors the threshold ``find_pcb_regions``'s own mask already uses.
_SAT_MIN = 30
_VAL_MIN = 20

#: Minimum share of colourful pixels that must sit within one 30 deg hue
#: window (the peak bin plus its two neighbours, out of 18 bins spanning
#: OpenCV's 0-180 hue range) for the colour to count as "one substrate".
HUE_CONCENTRATION_MIN = 0.55

#: Minimum share of pixels flagged as a Canny edge, i.e. some real texture
#: rather than a flat fill.
EDGE_DENSITY_MIN = 0.015


def _region_metrics(region: np.ndarray) -> tuple[float, float]:
    """Return ``(hue_concentration, edge_density)`` for one BGR crop."""
    if region is None or region.size == 0:
        return 0.0, 0.0

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    colourful = (sat > _SAT_MIN) & (val > _VAL_MIN)

    if not colourful.any():
        hue_concentration = 0.0
    else:
        hues = hsv[:, :, 0][colourful]
        hist, _ = np.histogram(hues, bins=18, range=(0, 180))
        n = len(hist)
        best = max(int(hist[i]) + int(hist[(i - 1) % n]) + int(hist[(i + 1) % n]) for i in range(n))
        hue_concentration = best / float(hues.size)

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    return hue_concentration, edge_density


def looks_like_pcb(img: np.ndarray) -> tuple[bool, str]:
    """
    Independent second opinion for one BGR image.

    Locates the largest candidate board region with :func:`core.roi.find_pcb_regions`
    (Module 4's own, notebook-independent reproduction of Student 1's bounding-box
    step — reused here purely to find *where* to measure, not to decide anything),
    then checks that region's dominant-hue concentration and edge density against
    :data:`HUE_CONCENTRATION_MIN` / :data:`EDGE_DENSITY_MIN`.

    Args:
        img: BGR array to check — the original upload, not a preprocessed one.

    Returns:
        ``(True, message)`` when the largest candidate region clears both
        thresholds, ``(False, message)`` otherwise, including when no
        candidate region exists at all.
    """
    if img is None:
        return False, "Invalid image file."

    boxes = find_pcb_regions(img)
    if not boxes:
        return False, "No PCB-like region was found (secondary check)."

    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    region = img[y:y + h, x:x + w]
    hue_concentration, edge_density = _region_metrics(region)

    if hue_concentration < HUE_CONCENTRATION_MIN:
        return False, (
            "Colour in the candidate region is too scattered to be one PCB "
            f"substrate (hue concentration {hue_concentration:.2f}, need "
            f">= {HUE_CONCENTRATION_MIN})."
        )
    if edge_density < EDGE_DENSITY_MIN:
        return False, (
            "The candidate region has almost no texture — looks like a flat "
            f"colour block rather than a populated board (edge density "
            f"{edge_density:.3f}, need >= {EDGE_DENSITY_MIN})."
        )
    return True, "Secondary PCB check passed (colour concentration + texture)."


def evaluate(
    pcb_valid: bool | None,
    pcb_message: str | None,
    image: np.ndarray,
) -> tuple[bool | None, str | None]:
    """
    Combine Student 1's verdict with this module's own second opinion.

    Args:
        pcb_valid: ``StageResult.pcb_valid`` — Student 1's answer, or ``None``
            when no validator was available to ask.
        pcb_message: ``StageResult.pcb_message`` alongside it.
        image: the original upload (``StageResult.original``) to run the
            second opinion on. Deliberately the ORIGINAL, not the aligned/
            preprocessed image — the two checks should see the same input
            Student 1's own validator saw.

    Returns:
        ``(False, message)`` if either opinion rejects the image — Student
        1's own rejection is passed through unchanged, this module's own
        rejection message otherwise. ``(pcb_valid, pcb_message)`` unchanged
        (``True`` or ``None``) when this module's check also passes, so a
        checkout with no validator at all (``pcb_valid is None``) still
        degrades to "unknown" rather than "rejected" once this check agrees.
    """
    if pcb_valid is False:
        return False, pcb_message

    ok, message = looks_like_pcb(image)
    if not ok:
        return False, message

    return pcb_valid, pcb_message
