"""
analysis.py
===========
Turns the raw output of Module 3 into the inspection information an operator
actually needs: per-class counts, a severity-weighted quality score, and a
pass / review / fail verdict against configurable acceptance criteria.

Keeping this logic out of ``app.py`` means the verdict rules are testable
without a running Streamlit server, and the same rules serve the single-image
page, the batch page, the video page and the PDF report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pandas as pd

from .detector import Detection, DetectionResult

# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
VERDICT_PASS = "PASS"
VERDICT_REVIEW = "REVIEW"
VERDICT_FAIL = "FAIL"

#: Relative severity of each defect class, used to weight the quality score.
#: Open circuits and shorts break the board electrically, so they carry the most
#: weight; spurs and spurious copper are cosmetic or marginal by comparison.
DEFAULT_SEVERITY: dict[str, float] = {
    "missing_hole": 0.8,
    "mouse_bite": 0.6,
    "open_circuit": 1.0,
    "short": 1.0,
    "spur": 0.4,
    "spurious_copper": 0.5,
}

#: Defect classes that fail a board outright, whatever the allowance.
DEFAULT_CRITICAL_CLASSES: tuple[str, ...] = ("open_circuit", "short")


@dataclass
class InspectionCriteria:
    """
    Acceptance rules applied to one board.

    Attributes:
        max_defects: how many non-critical defects a board may carry and still
            avoid an outright failure. ``0`` means any defect fails the board.
        critical_classes: defect types that fail the board on sight.
        review_confidence: detections scoring below this value are treated as
            uncertain. A board whose only findings are uncertain is sent for
            manual review rather than being failed automatically.
        severity: per-class severity weights for the quality score.
    """

    max_defects: int = 0
    critical_classes: tuple[str, ...] = DEFAULT_CRITICAL_CLASSES
    review_confidence: float = 0.50
    severity: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SEVERITY))

    def severity_of(self, class_name: str) -> float:
        """Severity weight for ``class_name``; unknown classes weigh 0.5."""
        return float(self.severity.get(class_name, 0.5))


@dataclass
class InspectionSummary:
    """Everything the interface displays about one inspected board."""

    verdict: str
    quality_score: float                      # 0-100, higher is better
    total_defects: int
    confident_defects: int
    uncertain_defects: int
    critical_defects: int
    class_counts: dict[str, int]
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    reasons: list[str]
    inference_ms: float = 0.0
    image_shape: tuple[int, int] = (0, 0)

    @property
    def passed(self) -> bool:
        return self.verdict == VERDICT_PASS

    def as_row(self, name: str = "") -> dict[str, Any]:
        """Flat representation for the batch table and CSV export."""
        row: dict[str, Any] = {}
        if name:
            row["image"] = name
        row.update(
            {
                "verdict": self.verdict,
                "quality_score": round(self.quality_score, 1),
                "total_defects": self.total_defects,
                "critical_defects": self.critical_defects,
                "uncertain_defects": self.uncertain_defects,
                "mean_confidence": round(self.mean_confidence, 3) if self.total_defects else 0.0,
                "inference_ms": round(self.inference_ms, 1),
            }
        )
        for class_name in sorted(DEFAULT_SEVERITY):
            row[class_name] = self.class_counts.get(class_name, 0)
        return row


# --------------------------------------------------------------------------- #
# Single-board analysis
# --------------------------------------------------------------------------- #
def summarise(
    result: DetectionResult,
    criteria: InspectionCriteria | None = None,
) -> InspectionSummary:
    """
    Apply the acceptance criteria to one detection result.

    The verdict follows three rules, checked in order:

    1. Any detection belonging to a critical class fails the board.
    2. More confident detections than ``max_defects`` fails the board.
    3. Findings that exist but are all below ``review_confidence``, or that stay
       within the allowance, are flagged for manual review rather than failed.

    Args:
        result: output of :meth:`core.detector.DefectDetector.predict`.
        criteria: acceptance rules. Defaults to a zero-tolerance policy.

    Returns:
        An :class:`InspectionSummary`.
    """
    criteria = criteria or InspectionCriteria()
    detections = result.detections

    class_counts: dict[str, int] = {}
    for detection in detections:
        class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1

    confidences = [d.confidence for d in detections]
    confident = [d for d in detections if d.confidence >= criteria.review_confidence]
    uncertain = [d for d in detections if d.confidence < criteria.review_confidence]
    critical = [d for d in detections if d.class_name in criteria.critical_classes]

    reasons: list[str] = []
    if critical:
        verdict = VERDICT_FAIL
        names = sorted({d.class_name for d in critical})
        reasons.append(
            f"{len(critical)} critical defect(s) detected ({', '.join(names)}) — "
            "these break board continuity."
        )
    elif len(confident) > criteria.max_defects:
        verdict = VERDICT_FAIL
        reasons.append(
            f"{len(confident)} confident defect(s) detected, above the allowance of "
            f"{criteria.max_defects}."
        )
    elif detections:
        verdict = VERDICT_REVIEW
        if uncertain and not confident:
            reasons.append(
                f"{len(uncertain)} low-confidence finding(s) below "
                f"{criteria.review_confidence:.2f} — manual review is advised."
            )
        else:
            reasons.append(
                f"{len(confident)} defect(s) detected, within the allowance of "
                f"{criteria.max_defects}."
            )
    else:
        verdict = VERDICT_PASS
        reasons.append("No defects were detected above the confidence threshold.")

    if result.error:
        reasons.append(f"Detector note: {result.error}")

    return InspectionSummary(
        verdict=verdict,
        quality_score=quality_score(detections, criteria),
        total_defects=len(detections),
        confident_defects=len(confident),
        uncertain_defects=len(uncertain),
        critical_defects=len(critical),
        class_counts=class_counts,
        mean_confidence=float(sum(confidences) / len(confidences)) if confidences else 0.0,
        min_confidence=float(min(confidences)) if confidences else 0.0,
        max_confidence=float(max(confidences)) if confidences else 0.0,
        reasons=reasons,
        inference_ms=result.inference_ms,
        image_shape=result.image_shape,
    )


def rejected(reason: str, image_shape: tuple[int, int] = (0, 0)) -> InspectionSummary:
    """
    Build the summary for an upload that never reached the detector because
    Student 1's ``validate_pcb_image`` (wired in via
    ``core.pipeline_bridge``/``core.pipeline_remote``) determined it does not
    contain a PCB.

    Reported as an outright :data:`VERDICT_FAIL` rather than a fourth verdict,
    so every existing verdict count, filter, chart and CSV/PDF export — which
    only know PASS / REVIEW / FAIL — handles a rejected upload correctly
    without any changes of their own.

    Args:
        reason: the validator's own message (e.g. "Uploaded image does not
            appear to contain a PCB."), shown to the operator as-is.
        image_shape: the rejected image's ``(height, width)``, for the record.

    Returns:
        An :class:`InspectionSummary` with zeroed detection statistics.
    """
    return InspectionSummary(
        verdict=VERDICT_FAIL,
        quality_score=0.0,
        total_defects=0,
        confident_defects=0,
        uncertain_defects=0,
        critical_defects=0,
        class_counts={},
        mean_confidence=0.0,
        min_confidence=0.0,
        max_confidence=0.0,
        reasons=[f"Rejected before detection: {reason}"],
        inference_ms=0.0,
        image_shape=image_shape,
    )


def quality_score(
    detections: Sequence[Detection],
    criteria: InspectionCriteria | None = None,
) -> float:
    """
    Score a board from 100 (flawless) down to 0.

    Each detection subtracts ``25 x severity x confidence``, so a single
    high-confidence open circuit costs 25 points while a marginal spur costs
    around 5. The score is a presentation aid that ranks boards by how badly
    they are affected; the verdict, not the score, decides acceptance.

    Args:
        detections: detections found on the board.
        criteria: supplies the severity weights.

    Returns:
        A score in the closed interval [0, 100].
    """
    criteria = criteria or InspectionCriteria()
    if not detections:
        return 100.0
    penalty = sum(
        25.0 * criteria.severity_of(d.class_name) * float(d.confidence) for d in detections
    )
    return max(0.0, min(100.0, 100.0 - penalty))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def detections_dataframe(detections: Iterable[Detection]) -> pd.DataFrame:
    """
    Build the per-defect table shown under a single inspected board.

    Returns:
        A ``DataFrame`` with one row per detection, or an empty frame with the
        correct columns when nothing was found.
    """
    rows = [d.as_row() for d in detections]
    if not rows:
        return pd.DataFrame(
            columns=[
                "class_id", "defect_type", "confidence", "x1", "y1", "x2", "y2",
                "width_px", "height_px", "area_px2", "centre_x", "centre_y",
            ]
        )
    frame = pd.DataFrame(rows)
    frame.insert(0, "#", range(1, len(frame) + 1))
    return frame


def class_distribution(summaries: Iterable[InspectionSummary]) -> pd.DataFrame:
    """
    Aggregate defect counts across any number of boards.

    Returns:
        A ``DataFrame`` indexed by defect type with ``count`` and ``share_pct``
        columns, sorted by descending count.
    """
    totals: dict[str, int] = {}
    for summary in summaries:
        for class_name, count in summary.class_counts.items():
            totals[class_name] = totals.get(class_name, 0) + count
    if not totals:
        return pd.DataFrame(columns=["defect_type", "count", "share_pct"])

    grand_total = sum(totals.values())
    frame = pd.DataFrame(
        [
            {
                "defect_type": name,
                "count": count,
                "share_pct": round(100.0 * count / grand_total, 1),
            }
            for name, count in totals.items()
        ]
    )
    return frame.sort_values("count", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Batch analysis
# --------------------------------------------------------------------------- #
@dataclass
class BatchSummary:
    """Aggregate statistics over a batch of inspected boards."""

    total_boards: int
    passed: int
    review: int
    failed: int
    total_defects: int
    mean_quality: float
    mean_inference_ms: float
    class_counts: dict[str, int]

    @property
    def yield_pct(self) -> float:
        """Percentage of boards that passed outright — the production yield."""
        if self.total_boards == 0:
            return 0.0
        return 100.0 * self.passed / self.total_boards

    @property
    def defect_rate(self) -> float:
        """Average number of defects per board."""
        if self.total_boards == 0:
            return 0.0
        return self.total_defects / self.total_boards


def summarise_batch(summaries: Sequence[InspectionSummary]) -> BatchSummary:
    """
    Roll a list of per-board summaries into one batch summary.

    Args:
        summaries: one entry per inspected board.

    Returns:
        A :class:`BatchSummary`. An empty input yields an all-zero summary
        rather than raising, so the interface can render before any board runs.
    """
    if not summaries:
        return BatchSummary(0, 0, 0, 0, 0, 0.0, 0.0, {})

    class_counts: dict[str, int] = {}
    for summary in summaries:
        for class_name, count in summary.class_counts.items():
            class_counts[class_name] = class_counts.get(class_name, 0) + count

    return BatchSummary(
        total_boards=len(summaries),
        passed=sum(1 for s in summaries if s.verdict == VERDICT_PASS),
        review=sum(1 for s in summaries if s.verdict == VERDICT_REVIEW),
        failed=sum(1 for s in summaries if s.verdict == VERDICT_FAIL),
        total_defects=sum(s.total_defects for s in summaries),
        mean_quality=sum(s.quality_score for s in summaries) / len(summaries),
        mean_inference_ms=sum(s.inference_ms for s in summaries) / len(summaries),
        class_counts=class_counts,
    )
