"""Interpretation of OpenFold confidence scores.

pLDDT bands follow the AlphaFold Protein Structure Database convention:
  > 90   very high: backbone and side chains are expected to be accurate
  70-90  confident: backbone is expected to be accurate
  50-70  low: treat with caution
  < 50   very low: often a disordered region
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceBand:
    key: str
    label: str
    lower: float  # inclusive lower bound of the band


BANDS = (
    ConfidenceBand("very_high", "Very high (> 90)", 90.0),
    ConfidenceBand("confident", "Confident (70–90)", 70.0),
    ConfidenceBand("low", "Low (50–70)", 50.0),
    ConfidenceBand("very_low", "Very low (< 50)", 0.0),
)

# Minimum length of a stretch of low-confidence residues reported as a region.
MIN_REGION_LENGTH = 3
LOW_CONFIDENCE_THRESHOLD = 70.0


def band_for(plddt: float) -> ConfidenceBand:
    for band in BANDS:
        if plddt >= band.lower:
            return band
    return BANDS[-1]


def band_fractions(plddt: list[float]) -> dict[str, float]:
    """Fraction of residues in each confidence band (sums to 1)."""
    if not plddt:
        return {band.key: 0.0 for band in BANDS}
    counts = {band.key: 0 for band in BANDS}
    for value in plddt:
        counts[band_for(value).key] += 1
    return {key: count / len(plddt) for key, count in counts.items()}


def low_confidence_regions(plddt: list[float]) -> list[dict]:
    """Contiguous stretches (1-based, inclusive) with pLDDT below 70."""
    regions = []
    start = None
    for position, value in enumerate([*plddt, 100.0], start=1):  # sentinel closes the last run
        if value < LOW_CONFIDENCE_THRESHOLD and start is None:
            start = position
        elif value >= LOW_CONFIDENCE_THRESHOLD and start is not None:
            end = position - 1
            if end - start + 1 >= MIN_REGION_LENGTH:
                segment = plddt[start - 1 : end]
                regions.append({"start": start, "end": end, "mean_plddt": sum(segment) / len(segment)})
            start = None
    return regions


def verdict(mean_plddt: float | None, ptm: float | None) -> str:
    """One-sentence reading of the global confidence."""
    if mean_plddt is None:
        return "No confidence scores available."
    band = band_for(mean_plddt)
    fold = ""
    if ptm is not None:
        fold = (
            " The global fold is likely correct (pTM > 0.8)."
            if ptm > 0.8
            else " The global fold is plausible (pTM > 0.5)."
            if ptm > 0.5
            else " The global fold is uncertain (pTM ≤ 0.5)."
        )
    local = {
        "very_high": "Very high local confidence: backbone and side chains should be accurate.",
        "confident": "Confident prediction: the backbone should be accurate.",
        "low": "Low confidence: interpret the structure with caution.",
        "very_low": "Very low confidence: the protein may be largely disordered.",
    }[band.key]
    return local + fold


def downsample_matrix(matrix: np.ndarray, max_size: int = 256) -> np.ndarray:
    """Block-averages a square matrix to at most `max_size` rows for display."""
    n = matrix.shape[0]
    if n <= max_size:
        return matrix
    factor = int(np.ceil(n / max_size))
    padded = np.full((factor * int(np.ceil(n / factor)),) * 2, np.nan, dtype=np.float32)
    padded[:n, :n] = matrix
    size = padded.shape[0] // factor
    return np.nanmean(padded.reshape(size, factor, size, factor), axis=(1, 3))
