"""Guards the single GPU: only one prediction or attention extraction at a time.

Both computations load the full OpenFold3 model in unified memory; running
two at once on a laptop would exhaust it.
"""

from predictor.errors import ComputeBusyError
from predictor.models import AttentionRun, PredictionJob


def ensure_compute_available() -> None:
    """Raises ComputeBusyError if a prediction or an attention run is in progress."""
    if job := PredictionJob.objects.active().first():
        raise ComputeBusyError(
            "A prediction is already running. Please wait for it to finish.",
            blocking=job,
        )
    if run := AttentionRun.objects.active().first():
        raise ComputeBusyError(
            "An attention analysis is already running. Please wait for it to finish.",
            blocking=run,
        )
