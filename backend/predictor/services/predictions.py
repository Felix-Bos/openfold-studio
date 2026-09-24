"""Prediction use cases: submit a sequence, follow progress, read and interpret results."""

from django.db.models import Avg
from django.utils import timezone

from predictor import tasks
from predictor.domain import confidence
from predictor.errors import ResourceNotFoundError
from predictor.models import AttentionRun, PredictionJob
from predictor.openfold import results

from .compute import ensure_compute_available

RECENT_JOBS_LIMIT = 20
# Largest PDE matrix sent to the browser (bigger ones are block-averaged).
PDE_DISPLAY_SIZE = 256


def submit_prediction(sequence: str) -> PredictionJob:
    """Creates a job for an already validated sequence and starts it in background."""
    ensure_compute_available()
    job = PredictionJob.objects.create(sequence=sequence)
    job.workspace.create()
    tasks.start_prediction(job.id)
    return job


def recent_jobs(limit: int = RECENT_JOBS_LIMIT):
    return PredictionJob.objects.all()[:limit]


def dashboard_stats() -> dict:
    """Headline numbers for the home page."""
    completed = PredictionJob.objects.filter(status=PredictionJob.Status.COMPLETED)
    return {
        "total": PredictionJob.objects.count(),
        "completed": completed.count(),
        "mean_plddt": completed.aggregate(value=Avg("best_plddt"))["value"],
        "attention_runs": AttentionRun.objects.filter(status=AttentionRun.Status.COMPLETED).count(),
    }


def prediction_progress(job: PredictionJob) -> dict:
    """Progress snapshot served to the polling front-end."""
    elapsed_seconds = job.elapsed_seconds
    if job.is_active and job.started_at:
        elapsed_seconds = (timezone.now() - job.started_at).total_seconds()

    # Inference reports a single-batch bar (0 % until the very end), so it
    # carries no real intermediate progress: show it as indeterminate instead
    # of a bar stuck at 0 %.
    progress_percent = job.progress_percent
    eta_seconds = job.eta_seconds
    if job.status == PredictionJob.Status.RUNNING_INFERENCE and (progress_percent or 0) < 100:
        progress_percent = None
        eta_seconds = None

    return {
        "status": job.status,
        "status_display": job.get_status_display(),
        "current_step": job.current_step,
        "progress_percent": progress_percent,
        "eta_seconds": eta_seconds,
        "elapsed_seconds": elapsed_seconds,
        "error_message": job.error_message,
        "is_active": job.is_active,
    }


def _ensure_known_sample(job: PredictionJob, sample: int) -> None:
    if sample not in {s["sample"] for s in job.result_summary or []}:
        raise ResourceNotFoundError("Sample not found.")


def sample_structure(job: PredictionJob, sample: int) -> str:
    """mmCIF text of one sample of a completed job."""
    _ensure_known_sample(job, sample)
    return results.read_structure(job.workspace, sample)


def best_structure(job: PredictionJob) -> str | None:
    """mmCIF text of the best ranked sample, or None if unavailable."""
    best = job.best_sample
    if best is None:
        return None
    try:
        return results.read_structure(job.workspace, best["sample"])
    except ResourceNotFoundError:
        return None


def sample_confidence(job: PredictionJob, sample: int) -> dict:
    """Everything the job page needs to explain how reliable one sample is."""
    _ensure_known_sample(job, sample)
    residues = results.read_residues(job.workspace, sample)
    details = results.read_confidence_details(job.workspace, sample)
    aggregated = details["aggregated"]
    plddt = [r.plddt for r in residues]

    pde = details["pde"]
    pde_payload = None
    if pde is not None:
        shown = confidence.downsample_matrix(pde, PDE_DISPLAY_SIZE)
        pde_payload = {
            "values": [[round(float(v), 2) for v in row] for row in shown],
            "residues_per_cell": pde.shape[0] / shown.shape[0],
            "max": round(float(pde.max()), 2),
            "mean": round(float(pde.mean()), 2),
        }

    mean_plddt = sum(plddt) / len(plddt) if plddt else aggregated.get("avg_plddt")
    return {
        "sample": sample,
        "residues": [{"index": r.index, "name": r.name, "plddt": round(r.plddt, 2)} for r in residues],
        "bands": [
            {"key": band.key, "label": band.label, "fraction": fraction}
            for band, fraction in zip(confidence.BANDS, confidence.band_fractions(plddt).values(), strict=True)
        ],
        "low_confidence_regions": confidence.low_confidence_regions(plddt),
        "verdict": confidence.verdict(mean_plddt, aggregated.get("ptm")),
        "scores": {
            "avg_plddt": aggregated.get("avg_plddt"),
            "ptm": aggregated.get("ptm"),
            "gpde": aggregated.get("gpde"),
            "disorder": aggregated.get("disorder"),
            "has_clash": bool(aggregated.get("has_clash")),
            "ranking_score": aggregated.get("sample_ranking_score"),
        },
        "pde": pde_payload,
    }
