import re
import uuid
from pathlib import Path

import numpy as np
from django.conf import settings
from django.contrib import messages
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from .attention_runner import start_attention_extraction
from .models import AttentionRun, PredictionJob
from .runner import start_prediction_job

VALID_SEQUENCE_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYXBZJUO\s]+$", re.IGNORECASE)


def architecture(request):
    return render(request, "predictor/architecture.html")


def index(request):
    if request.method == "POST":
        sequence = request.POST.get("sequence", "").strip().upper()
        sequence = re.sub(r"\s+", "", sequence)

        if not sequence:
            messages.error(request, "Merci de saisir une séquence protéique.")
            return redirect("index")

        if not VALID_SEQUENCE_RE.match(sequence):
            messages.error(
                request,
                "Séquence invalide : seuls les codes d'acides aminés à une lettre sont acceptés.",
            )
            return redirect("index")

        active_job = PredictionJob.objects.filter(
            status__in=[
                PredictionJob.Status.PENDING,
                PredictionJob.Status.RUNNING_MSA,
                PredictionJob.Status.RUNNING_INFERENCE,
            ]
        ).first()
        if active_job:
            messages.error(
                request,
                "Une prédiction est déjà en cours. Merci d'attendre qu'elle se termine.",
            )
            return redirect("job_detail", job_id=active_job.id)

        active_attention_run = AttentionRun.objects.filter(
            status__in=[AttentionRun.Status.PENDING, AttentionRun.Status.RUNNING]
        ).first()
        if active_attention_run:
            messages.error(
                request,
                "Une analyse d'attention est en cours — merci d'attendre qu'elle se termine avant de lancer une nouvelle prédiction.",
            )
            return redirect(
                "attention_explorer",
                job_id=active_attention_run.job_id,
                run_id=active_attention_run.id,
            )

        job_id = uuid.uuid4()
        output_dir = settings.OPENFOLD_JOBS_DIR / job_id.hex
        output_dir.mkdir(parents=True, exist_ok=True)

        job = PredictionJob.objects.create(
            id=job_id,
            sequence=sequence,
            output_dir=str(output_dir),
            query_json_path="",
            log_path=str(output_dir / "run.log"),
        )
        start_prediction_job(job)
        return redirect("job_detail", job_id=job.id)

    jobs = PredictionJob.objects.all()[:20]
    return render(request, "predictor/index.html", {"jobs": jobs})


def job_detail(request, job_id):
    job = get_object_or_404(PredictionJob, id=job_id)

    best_cif = None
    if job.result_summary:
        best_cif_path = Path(job.result_summary[0]["cif_path"])
        if best_cif_path.exists():
            best_cif = best_cif_path.read_text()

    return render(
        request,
        "predictor/job_detail.html",
        {
            "job": job,
            "best_cif": best_cif,
            "attention_family_choices": AttentionRun.FAMILY_CHOICES,
        },
    )


def job_sample_cif(request, job_id, sample_num):
    job = get_object_or_404(PredictionJob, id=job_id)

    if not job.result_summary:
        raise Http404("Aucun résultat pour ce job.")

    sample = next(
        (s for s in job.result_summary if s["sample"] == sample_num), None
    )
    if sample is None:
        raise Http404("Échantillon introuvable.")

    cif_path = Path(sample["cif_path"])
    if not cif_path.exists():
        raise Http404("Fichier de structure introuvable.")

    return HttpResponse(cif_path.read_text(), content_type="chemical/x-cif")


ALL_LAYER_FAMILIES = [choice[0] for choice in AttentionRun.FAMILY_CHOICES]
TRIANGLE_LAYER_FAMILIES = {"triangle_start", "triangle_end"}


def attention_start(request, job_id):
    job = get_object_or_404(PredictionJob, id=job_id)

    if request.method != "POST":
        raise Http404()

    if job.status != PredictionJob.Status.COMPLETED:
        messages.error(
            request, "Le job doit être terminé avant d'analyser son attention."
        )
        return redirect("job_detail", job_id=job.id)

    active_run = AttentionRun.objects.filter(
        status__in=[AttentionRun.Status.PENDING, AttentionRun.Status.RUNNING]
    ).first()
    if active_run:
        messages.error(
            request,
            "Une extraction d'attention est déjà en cours. Merci d'attendre qu'elle se termine.",
        )
        return redirect("attention_explorer", job_id=job.id, run_id=active_run.id)

    active_prediction = PredictionJob.objects.filter(
        status__in=[
            PredictionJob.Status.PENDING,
            PredictionJob.Status.RUNNING_MSA,
            PredictionJob.Status.RUNNING_INFERENCE,
        ]
    ).first()
    if active_prediction:
        messages.error(
            request,
            "Une prédiction est en cours — merci d'attendre qu'elle se termine avant de lancer une analyse d'attention.",
        )
        return redirect("job_detail", job_id=job.id)

    families = request.POST.getlist("families") or ALL_LAYER_FAMILIES

    run_id = uuid.uuid4()
    output_dir = settings.OPENFOLD_JOBS_DIR / job.id.hex / "attention" / run_id.hex
    output_dir.mkdir(parents=True, exist_ok=True)

    run = AttentionRun.objects.create(
        id=run_id,
        job=job,
        layer_families=families,
        output_dir=str(output_dir),
        log_path=str(output_dir / "run.log"),
    )
    start_attention_extraction(run)
    return redirect("attention_explorer", job_id=job.id, run_id=run.id)


def attention_status(request, job_id, run_id):
    run = get_object_or_404(AttentionRun, id=run_id, job_id=job_id)
    return JsonResponse(
        {
            "status": run.status,
            "status_display": run.get_status_display(),
            "error_message": run.error_message,
            "is_active": run.is_active,
        }
    )


def attention_explorer(request, job_id, run_id):
    job = get_object_or_404(PredictionJob, id=job_id)
    run = get_object_or_404(AttentionRun, id=run_id, job_id=job_id)

    best_cif = None
    if job.result_summary:
        best_cif_path = Path(job.result_summary[0]["cif_path"])
        if best_cif_path.exists():
            best_cif = best_cif_path.read_text()

    return render(
        request,
        "predictor/attention_explorer.html",
        {
            "job": job,
            "run": run,
            "best_cif": best_cif,
            "attention_family_choices": AttentionRun.FAMILY_CHOICES,
        },
    )


def _load_family_weights(run: AttentionRun, family: str) -> np.ndarray:
    if family not in ALL_LAYER_FAMILIES:
        raise Http404("Famille de couches inconnue.")

    npz_path = Path(run.output_dir) / f"{family}.npz"
    if not npz_path.exists():
        raise Http404("Données d'attention introuvables pour cette famille.")

    return np.load(npz_path)["weights"]  # [n_blocks, N, N]


def attention_data(request, job_id, run_id, family):
    """Returns only the shape for a family — the full tensor (48 blocks x NxN)
    would be tens of MB of JSON. Individual blocks are fetched on demand via
    attention_block_data as the user moves the block slider."""
    run = get_object_or_404(AttentionRun, id=run_id, job_id=job_id)
    weights = _load_family_weights(run, family)

    return JsonResponse({"family": family, "shape": list(weights.shape)})


def attention_block_data(request, job_id, run_id, family, block):
    run = get_object_or_404(AttentionRun, id=run_id, job_id=job_id)
    weights = _load_family_weights(run, family)

    if block < 0 or block >= weights.shape[0]:
        raise Http404("Bloc introuvable pour cette famille.")

    return JsonResponse(
        {
            "family": family,
            "block": block,
            "weights": weights[block].astype(float).tolist(),
        }
    )


def attention_cube_block_data(request, job_id, run_id, family, block):
    """Serves a sparse view of a single block's [I, J, K] triangle-attention
    cube: only points whose normalized weight is above `threshold` are sent.

    A dense cube (e.g. 129^3 ~= 2.1M floats) would be tens of MB of JSON per
    block — sending every point above a threshold instead keeps the payload
    proportional to what's actually going to be drawn (the 3D view only
    renders points above a visibility threshold anyway, see
    attention_explorer.html's drawCube()).
    """
    run = get_object_or_404(AttentionRun, id=run_id, job_id=job_id)

    if family not in TRIANGLE_LAYER_FAMILIES:
        raise Http404("Cette famille n'a pas de vue cube 3D.")

    npz_path = Path(run.output_dir) / f"{family}_cube.npz"
    if not npz_path.exists():
        raise Http404("Données de cube introuvables pour cette famille.")

    weights = np.load(npz_path)["weights"]
    if block < 0 or block >= weights.shape[0]:
        raise Http404("Bloc introuvable pour cette famille.")

    threshold = float(request.GET.get("threshold", 0.6))
    cube = weights[block].astype(np.float32)
    max_val = float(cube.max()) or 1.0
    normalized = cube / max_val

    idx = np.argwhere(normalized >= threshold)
    values = normalized[idx[:, 0], idx[:, 1], idx[:, 2]]

    # Hard cap so a very low threshold on a large cube can't still blow up
    # the payload — keep only the strongest points if there are too many.
    max_points = 60000
    if len(values) > max_points:
        top = np.argsort(values)[-max_points:]
        idx = idx[top]
        values = values[top]

    return JsonResponse(
        {
            "family": family,
            "block": block,
            "shape": list(cube.shape),
            "threshold": threshold,
            "points": [
                [int(i), int(j), int(k), float(v)]
                for (i, j, k), v in zip(idx.tolist(), values.tolist())
            ],
        }
    )


def job_status(request, job_id):
    job = get_object_or_404(PredictionJob, id=job_id)

    elapsed_seconds = job.elapsed_seconds
    if job.is_active and job.started_at:
        elapsed_seconds = (timezone.now() - job.started_at).total_seconds()

    # The inference phase reports progress as a single-batch tqdm bar (0% then
    # 100% at the very end), so it never carries real intermediate progress or
    # ETA. Treat it as indeterminate rather than showing a progress bar stuck
    # at 0%.
    progress_percent = job.progress_percent
    eta_seconds = job.eta_seconds
    if job.status == PredictionJob.Status.RUNNING_INFERENCE and (
        progress_percent is None or progress_percent < 100
    ):
        progress_percent = None
        eta_seconds = None

    return JsonResponse(
        {
            "status": job.status,
            "status_display": job.get_status_display(),
            "current_step": job.current_step,
            "progress_percent": progress_percent,
            "eta_seconds": eta_seconds,
            "elapsed_seconds": elapsed_seconds,
            "error_message": job.error_message,
            "is_active": job.is_active,
        }
    )
