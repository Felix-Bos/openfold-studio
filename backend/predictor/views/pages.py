"""HTML pages."""

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from predictor.domain.attention import CUBE_FAMILY_KEYS, LAYER_FAMILIES
from predictor.domain.sequence import EXAMPLE_SEQUENCES
from predictor.errors import ComputeBusyError, DomainError
from predictor.forms import AttentionRunForm, PredictionForm
from predictor.models import AttentionRun, PredictionJob
from predictor.services import attention as attention_service
from predictor.services import predictions as prediction_service


def _redirect_to(obj):
    """Page showing a job or an attention run."""
    if isinstance(obj, AttentionRun):
        return redirect("predictor:attention_explorer", job_id=obj.job_id, run_id=obj.id)
    return redirect("predictor:job_detail", job_id=obj.id)


@require_http_methods(["GET", "POST"])
def index(request):
    """Home page: sequence form (GET) and job submission (POST)."""
    form = PredictionForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        try:
            job = prediction_service.submit_prediction(form.cleaned_data["sequence"])
        except ComputeBusyError as error:
            messages.error(request, error.message)
            return _redirect_to(error.blocking)
        return _redirect_to(job)

    return render(
        request,
        "predictor/index.html",
        {
            "form": form,
            "jobs": prediction_service.recent_jobs(),
            "stats": prediction_service.dashboard_stats(),
            "examples": EXAMPLE_SEQUENCES,
        },
    )


@require_GET
def job_detail(request, job_id):
    job = get_object_or_404(PredictionJob, id=job_id)
    return render(
        request,
        "predictor/job_detail.html",
        {
            "job": job,
            "layer_families": LAYER_FAMILIES,
            "attention_runs": job.attention_runs.all(),
        },
    )


@require_POST
def attention_create(request, job_id):
    job = get_object_or_404(PredictionJob, id=job_id)
    form = AttentionRunForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Invalid layer families.")
        return _redirect_to(job)

    try:
        run = attention_service.start_attention_run(job, form.cleaned_data["families"])
    except ComputeBusyError as error:
        messages.error(request, error.message)
        return _redirect_to(error.blocking)
    except DomainError as error:
        messages.error(request, error.message)
        return _redirect_to(job)
    return _redirect_to(run)


@require_GET
def attention_explorer(request, job_id, run_id):
    run = get_object_or_404(AttentionRun.objects.select_related("job"), id=run_id, job_id=job_id)
    return render(
        request,
        "predictor/attention_explorer.html",
        {
            "job": run.job,
            "run": run,
            "best_cif": prediction_service.best_structure(run.job),
            "layer_families": LAYER_FAMILIES,
            "cube_families": CUBE_FAMILY_KEYS,
        },
    )


@require_GET
def architecture(request):
    """Static explanation of the OpenFold3 architecture."""
    return render(request, "predictor/architecture.html")
