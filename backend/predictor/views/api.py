"""JSON endpoints polled and fetched by the front-end scripts.

Every endpoint is wrapped by `json_endpoint`, which turns domain errors into
JSON error responses: 404 for missing resources, 400 for other domain errors.
"""

from functools import wraps

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_GET

from predictor.domain.attention import parse_threshold
from predictor.errors import DomainError, ResourceNotFoundError
from predictor.models import AttentionRun, PredictionJob
from predictor.services import attention as attention_service
from predictor.services import predictions as prediction_service


def json_endpoint(view):
    @require_GET
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except ResourceNotFoundError as error:
            return JsonResponse({"error": error.message}, status=404)
        except DomainError as error:
            return JsonResponse({"error": error.message}, status=400)

    return wrapper


def _get_run(job_id, run_id) -> AttentionRun:
    return get_object_or_404(AttentionRun, id=run_id, job_id=job_id)


@json_endpoint
def job_status(request, job_id):
    job = get_object_or_404(PredictionJob, id=job_id)
    return JsonResponse(prediction_service.prediction_progress(job))


@json_endpoint
def sample_structure(request, job_id, sample):
    """mmCIF file of one sample (plain text, loaded by 3Dmol.js)."""
    job = get_object_or_404(PredictionJob, id=job_id)
    cif = prediction_service.sample_structure(job, sample)
    return HttpResponse(cif, content_type="chemical/x-cif")


@json_endpoint
def sample_confidence(request, job_id, sample):
    """Per-residue pLDDT, confidence bands, low-confidence regions and PDE matrix."""
    job = get_object_or_404(PredictionJob, id=job_id)
    return JsonResponse(prediction_service.sample_confidence(job, sample))


@json_endpoint
def attention_status(request, job_id, run_id):
    return JsonResponse(attention_service.attention_status(_get_run(job_id, run_id)))


@json_endpoint
def attention_families(request, job_id, run_id):
    """Shape and summary of every extracted family; its URL is also the base of the family routes."""
    families = attention_service.available_families(_get_run(job_id, run_id))
    return JsonResponse({"families": families})


@json_endpoint
def attention_family(request, job_id, run_id, family):
    shape = attention_service.family_shape(_get_run(job_id, run_id), family)
    return JsonResponse({"family": family, "shape": shape})


@json_endpoint
def attention_block(request, job_id, run_id, family, block):
    """Matrix of one block with its statistics and strongest long-range links."""
    insights = attention_service.block_insights(_get_run(job_id, run_id), family, block)
    return JsonResponse({"family": family, "block": block, **insights})


@json_endpoint
def attention_cube(request, job_id, run_id, family, block):
    threshold = parse_threshold(request.GET.get("threshold"))
    cube = attention_service.cube_points(_get_run(job_id, run_id), family, block, threshold)
    return JsonResponse({"family": family, "block": block, **cube})
