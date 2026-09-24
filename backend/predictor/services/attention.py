"""Attention use cases: start an extraction and serve its maps to the explorer.

Full tensors are large (48 blocks x N x N, and N^3 per block for cubes), so
the explorer never downloads a whole family: it asks for a summary first,
then for one block at a time as the user moves the block slider.
"""

from predictor import tasks
from predictor.domain.attention import (
    LAYER_FAMILY_KEYS,
    block_statistics,
    contact_precision,
    family_summary,
    get_layer_family,
    select_block,
    sparse_cube_points,
    top_long_range_links,
)
from predictor.domain.structure import ca_distance_matrix
from predictor.errors import JobNotReadyError, ResourceNotFoundError
from predictor.models import AttentionRun, PredictionJob
from predictor.openfold import results

from .compute import ensure_compute_available


def start_attention_run(job: PredictionJob, families: list[str]) -> AttentionRun:
    """Starts extracting the given layer families (all of them if empty)."""
    if not job.is_completed:
        raise JobNotReadyError("The prediction must be completed before its attention can be analysed.")
    ensure_compute_available()

    run = AttentionRun.objects.create(job=job, layer_families=families or list(LAYER_FAMILY_KEYS))
    run.workspace.root.mkdir(parents=True, exist_ok=True)
    tasks.start_attention_run(run.id)
    return run


def attention_status(run: AttentionRun) -> dict:
    return {
        "status": run.status,
        "status_display": run.get_status_display(),
        "error_message": run.error_message,
        "is_active": run.is_active,
    }


def available_families(run: AttentionRun) -> dict[str, dict]:
    """{family: {shape, summary}} for every requested family whose data file exists."""
    families = {}
    for family in run.layer_families:
        try:
            weights = _weights(run, family)
        except ResourceNotFoundError:
            continue
        families[family] = {"shape": list(weights.shape), "summary": family_summary(weights)}
    return families


def family_shape(run: AttentionRun, family: str) -> list[int]:
    """[n_blocks, N, N] shape of a family, used to size the block slider."""
    return list(_weights(run, family).shape)


def block_weights(run: AttentionRun, family: str, block: int) -> list[list[float]]:
    """N x N attention matrix of one block."""
    return select_block(_weights(run, family), block).astype(float).tolist()


def block_insights(run: AttentionRun, family: str, block: int) -> dict:
    """Matrix of one block plus how to read it: where attention goes, strongest
    long-range links and whether they are real contacts in the predicted structure."""
    matrix = select_block(_weights(run, family), block)
    distances = _ca_distances(run.job)
    links = top_long_range_links(matrix, distances)
    return {
        "weights": matrix.astype(float).tolist(),
        "stats": block_statistics(matrix),
        "links": links,
        "contact_precision": contact_precision(links),
    }


def cube_points(run: AttentionRun, family: str, block: int, threshold: float) -> dict:
    """Sparse [i, j, k, value] points of one triangle-attention cube block."""
    if not get_layer_family(family).has_cube:
        raise ResourceNotFoundError("This layer family has no 3D cube view.")
    cube = select_block(results.load_attention_cube(run.workspace, family), block)
    return {
        "shape": list(cube.shape),
        "threshold": threshold,
        "points": sparse_cube_points(cube, threshold),
    }


def _weights(run: AttentionRun, family: str):
    get_layer_family(family)
    return results.load_attention_weights(run.workspace, family)


def _ca_distances(job: PredictionJob):
    """Alpha-carbon distance matrix of the best sample, or None if unavailable."""
    best = job.best_sample
    if best is None:
        return None
    try:
        return ca_distance_matrix(results.read_residues(job.workspace, best["sample"]))
    except ResourceNotFoundError:
        return None
