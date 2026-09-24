"""Attention layer families and the numeric transforms served to the explorer.

A "layer family" is a group of attention layers of the same kind inside
OpenFold3. The extraction worker (openfold_worker/attention_extraction.py)
writes one `<key>.npz` file per family, holding a `weights` array of shape
[n_blocks, N, N] (N = number of residues, weights averaged over heads).
Triangle-attention families also get a `<key>_cube.npz` of shape
[n_blocks, N, N, N], which the explorer can draw as a 3D point cloud; their
[N, N] map is the cube averaged over the query index j (averaging over k,
the softmax axis, would give a constant 1/N map).
"""

from dataclasses import dataclass

import numpy as np

from predictor.domain.structure import CONTACT_DISTANCE
from predictor.errors import ResourceNotFoundError


@dataclass(frozen=True)
class LayerFamily:
    key: str
    label: str
    description: str
    has_cube: bool = False


LAYER_FAMILIES = (
    LayerFamily(
        "pairformer_self_attn",
        "Pairformer self-attention",
        "Each residue attends to all others, biased by the pair representation. 48 blocks.",
    ),
    LayerFamily(
        "triangle_start",
        "Triangle attention · starting node",
        "Pair (i, j) attends to pairs (i, k): enforces geometric consistency. 48 blocks.",
        has_cube=True,
    ),
    LayerFamily(
        "triangle_end",
        "Triangle attention · ending node",
        "Pair (i, j) attends to pairs (k, j): the mirror update of the triangle. 48 blocks.",
        has_cube=True,
    ),
    LayerFamily(
        "diffusion_token",
        "Diffusion transformer",
        "Token attention while the 3D coordinates are denoised (mid-rollout step). 24 blocks.",
    ),
)

LAYER_FAMILY_KEYS = tuple(family.key for family in LAYER_FAMILIES)
LAYER_FAMILY_CHOICES = tuple((family.key, family.label) for family in LAYER_FAMILIES)
CUBE_FAMILY_KEYS = tuple(family.key for family in LAYER_FAMILIES if family.has_cube)

# Upper bound on the number of points returned for one cube block, so that a
# very low threshold on a large protein cannot produce a huge JSON payload.
MAX_CUBE_POINTS = 60_000
DEFAULT_CUBE_THRESHOLD = 0.6


def get_layer_family(key: str) -> LayerFamily:
    for family in LAYER_FAMILIES:
        if family.key == key:
            return family
    raise ResourceNotFoundError(f"Unknown layer family: {key}.")


def select_block(weights: np.ndarray, block: int) -> np.ndarray:
    """Returns `weights[block]`, raising ResourceNotFoundError when out of range."""
    if not 0 <= block < weights.shape[0]:
        raise ResourceNotFoundError(f"Block {block} does not exist for this family.")
    return weights[block]


def parse_threshold(raw: str | None, default: float = DEFAULT_CUBE_THRESHOLD) -> float:
    """Parses a query-string threshold and clamps it to [0, 1]."""
    try:
        value = float(raw) if raw is not None else default
    except ValueError:
        value = default
    return min(max(value, 0.0), 1.0)


def sparse_cube_points(
    cube: np.ndarray, threshold: float, max_points: int = MAX_CUBE_POINTS
) -> list[list[float]]:
    """Converts a dense [I, J, K] cube into a sparse list of `[i, j, k, value]`.

    Values are normalised by the cube maximum so `threshold` is relative
    (0.6 = "at least 60 % of the strongest weight"). Only points at or above
    the threshold are kept, and at most `max_points` of the strongest ones.
    A dense 129^3 cube is ~2 million floats; sending only what will be drawn
    keeps the payload proportional to the visible points.
    """
    cube = cube.astype(np.float32)
    max_value = float(cube.max()) or 1.0
    normalized = cube / max_value

    indices = np.argwhere(normalized >= threshold)
    values = normalized[indices[:, 0], indices[:, 1], indices[:, 2]]

    if len(values) > max_points:
        strongest = np.argsort(values)[-max_points:]
        indices, values = indices[strongest], values[strongest]

    return [
        [int(i), int(j), int(k), float(v)]
        for (i, j, k), v in zip(indices.tolist(), values.tolist(), strict=True)
    ]


# --- Interpretation metrics ----------------------------------------------------
# Sequence-separation thresholds used to classify attention, as in docs/RESEARCH.md.
LOCAL_RANGE = 3  # |i - j| <= 3: neighbours along the chain
LONG_RANGE = 24  # |i - j| >= 24: residues far apart in the sequence
SEPARATION_BINS = ((0, 0), (1, 3), (4, 11), (12, 23), (24, None))
TOP_LINKS = 10
# A residue receiving more than this many times its uniform share (1/N) of
# the attention is reported as an "attention sink".
SINK_FACTOR = 5.0


def _separation(n: int) -> np.ndarray:
    index = np.arange(n)
    return np.abs(index[:, None] - index[None, :])


def mean_row_entropy(matrix: np.ndarray) -> float:
    """Mean Shannon entropy (nats) of the rows, each normalised to sum to 1.

    Low = each residue attends to few partners; the maximum, log(N), means
    perfectly uniform attention.
    """
    rows = matrix.astype(np.float64)
    rows = rows / np.clip(rows.sum(axis=-1, keepdims=True), 1e-12, None)
    rows = np.clip(rows, 1e-12, 1.0)
    return float((-(rows * np.log(rows)).sum(axis=-1)).mean())


def block_statistics(matrix: np.ndarray) -> dict:
    """Where the attention of one [N, N] block goes, as fractions of its total weight."""
    matrix = matrix.astype(np.float64)
    total = matrix.sum() or 1.0
    separation = _separation(matrix.shape[0])
    return {
        "entropy": mean_row_entropy(matrix),
        "max_entropy": float(np.log(matrix.shape[0])),
        "diagonal_mass": float(matrix[separation == 0].sum() / total),
        "local_mass": float(matrix[(separation >= 1) & (separation <= LOCAL_RANGE)].sum() / total),
        "long_range_mass": float(matrix[separation >= LONG_RANGE].sum() / total),
        "max_weight": float(matrix.max()),
        "sinks": attention_sinks(matrix),
        "separation_profile": [
            {
                "label": f"{low}" if low == high else (f"{low}+" if high is None else f"{low}–{high}"),
                "mass": float(
                    matrix[(separation >= low) & ((separation <= high) if high is not None else True)].sum()
                    / total
                ),
            }
            for low, high in SEPARATION_BINS
        ],
    }


def attention_sinks(matrix: np.ndarray, limit: int = 3) -> list[dict]:
    """Residues that receive a disproportionate share of all attention.

    Transformers often park attention on a few "sink" tokens whatever the
    query; such columns dominate the map but say little about structure.
    Returns up to `limit` residues (0-based index) receiving more than
    SINK_FACTOR times the uniform share, strongest first.
    """
    matrix = matrix.astype(np.float64)
    received = matrix.sum(axis=0) / (matrix.sum() or 1.0)
    uniform = 1.0 / matrix.shape[0]
    order = np.argsort(received)[::-1][:limit]
    return [
        {"index": int(k), "share": float(received[k]), "fold_over_uniform": float(received[k] / uniform)}
        for k in order
        if received[k] > SINK_FACTOR * uniform
    ]


def top_long_range_links(
    matrix: np.ndarray, distances: np.ndarray | None = None, count: int = TOP_LINKS
) -> list[dict]:
    """Strongest residue pairs at least LONG_RANGE apart in the sequence.

    Pairs are unordered: w(i, j) and w(j, i) are merged by taking the larger.
    With `distances` (alpha-carbon distances in Å), each link also says
    whether the two residues are actually in contact in the predicted 3D
    structure, the key question when interpreting attention.
    """
    n = matrix.shape[0]
    symmetric = np.maximum(matrix, matrix.T).astype(np.float64)
    upper = np.triu(_separation(n) >= LONG_RANGE)
    rows, cols = np.nonzero(upper)
    if len(rows) == 0:
        return []
    weights = symmetric[rows, cols]
    order = np.argsort(weights)[::-1][:count]
    max_weight = float(symmetric.max()) or 1.0

    links = []
    for k in order:
        i, j = int(rows[k]), int(cols[k])
        link = {
            "i": i,
            "j": j,
            "weight": float(weights[k]),
            "normalized": float(weights[k] / max_weight),
            "separation": j - i,
        }
        if distances is not None and max(i, j) < distances.shape[0]:
            distance = float(distances[i, j])
            link["distance"] = distance
            link["is_contact"] = distance < CONTACT_DISTANCE
        links.append(link)
    return links


def contact_precision(links: list[dict]) -> float | None:
    """Fraction of links that are 3D contacts, or None without distances."""
    measured = [link for link in links if "is_contact" in link]
    if not measured:
        return None
    return sum(link["is_contact"] for link in measured) / len(measured)


def family_summary(weights: np.ndarray) -> dict:
    """Aggregates block_statistics over all blocks of a family."""
    stats = [block_statistics(block) for block in weights]
    entropies = [s["entropy"] for s in stats]
    return {
        "blocks": len(stats),
        "mean_entropy": float(np.mean(entropies)),
        "max_entropy": stats[0]["max_entropy"],
        "most_focused_block": int(np.argmin(entropies)),
        "most_diffuse_block": int(np.argmax(entropies)),
        "mean_local_mass": float(np.mean([s["local_mass"] for s in stats])),
        "mean_long_range_mass": float(np.mean([s["long_range_mass"] for s in stats])),
        "entropy_per_block": entropies,
    }
