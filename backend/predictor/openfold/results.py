"""Reads the files OpenFold3-MLX and the attention worker write to disk."""

import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from predictor.domain.structure import Residue, parse_residues
from predictor.errors import ResourceNotFoundError

from .workspace import AttentionWorkspace, JobWorkspace

_SAMPLE_NUMBER_RE = re.compile(r"_sample_(\d+)_")


@dataclass(frozen=True)
class SampleScores:
    """Confidence scores of one diffusion sample.

    avg_plddt: mean per-residue confidence (0-100, higher is better).
    ptm: predicted TM-score, quality of the global fold (0-1).
    gpde: global predicted distance error (lower is better).
    sample_ranking_score: score OpenFold uses to rank the samples.
    disorder: predicted fraction of disordered residues (0-1).
    has_clash: 1.0 when atoms of the structure overlap.
    """

    sample: int
    avg_plddt: float | None
    ptm: float | None
    gpde: float | None
    sample_ranking_score: float | None
    disorder: float | None = None
    has_clash: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def read_sample_scores(workspace: JobWorkspace) -> list[SampleScores]:
    """Returns every sample's scores, best ranked first."""
    samples = []
    for path in workspace.confidence_files():
        match = _SAMPLE_NUMBER_RE.search(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        samples.append(
            SampleScores(
                sample=int(match.group(1)),
                avg_plddt=data.get("avg_plddt"),
                ptm=data.get("ptm"),
                gpde=data.get("gpde"),
                sample_ranking_score=data.get("sample_ranking_score"),
                disorder=data.get("disorder"),
                has_clash=data.get("has_clash"),
            )
        )
    return sorted(samples, key=lambda s: -(s.sample_ranking_score or 0))


def read_structure(workspace: JobWorkspace, sample: int) -> str:
    """mmCIF text of one predicted sample."""
    path = workspace.structure_file(sample)
    if not path.exists():
        raise ResourceNotFoundError("Structure file not found.")
    return path.read_text()


def read_residues(workspace: JobWorkspace, sample: int) -> list[Residue]:
    """Per-residue pLDDT and alpha-carbon position of one sample."""
    path = workspace.structure_file(sample)
    if not path.exists():
        raise ResourceNotFoundError("Structure file not found.")
    return list(_parse_residues_cached(str(path), path.stat().st_mtime))


@lru_cache(maxsize=16)
def _parse_residues_cached(path: str, _mtime: float) -> tuple[Residue, ...]:
    with open(path) as handle:
        return tuple(parse_residues(handle.read()))


def read_confidence_details(workspace: JobWorkspace, sample: int) -> dict:
    """Aggregated scores plus the [N, N] predicted distance error (PDE) matrix.

    PDE[i, j] is the expected error, in ångströms, of the predicted distance
    between residues i and j: low-error blocks reveal rigid domains whose
    relative placement the model is sure about.
    """
    aggregated_path = workspace.scores_file(sample)
    details_path = workspace.confidence_details_file(sample)
    if not aggregated_path.exists():
        raise ResourceNotFoundError("Confidence scores not found.")

    aggregated = json.loads(aggregated_path.read_text())
    pde = None
    if details_path.exists():
        raw = json.loads(details_path.read_text()).get("pde")
        pde = np.asarray(raw, dtype=np.float32) if raw is not None else None
    return {"aggregated": aggregated, "pde": pde}


def read_attention_meta(workspace: AttentionWorkspace) -> dict | None:
    if not workspace.meta_file.exists():
        return None
    return json.loads(workspace.meta_file.read_text())


def load_attention_weights(workspace: AttentionWorkspace, family: str) -> np.ndarray:
    """[n_blocks, N, N] attention weights of one layer family."""
    return _load_npz(workspace.weights_file(family), "Attention data not found.")


def load_attention_cube(workspace: AttentionWorkspace, family: str) -> np.ndarray:
    """[n_blocks, N, N, N] triangle-attention cube of one layer family."""
    return _load_npz(workspace.cube_file(family), "Cube data not found.")


def _load_npz(path, missing_message: str) -> np.ndarray:
    if not path.exists():
        raise ResourceNotFoundError(missing_message)
    with np.load(path) as archive:
        return archive["weights"]
