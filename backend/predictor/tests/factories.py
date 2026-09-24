"""Helpers that build fake OpenFold outputs in a temporary jobs directory."""

import json

import numpy as np

from predictor.models import AttentionRun, PredictionJob

SEQUENCE = "MKTAYIAKQR"

# Minimal mmCIF with four alpha carbons 5 Å apart along x, pLDDT in B-factor.
CIF = """data_test
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_seq_id
_atom_site.B_iso_or_equiv
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM N MET 1 80.0 0.0 1.0 0.0
ATOM CA MET 1 91.0 0.0 0.0 0.0
ATOM CA LYS 2 85.0 5.0 0.0 0.0
ATOM CA THR 3 60.0 10.0 0.0 0.0
ATOM CA ALA 4 40.0 15.0 0.0 0.0
#
"""


def completed_job(**fields) -> PredictionJob:
    """A completed job with two scored samples written on disk."""
    job = PredictionJob.objects.create(sequence=SEQUENCE, status=PredictionJob.Status.COMPLETED, **fields)
    workspace = job.workspace
    workspace.seed_dir.mkdir(parents=True)
    for sample, score in ((1, 0.2), (2, 0.5)):
        prefix = f"{workspace.query_name}_seed_{workspace.seed}_sample_{sample}"
        (workspace.seed_dir / f"{prefix}_confidences_aggregated.json").write_text(
            json.dumps({"avg_plddt": 80 + sample, "ptm": 0.8, "gpde": 0.4, "sample_ranking_score": score})
        )
        (workspace.seed_dir / f"{prefix}_model.cif").write_text(CIF.replace("data_test", f"data_sample_{sample}"))
        (workspace.seed_dir / f"{prefix}_confidences.json").write_text(
            json.dumps({"plddt": [90.0] * 5, "pde": [[0.5] * 4 for _ in range(4)]})
        )
    job.result_summary = [
        {"sample": 2, "avg_plddt": 82, "ptm": 0.8, "gpde": 0.4, "sample_ranking_score": 0.5},
        {"sample": 1, "avg_plddt": 81, "ptm": 0.8, "gpde": 0.4, "sample_ranking_score": 0.2},
    ]
    job.save()
    return job


def completed_attention_run(job: PredictionJob, n_blocks: int = 3, n: int = 4) -> AttentionRun:
    """A completed run with a self-attention family and a triangle family with cube."""
    run = AttentionRun.objects.create(
        job=job,
        status=AttentionRun.Status.COMPLETED,
        layer_families=["pairformer_self_attn", "triangle_start"],
        meta={"n_residues": n, "families": {}},
    )
    workspace = run.workspace
    workspace.root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for family in run.layer_families:
        np.savez(workspace.weights_file(family), weights=rng.random((n_blocks, n, n)))
    np.savez(workspace.cube_file("triangle_start"), weights=rng.random((n_blocks, n, n, n)))
    return run
