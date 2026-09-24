"""Builds the command lines and environment used to call OpenFold3-MLX.

Both commands run with the OpenFold interpreter (its own virtualenv), never
with the Django interpreter: the two environments stay independent.
"""

import json
import os
from pathlib import Path

from django.conf import settings


def build_query(query_name: str, sequence: str, seed: int) -> dict:
    """Input document for `run_openfold.py predict`: one protein chain."""
    return {
        "seeds": [seed],
        "queries": {
            query_name: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": sequence,
                    }
                ]
            }
        },
    }


def write_query(path: Path, query_name: str, sequence: str, seed: int) -> None:
    path.write_text(json.dumps(build_query(query_name, sequence, seed), indent=2))


def predict_command(query_file: Path, output_dir: Path) -> list[str]:
    return [
        str(settings.OPENFOLD_PYTHON),
        str(settings.OPENFOLD_RUN_SCRIPT),
        "predict",
        "--query_json", str(query_file),
        "--runner_yaml", str(settings.OPENFOLD_RUNNER_YAML),
        "--output_dir", str(output_dir),
        "--num_diffusion_samples", str(settings.OPENFOLD_NUM_DIFFUSION_SAMPLES),
    ]


def attention_command(query_file: Path, output_dir: Path, families: list[str]) -> list[str]:
    return [
        str(settings.OPENFOLD_PYTHON),
        str(settings.OPENFOLD_WORKER_DIR / "attention_extraction.py"),
        "--query_json", str(query_file),
        "--runner_yaml", str(settings.OPENFOLD_RUNNER_YAML),
        "--output_dir", str(output_dir),
        "--families", ",".join(families),
    ]


def subprocess_env() -> dict[str, str]:
    """Environment for OpenFold subprocesses.

    Puts the openfold-3-mlx checkout on PYTHONPATH so `import openfold3` works
    even if the checkout was moved after its editable install.
    """
    env = os.environ.copy()
    paths = [str(settings.OPENFOLD_PROJECT_DIR)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env
