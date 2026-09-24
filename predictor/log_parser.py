"""Parses stdout lines from `openfold3/run_openfold.py predict` into progress events.

The pipeline has two phases that each print their own tqdm-style progress bar:

1. MSA retrieval (ColabFold server), e.g.:
   "SUBMIT:   0%|          | 0/150 [elapsed: 00:00 remaining: ?]"
   "COMPLETE: 100%|##########| 150/150 [elapsed: 00:02 remaining: 00:00]"

2. Model inference (PyTorch Lightning), e.g.:
   "Predicting DataLoader 0: 100%|##########| 1/1 [08:03<00:00,  0.00it/s]"

`parse_line` is a pure function: given one line of text, it returns a dict of
fields to update on the job, or None if the line carries no progress signal.
"""

import re

MSA_BAR_RE = re.compile(
    r"(SUBMIT|COMPLETE):\s+(\d+)%\|.*?\|\s*\d+/\d+\s*"
    r"\[elapsed:\s*([\d:]+)\s+remaining:\s*(\?|[\d:]+)\]"
)

INFERENCE_BAR_RE = re.compile(
    r"Predicting DataLoader 0:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*"
    r"\[([\d:]+)<([\d:]+|\?)"
)

STEP_MSA_SUBMIT = "Récupération des alignements (MSA)"
STEP_TEMPLATES = "Recherche de templates structuraux"
STEP_INFERENCE = "Inférence du modèle (diffusion)"


def _parse_clock(value: str) -> float | None:
    """Parses a "MM:SS" or "HH:MM:SS" clock string into seconds."""
    if value in ("?", ""):
        return None
    parts = value.split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return float(seconds)


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None

    match = MSA_BAR_RE.search(line)
    if match:
        _, percent, elapsed, remaining = match.groups()
        return {
            "current_step": STEP_MSA_SUBMIT,
            "progress_percent": float(percent),
            "elapsed_seconds": _parse_clock(elapsed),
            "eta_seconds": _parse_clock(remaining),
        }

    match = INFERENCE_BAR_RE.search(line)
    if match:
        percent, _done, _total, elapsed, remaining = match.groups()
        return {
            "current_step": STEP_INFERENCE,
            "progress_percent": float(percent),
            "elapsed_seconds": _parse_clock(elapsed),
            "eta_seconds": _parse_clock(remaining),
        }

    if "Submitting" in line and "MSA server" in line:
        return {"current_step": STEP_MSA_SUBMIT, "progress_percent": 0.0}

    if "Preprocessing templates" in line:
        return {"current_step": STEP_TEMPLATES, "progress_percent": None}

    if "PREDICTION SUMMARY (COMPLETE)" in line:
        return {"current_step": "Finalisation", "progress_percent": 100.0}

    return None


def is_error_line(line: str) -> bool:
    return "Traceback (most recent call last)" in line or line.strip().startswith(
        "Exception:"
    )
