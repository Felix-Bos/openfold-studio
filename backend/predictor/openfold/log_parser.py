"""Turns lines printed by `run_openfold.py predict` into progress events.

The pipeline has two phases that each print a tqdm-style progress bar:

1. MSA retrieval from the ColabFold server, e.g.
   "SUBMIT:   0%|          | 0/150 [elapsed: 00:00 remaining: ?]"
   "COMPLETE: 100%|##########| 150/150 [elapsed: 00:02 remaining: 00:00]"

2. Model inference (PyTorch Lightning), e.g.
   "Predicting DataLoader 0: 100%|##########| 1/1 [08:03<00:00,  0.00it/s]"

`parse_line` is a pure function: one line in, a ProgressEvent (or None when
the line carries no progress information) out.
"""

import re
from dataclasses import dataclass
from enum import Enum

_MSA_BAR_RE = re.compile(
    r"(SUBMIT|COMPLETE):\s+(\d+)%\|.*?\|\s*\d+/\d+\s*"
    r"\[elapsed:\s*([\d:]+)\s+remaining:\s*(\?|[\d:]+)\]"
)
_INFERENCE_BAR_RE = re.compile(
    r"Predicting DataLoader 0:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*"
    r"\[([\d:]+)<([\d:]+|\?)"
)


class Phase(Enum):
    MSA = "msa"
    INFERENCE = "inference"


STEP_MSA = "Fetching multiple sequence alignment (MSA)"
STEP_TEMPLATES = "Searching structural templates"
STEP_INFERENCE = "Running the model (diffusion)"
STEP_FINALIZING = "Writing results"


@dataclass(frozen=True)
class ProgressEvent:
    """What a log line tells us. `None` fields mean "unknown, keep the previous value"
    except `eta_seconds`, which is always replaced (an unknown ETA must be cleared)."""

    step: str
    phase: Phase | None = None
    progress_percent: float | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None


def parse_clock(value: str) -> float | None:
    """Parses "MM:SS" or "HH:MM:SS" into seconds; "?" or garbage gives None."""
    try:
        parts = [int(part) for part in value.split(":")]
    except ValueError:
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return float(seconds)


def parse_line(line: str) -> ProgressEvent | None:
    line = line.strip()
    if not line:
        return None

    if match := _MSA_BAR_RE.search(line):
        _, percent, elapsed, remaining = match.groups()
        return ProgressEvent(
            step=STEP_MSA,
            phase=Phase.MSA,
            progress_percent=float(percent),
            elapsed_seconds=parse_clock(elapsed),
            eta_seconds=parse_clock(remaining),
        )

    if match := _INFERENCE_BAR_RE.search(line):
        percent, _done, _total, elapsed, remaining = match.groups()
        return ProgressEvent(
            step=STEP_INFERENCE,
            phase=Phase.INFERENCE,
            progress_percent=float(percent),
            elapsed_seconds=parse_clock(elapsed),
            eta_seconds=parse_clock(remaining),
        )

    if "Submitting" in line and "MSA server" in line:
        return ProgressEvent(step=STEP_MSA, phase=Phase.MSA, progress_percent=0.0)

    if "Preprocessing templates" in line:
        return ProgressEvent(step=STEP_TEMPLATES, phase=Phase.MSA)

    if "PREDICTION SUMMARY (COMPLETE)" in line:
        return ProgressEvent(step=STEP_FINALIZING, progress_percent=100.0)

    return None


def is_error_line(line: str) -> bool:
    """True for the first line of a Python error report."""
    stripped = line.strip()
    return "Traceback (most recent call last)" in line or stripped.startswith("Exception:")


class ErrorCollector:
    """Keeps every line from the first error report on.

    A traceback's useful part (the stack and the final "ValueError: ...")
    comes *after* its "Traceback" header, so all following lines are kept,
    not only the header.
    """

    def __init__(self):
        self.lines: list[str] = []

    def feed(self, line: str) -> None:
        if self.lines or is_error_line(line):
            self.lines.append(line)
