"""Launches the OpenFold3-MLX predict subprocess and tracks its progress.

Runs in a background thread started by the view that creates a PredictionJob.
Reads the subprocess's combined stdout/stderr character-by-character (tqdm
progress bars rewrite the line in place using '\\r', not '\\n', so a plain
readline() would only ever see the final state of each bar) and updates the
job row in the database as progress events are recognized.

The pipe is read from a dedicated daemon thread into a queue rather than
directly in the tracking loop: PyTorch Lightning's dataloader workers
(num_workers > 0) fork subprocesses that inherit the parent's stdout file
descriptor, so the pipe can stay open (never returning EOF) even after the
main `run_openfold.py` process has exited and been reaped. Watching
`process.poll()` alongside the queue lets the tracking loop notice the real
process has finished instead of hanging forever waiting for EOF.
"""

import json
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .log_parser import is_error_line, parse_line
from .models import PredictionJob

DB_UPDATE_MIN_INTERVAL = 1.0  # seconds, avoid hammering sqlite on fast tqdm ticks
DRAIN_TIMEOUT_AFTER_EXIT = 2.0  # seconds to keep draining the pipe after the process exits


def _pump_stream_to_queue(pipe, line_queue):
    """Reads a subprocess pipe and pushes logical lines (split on \\r or \\n)."""
    buf = ""
    while True:
        chunk = pipe.read(1)
        if chunk == "":
            break
        if chunk in ("\n", "\r"):
            if buf:
                line_queue.put(buf)
                buf = ""
        else:
            buf += chunk
    if buf:
        line_queue.put(buf)
    line_queue.put(None)  # sentinel: reader thread is done


def _iter_process_lines(process):
    """Yields output lines from `process`, guaranteed to stop once it exits.

    Reads happen on a background thread so this generator can also watch
    `process.poll()` and give up on the pipe if it never reaches EOF after
    the process has actually terminated (see module docstring).
    """
    line_queue = queue.Queue()
    reader = threading.Thread(
        target=_pump_stream_to_queue, args=(process.stdout, line_queue), daemon=True
    )
    reader.start()

    process_exited_at = None
    while True:
        try:
            line = line_queue.get(timeout=0.5)
        except queue.Empty:
            if process.poll() is not None:
                if process_exited_at is None:
                    process_exited_at = time.monotonic()
                elif time.monotonic() - process_exited_at > DRAIN_TIMEOUT_AFTER_EXIT:
                    return
            continue

        if line is None:
            return
        yield line


def openfold_subprocess_env():
    """Environment for subprocesses using the OpenFold3-MLX interpreter.

    Puts the openfold-3-mlx checkout on PYTHONPATH so `import openfold3` works
    even if the checkout was moved after its editable install.
    """
    env = os.environ.copy()
    extra = [str(settings.OPENFOLD_PROJECT_DIR)]
    if env.get("PYTHONPATH"):
        extra.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(extra)
    return env


def _build_query_json(job: PredictionJob) -> Path:
    query_name = f"query_{job.id.hex}"
    query = {
        "seeds": [42],
        "queries": {
            query_name: {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A"],
                        "sequence": job.sequence,
                    }
                ]
            }
        },
    }
    query_path = Path(job.output_dir) / "query.json"
    query_path.write_text(json.dumps(query, indent=2))
    return query_path


def _collect_results(job: PredictionJob):
    query_name = f"query_{job.id.hex}"
    seed_dir = Path(job.output_dir) / query_name / "seed_42"
    if not seed_dir.is_dir():
        return

    samples = []
    for conf_file in sorted(seed_dir.glob("*_confidences_aggregated.json")):
        match = re.search(r"_sample_(\d+)_", conf_file.name)
        if not match:
            continue
        sample_num = int(match.group(1))
        data = json.loads(conf_file.read_text())
        samples.append(
            {
                "sample": sample_num,
                "avg_plddt": data.get("avg_plddt"),
                "ptm": data.get("ptm"),
                "gpde": data.get("gpde"),
                "sample_ranking_score": data.get("sample_ranking_score"),
                "cif_path": str(
                    seed_dir / conf_file.name.replace(
                        "_confidences_aggregated.json", "_model.cif"
                    )
                ),
            }
        )

    samples.sort(key=lambda s: -(s["sample_ranking_score"] or 0))
    job.result_summary = samples
    job.best_plddt = samples[0]["avg_plddt"] if samples else None


def run_prediction_job(job_id):
    job = PredictionJob.objects.get(id=job_id)

    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_path = _build_query_json(job)
    job.query_json_path = str(query_path)
    job.status = PredictionJob.Status.RUNNING_MSA
    job.started_at = timezone.now()
    job.current_step = "Démarrage"
    job.save()

    cmd = [
        str(settings.OPENFOLD_PYTHON),
        str(settings.OPENFOLD_RUN_SCRIPT),
        "predict",
        "--query_json",
        str(query_path),
        "--runner_yaml",
        str(settings.OPENFOLD_RUNNER_YAML),
        "--output_dir",
        str(output_dir),
        "--num_diffusion_samples",
        "8",
    ]

    log_path = Path(job.log_path)
    log_file = log_path.open("w")

    process = subprocess.Popen(
        cmd,
        cwd=str(settings.OPENFOLD_PROJECT_DIR),
        env=openfold_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    job.pid = process.pid
    job.save(update_fields=["pid"])

    last_db_write = 0.0
    saw_inference = False
    error_lines = []

    try:
        for line in _iter_process_lines(process):
            log_file.write(line + "\n")
            log_file.flush()

            if is_error_line(line):
                error_lines.append(line)

            event = parse_line(line)
            if event is None:
                continue

            if event["current_step"] == "Inférence du modèle (diffusion)":
                saw_inference = True

            now = time.monotonic()
            if now - last_db_write < DB_UPDATE_MIN_INTERVAL:
                continue
            last_db_write = now

            job.current_step = event["current_step"]
            if event.get("progress_percent") is not None:
                job.progress_percent = event["progress_percent"]
            job.eta_seconds = event.get("eta_seconds")
            if event.get("elapsed_seconds") is not None:
                job.elapsed_seconds = event["elapsed_seconds"]

            new_status = (
                PredictionJob.Status.RUNNING_INFERENCE
                if saw_inference
                else PredictionJob.Status.RUNNING_MSA
            )
            if job.status != new_status:
                job.status = new_status
            job.save()

        return_code = process.wait()
    finally:
        log_file.close()

    job.finished_at = timezone.now()

    if return_code == 0:
        job.status = PredictionJob.Status.COMPLETED
        job.progress_percent = 100.0
        job.eta_seconds = 0.0
        job.current_step = "Terminé"
        _collect_results(job)
    else:
        job.status = PredictionJob.Status.FAILED
        job.error_message = "\n".join(error_lines[-40:]) or (
            f"Le process s'est terminé avec le code {return_code}."
        )

    job.save()


def start_prediction_job(job: PredictionJob):
    thread = threading.Thread(
        target=run_prediction_job, args=(job.id,), daemon=True
    )
    thread.start()
    return thread
