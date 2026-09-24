"""Long-running work executed outside the HTTP request.

A prediction takes minutes, so the view only creates the database row and
calls `start_*`, which runs the task on a daemon thread and returns at once.
The browser then polls the JSON API to follow progress.

Threads are enough for a single-user local tool; a multi-user deployment
would swap this module for a real task queue (Celery, RQ, ...) without
touching the views or services, which only call `start_prediction` and
`start_attention_run`.
"""

import logging
import threading
import time
from uuid import UUID

from django.db import close_old_connections, connection

from .models import AttentionRun, PredictionJob
from .openfold import commands, process
from .openfold.log_parser import ErrorCollector, parse_line
from .openfold.results import read_attention_meta, read_sample_scores

logger = logging.getLogger(__name__)

# Minimum delay between two progress writes, so fast tqdm ticks don't hammer SQLite.
PROGRESS_SAVE_INTERVAL = 1.0


def start_prediction(job_id: UUID) -> threading.Thread:
    return _run_in_background(_run_prediction, PredictionJob, job_id)


def start_attention_run(run_id: UUID) -> threading.Thread:
    return _run_in_background(_run_attention, AttentionRun, run_id)


def _run_in_background(task, model, object_id: UUID) -> threading.Thread:
    def guarded():
        close_old_connections()
        try:
            task(object_id)
        except Exception as exc:  # never leave a row stuck in a running state
            logger.exception("%s %s crashed", model.__name__, object_id)
            model.objects.get(id=object_id).mark_failed([f"Internal error: {exc}"])
        finally:
            connection.close()

    thread = threading.Thread(target=guarded, daemon=True)
    thread.start()
    return thread


def _run_prediction(job_id: UUID) -> None:
    job = PredictionJob.objects.get(id=job_id)
    workspace = job.workspace
    workspace.create()
    commands.write_query(workspace.query_file, workspace.query_name, job.sequence, workspace.seed)

    proc = process.start(commands.predict_command(workspace.query_file, workspace.root))
    job.mark_started(proc.pid)

    errors = ErrorCollector()
    last_save = 0.0
    with workspace.log_file.open("w") as log:
        for line in process.iter_output_lines(proc):
            log.write(line + "\n")
            log.flush()
            errors.feed(line)

            event = parse_line(line)
            if event is None:
                continue
            job.apply_progress(event)
            if time.monotonic() - last_save >= PROGRESS_SAVE_INTERVAL:
                job.save()
                last_save = time.monotonic()

    return_code = proc.wait()
    if return_code == 0:
        job.mark_completed([s.as_dict() for s in read_sample_scores(workspace)])
    else:
        job.mark_failed(errors.lines, return_code)


def _run_attention(run_id: UUID) -> None:
    run = AttentionRun.objects.select_related("job").get(id=run_id)
    workspace = run.workspace
    workspace.root.mkdir(parents=True, exist_ok=True)

    command = commands.attention_command(
        run.job.workspace.query_file, workspace.root, run.layer_families
    )
    proc = process.start(command)
    run.mark_started(proc.pid)

    errors = ErrorCollector()
    with workspace.log_file.open("w") as log:
        for line in process.iter_output_lines(proc):
            log.write(line + "\n")
            log.flush()
            errors.feed(line)

    return_code = proc.wait()
    if return_code == 0:
        run.mark_completed(read_attention_meta(workspace))
    else:
        run.mark_failed(errors.lines, return_code)
