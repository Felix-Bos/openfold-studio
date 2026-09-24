"""Launches the OpenFold3 attention-extraction subprocess and tracks it.

Mirrors runner.py's approach (background thread, robust pipe draining via
_iter_process_lines) but the extraction script has no fine-grained progress
bars worth parsing — it's just pending -> running -> completed/failed.
"""

import json
import subprocess
import threading
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .log_parser import is_error_line
from .models import AttentionRun
from .runner import _iter_process_lines, openfold_subprocess_env


def _collect_meta(run: AttentionRun):
    meta_path = Path(run.output_dir) / "meta.json"
    if meta_path.exists():
        run.meta = json.loads(meta_path.read_text())


def run_attention_extraction(run_id):
    run = AttentionRun.objects.select_related("job").get(id=run_id)
    job = run.job

    output_dir = Path(run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run.status = AttentionRun.Status.RUNNING
    run.started_at = timezone.now()
    run.save()

    cmd = [
        str(settings.OPENFOLD_PYTHON),
        str(settings.OPENFOLD_EXT_DIR / "attention_extraction.py"),
        "--query_json",
        job.query_json_path,
        "--runner_yaml",
        str(settings.OPENFOLD_RUNNER_YAML),
        "--output_dir",
        str(output_dir),
        "--families",
        ",".join(run.layer_families),
    ]

    log_path = Path(run.log_path)
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
    run.pid = process.pid
    run.save(update_fields=["pid"])

    error_lines = []

    try:
        for line in _iter_process_lines(process):
            log_file.write(line + "\n")
            log_file.flush()
            if is_error_line(line):
                error_lines.append(line)

        return_code = process.wait()
    finally:
        log_file.close()

    run.finished_at = timezone.now()

    if return_code == 0:
        run.status = AttentionRun.Status.COMPLETED
        _collect_meta(run)
    else:
        run.status = AttentionRun.Status.FAILED
        run.error_message = "\n".join(error_lines[-40:]) or (
            f"Le process s'est terminé avec le code {return_code}."
        )

    run.save()


def start_attention_extraction(run: AttentionRun):
    thread = threading.Thread(
        target=run_attention_extraction, args=(run.id,), daemon=True
    )
    thread.start()
    return thread
