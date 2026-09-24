"""Starts an OpenFold subprocess and streams its output line by line.

Two subtleties drive this design:

- tqdm progress bars rewrite the same line with '\\r' instead of '\\n', so a
  plain readline() would only see each bar's final state. Output is read one
  character at a time and split on both '\\r' and '\\n'.
- PyTorch Lightning dataloader workers are forked processes that inherit the
  stdout pipe, which can therefore stay open after the main process exits.
  The pipe is read on a daemon thread into a queue, and the generator also
  watches `process.poll()`, so it stops shortly after the real process exits
  instead of waiting forever for an EOF.
"""

import queue
import subprocess
import threading
import time
from collections.abc import Iterator

from django.conf import settings

from .commands import subprocess_env

# Seconds to keep draining the pipe once the process has exited.
DRAIN_TIMEOUT_AFTER_EXIT = 2.0
_END_OF_STREAM = None


def start(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        cwd=str(settings.OPENFOLD_PROJECT_DIR),
        env=subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _pump_lines(pipe, line_queue: queue.Queue) -> None:
    buffer = ""
    while True:
        char = pipe.read(1)
        if char == "":
            break
        if char in ("\n", "\r"):
            if buffer:
                line_queue.put(buffer)
                buffer = ""
        else:
            buffer += char
    if buffer:
        line_queue.put(buffer)
    line_queue.put(_END_OF_STREAM)


def iter_output_lines(process: subprocess.Popen) -> Iterator[str]:
    """Yields logical output lines; guaranteed to stop once `process` exits."""
    line_queue: queue.Queue = queue.Queue()
    threading.Thread(
        target=_pump_lines, args=(process.stdout, line_queue), daemon=True
    ).start()

    exited_at = None
    while True:
        try:
            line = line_queue.get(timeout=0.5)
        except queue.Empty:
            if process.poll() is not None:
                exited_at = exited_at or time.monotonic()
                if time.monotonic() - exited_at > DRAIN_TIMEOUT_AFTER_EXIT:
                    return
            continue

        if line is _END_OF_STREAM:
            return
        yield line
